// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::pin::Pin;
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::common::Statistics;
use datafusion::common::stats::Precision;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::datasource::source::DataSource;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::limit::LimitStream;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayFormatType, Partitioning};
use datafusion::prelude::Expr;
use futures::{Stream, TryStreamExt};
use iceberg::expr::{Bind, Predicate};
use iceberg::spec::SchemaRef;
use iceberg::table::Table;

use super::predicate::convert_filters_to_predicate;
use crate::to_datafusion_error;

/// Snapshot summary field from the spec's "Optional Snapshot Summary Fields".
const TOTAL_RECORDS: &str = "total-records";

/// Manages the scanning process of an Iceberg [`Table`], encapsulating the
/// necessary details and computed properties required for execution planning.
#[derive(Debug, Clone)]
pub struct IcebergDataSource {
    /// A table in the catalog.
    table: Table,
    /// Snapshot of the table to scan.
    snapshot_id: Option<i64>,
    /// Schema returned by this data source after projection. Its columns are
    /// read from Iceberg by name.
    schema: ArrowSchemaRef,
    /// Filter to apply to the table scan
    predicate: Option<Predicate>,
    /// Optional limit on the number of rows to return
    fetch: Option<usize>,
    /// Execution metrics, reported by `EXPLAIN ANALYZE`.
    metrics: ExecutionPlanMetricsSet,
}

impl IcebergDataSource {
    /// Starts building a source that scans `table` and returns the columns of
    /// `schema`, all of them unless [`IcebergDataSourceBuilder::with_projection`]
    /// selects some. Columns are read from Iceberg by name.
    pub fn builder(table: Table, schema: ArrowSchemaRef) -> IcebergDataSourceBuilder {
        IcebergDataSourceBuilder {
            table,
            schema,
            projection: None,
            snapshot_id: None,
            predicate: None,
            fetch: None,
        }
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    pub fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    pub fn predicate(&self) -> Option<&Predicate> {
        self.predicate.as_ref()
    }

    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    /// Estimates the rows the scan returns before its predicate and fetch,
    /// from the snapshot summary.
    ///
    /// `total-records` counts rows before deletes, so the estimate is an upper
    /// bound. Deletes aren't subtracted, because the spec lets
    /// `total-position-deletes` count more rows than the deletes remove. A
    /// position delete file can stay live after the data files it references
    /// are removed. A deletion vector repeats the deletes of position delete
    /// files that can stay live beside it. An underestimate can make join
    /// selection collect a large table as the build side. Summary fields are
    /// optional, so a missing or invalid `total-records` leaves the estimate
    /// unknown rather than failing the query.
    fn estimated_num_rows(&self) -> Precision<usize> {
        let metadata = self.table.metadata();
        let snapshot = match self.snapshot_id {
            Some(snapshot_id) => metadata.snapshot_by_id(snapshot_id),
            None => match metadata.current_snapshot() {
                Some(snapshot) => Some(snapshot),
                // The scan of a table with no current snapshot reads nothing.
                None => return Precision::Exact(0),
            },
        };
        // The scan fails on a snapshot id that doesn't exist.
        let Some(snapshot) = snapshot else {
            return Precision::Absent;
        };
        match snapshot
            .summary()
            .additional_properties
            .get(TOTAL_RECORDS)
            .and_then(|records| records.parse::<usize>().ok())
        {
            Some(records) => Precision::Inexact(records),
            None => Precision::Absent,
        }
    }

    /// The names of the columns this source reads, in output order.
    fn column_names(&self) -> Vec<String> {
        self.schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }
}

/// Builds an [`IcebergDataSource`]. Created by [`IcebergDataSource::builder`].
///
/// A table provider passes the table's schema and selects columns with
/// [`Self::with_projection`]. To rebuild a source, such as in another process,
/// pass its [`IcebergDataSource::schema`], which is already projected, and no
/// projection.
#[derive(Debug, Clone)]
pub struct IcebergDataSourceBuilder {
    table: Table,
    schema: ArrowSchemaRef,
    projection: Option<Vec<usize>>,
    snapshot_id: Option<i64>,
    predicate: Option<Predicate>,
    fetch: Option<usize>,
}

impl IcebergDataSourceBuilder {
    /// Returns only the columns of the schema at these indices, or all of them
    /// if `None` (the default). Indices are into the schema passed to
    /// [`IcebergDataSource::builder`].
    pub fn with_projection(mut self, projection: Option<&Vec<usize>>) -> Self {
        self.projection = projection.cloned();
        self
    }

    /// Scans this snapshot, or the table's current one if `None` (the default).
    pub fn with_snapshot_id(mut self, snapshot_id: Option<i64>) -> Self {
        self.snapshot_id = snapshot_id;
        self
    }

    /// Filters the scan with `predicate`, or not at all if `None` (the default).
    ///
    /// The top-level conjuncts of `predicate` that do not bind to the schema of the
    /// snapshot being scanned are dropped in [`Self::build`], as Iceberg would fail
    /// the scan on them. A null test on a struct, list or map column is one:
    /// Iceberg binds references to primitive fields only.
    pub fn with_predicate(mut self, predicate: Option<Predicate>) -> Self {
        self.predicate = predicate;
        self
    }

    /// Filters the scan with the parts of `filters` that convert to an Iceberg
    /// predicate. The rest are dropped, so the scan may return rows that
    /// `filters` reject.
    pub fn with_filters(self, filters: &[Expr]) -> Self {
        self.with_predicate(convert_filters_to_predicate(filters))
    }

    /// Stops after `fetch` rows, or reads every row if `None` (the default).
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        self
    }

    /// Builds the source, which reads the columns of its projected schema by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the projection is out of bounds for the schema, or if
    /// the scanned snapshot's schema cannot be read.
    pub fn build(self) -> Result<IcebergDataSource> {
        let predicate = match self.predicate {
            Some(predicate) => bindable_conjuncts(
                predicate,
                &scan_schema(&self.table, self.snapshot_id)?,
            ),
            None => None,
        };
        let schema = match &self.projection {
            None => self.schema,
            Some(projection) => Arc::new(self.schema.project(projection)?),
        };
        Ok(IcebergDataSource {
            table: self.table,
            snapshot_id: self.snapshot_id,
            schema,
            predicate,
            fetch: self.fetch,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

/// The schema the scan binds its predicate to: that of the snapshot it reads, or
/// the table's current schema if the snapshot is missing (the scan reports that).
fn scan_schema(table: &Table, snapshot_id: Option<i64>) -> Result<SchemaRef> {
    let metadata = table.metadata();
    let snapshot = match snapshot_id {
        Some(id) => metadata.snapshot_by_id(id),
        None => metadata.current_snapshot(),
    };
    match snapshot {
        Some(snapshot) => snapshot.schema(metadata).map_err(to_datafusion_error),
        None => Ok(metadata.current_schema().clone()),
    }
}

/// The top-level conjuncts of `predicate` that bind to `schema`, combined with
/// `AND`, or `None` if none do.
///
/// Dropping a conjunct only lets the scan return more rows, and every filter is
/// pushed down as `Inexact`, so DataFusion still applies it above the scan.
fn bindable_conjuncts(predicate: Predicate, schema: &SchemaRef) -> Option<Predicate> {
    let mut conjuncts = Vec::new();
    split_conjunction(&predicate, &mut conjuncts);
    conjuncts
        .into_iter()
        .filter(|conjunct| conjunct.bind(schema.clone(), true).is_ok())
        .cloned()
        .reduce(Predicate::and)
}

fn split_conjunction<'a>(predicate: &'a Predicate, conjuncts: &mut Vec<&'a Predicate>) {
    match predicate {
        Predicate::And(and) => {
            for input in and.inputs() {
                split_conjunction(input, conjuncts);
            }
        }
        _ => conjuncts.push(predicate),
    }
}

impl DataSource for IcebergDataSource {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn open(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let fut = get_batch_stream(
            self.table.clone(),
            self.snapshot_id,
            Some(self.column_names()),
            self.predicate.clone(),
            context
                .session_config()
                .options()
                .execution
                .parquet
                .enable_page_index,
            context.session_config().batch_size(),
        );
        let stream = futures::stream::once(fut).try_flatten();
        let stream = Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), stream));
        // With no fetch, the limit passes every row through, still recording
        // the output metrics.
        Ok(Box::pin(LimitStream::new(
            stream,
            0,
            self.fetch,
            BaselineMetrics::new(&self.metrics, partition),
        )))
    }

    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "format=iceberg",)?;
        write!(f, ", projection=[{}]", self.column_names().join(", "))?;
        if let Some(predicate) = &self.predicate {
            write!(f, ", predicate={predicate}")?;
        }
        if let Some(fetch) = self.fetch {
            write!(f, ", fetch={fetch}")?;
        }
        Ok(())
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(self.schema.clone())
    }

    // The source has one output partition (see `output_partitioning`), so a
    // partition's statistics are the scan's.
    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        let mut statistics = Statistics::new_unknown(&self.schema);
        statistics.num_rows = self.estimated_num_rows();
        // Iceberg's `total-files-size` is compressed and covers every column
        // and delete file, so estimate the in-memory size of the projected
        // columns from their widths, as DataFusion's Parquet source does for
        // fixed-width columns.
        statistics.calculate_total_byte_size(&self.schema);
        Ok(Arc::new(statistics.with_fetch(self.fetch, 0, 1)?))
    }

    fn with_fetch(&self, fetch: Option<usize>) -> Option<Arc<dyn DataSource>> {
        let mut source = self.clone();
        source.fetch = fetch;
        Some(Arc::new(source))
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn metrics(&self) -> ExecutionPlanMetricsSet {
        self.metrics.clone()
    }

    fn try_swapping_with_projection(
        &self,
        _projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn DataSource>>> {
        Ok(None)
    }
}

/// Asynchronously retrieves a stream of [`RecordBatch`] instances
/// from a given table.
///
/// This function initializes a [`TableScan`], builds it,
/// and then converts it into a stream of Arrow [`RecordBatch`]es.
async fn get_batch_stream(
    table: Table,
    snapshot_id: Option<i64>,
    column_names: Option<Vec<String>>,
    predicate: Option<Predicate>,
    enable_page_index: bool,
    batch_size: usize,
) -> Result<Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>> {
    let scan_builder = match snapshot_id {
        Some(snapshot_id) => table.scan().snapshot_id(snapshot_id),
        None => table.scan(),
    };

    let mut scan_builder = match column_names {
        Some(column_names) => scan_builder.select(column_names),
        None => scan_builder.select_all(),
    };
    if let Some(predicate) = predicate {
        scan_builder = scan_builder.with_filter(predicate);
    }
    let table_scan = scan_builder
        .with_row_selection_enabled(enable_page_index)
        .with_batch_size(Some(batch_size))
        .build()
        .map_err(to_datafusion_error)?;

    let stream = table_scan
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?
        .map_err(to_datafusion_error);
    Ok(Box::pin(stream))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::common::Statistics;
    use datafusion::common::stats::Precision;
    use datafusion::datasource::source::DataSource;
    use iceberg::io::FileIO;
    use iceberg::spec::{
        FormatVersion, MAIN_BRANCH, NestedField, Operation, PartitionSpec, PrimitiveType,
        Schema, Snapshot, SortOrder, Summary, TableMetadataBuilder, Type,
    };
    use iceberg::table::Table;
    use iceberg::{Runtime, TableIdent};

    use super::IcebergDataSource;

    /// Returns a table with one snapshot per summary, oldest first, with ids
    /// counting up from 1. The last snapshot is the current one.
    fn table_with_snapshot_summaries(summaries: &[&[(&str, &str)]]) -> Table {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long))
                    .into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
                    .into(),
            ])
            .build()
            .unwrap();
        let partition_spec = PartitionSpec::builder(schema.clone()).build().unwrap();
        let sort_order = SortOrder::builder().build(&schema).unwrap();
        let mut builder = TableMetadataBuilder::new(
            schema,
            partition_spec,
            sort_order,
            "/test/table".to_string(),
            FormatVersion::V2,
            HashMap::new(),
        )
        .unwrap();
        for (snapshot_id, summary) in (1..).zip(summaries) {
            let snapshot = Snapshot::builder()
                .with_snapshot_id(snapshot_id)
                .with_parent_snapshot_id((snapshot_id > 1).then_some(snapshot_id - 1))
                .with_sequence_number(snapshot_id)
                .with_timestamp_ms(1_700_000_000_000 + snapshot_id)
                .with_manifest_list(format!(
                    "/test/table/metadata/snap-{snapshot_id}.avro"
                ))
                .with_summary(Summary {
                    operation: Operation::Append,
                    additional_properties: summary
                        .iter()
                        .map(|(key, value)| (key.to_string(), value.to_string()))
                        .collect(),
                })
                .with_schema_id(0)
                .build();
            builder = builder.set_branch_snapshot(snapshot, MAIN_BRANCH).unwrap();
        }

        Table::builder()
            .metadata(builder.build().unwrap().metadata)
            .identifier(TableIdent::from_strs(["test", "table"]).unwrap())
            .file_io(FileIO::new_with_fs())
            .metadata_location("/test/metadata.json")
            .runtime(Runtime::current())
            .build()
            .unwrap()
    }

    /// Statistics of a scan of `table`'s fixed-width `id` column.
    fn id_scan_statistics(
        table: Table,
        snapshot_id: Option<i64>,
        fetch: Option<usize>,
    ) -> Statistics {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let source = IcebergDataSource::builder(table, schema)
            .with_projection(Some(&vec![0]))
            .with_snapshot_id(snapshot_id)
            .with_fetch(fetch)
            .build()
            .unwrap();
        Arc::unwrap_or_clone(source.partition_statistics(None).unwrap())
    }

    #[tokio::test]
    async fn reports_no_rows_for_a_table_without_snapshots() {
        let statistics =
            id_scan_statistics(table_with_snapshot_summaries(&[]), None, None);
        assert_eq!(statistics.num_rows, Precision::Exact(0));
        assert_eq!(statistics.total_byte_size, Precision::Exact(0));
    }

    #[tokio::test]
    async fn estimates_rows_from_the_snapshot_summary() {
        let table = table_with_snapshot_summaries(&[&[("total-records", "100")]]);
        let statistics = id_scan_statistics(table, None, None);
        assert_eq!(statistics.num_rows, Precision::Inexact(100));
        assert_eq!(statistics.total_byte_size, Precision::Inexact(800));
    }

    #[tokio::test]
    async fn ignores_deletes() {
        for (position_deletes, equality_deletes) in [("30", "50"), ("200", "0")] {
            let table = table_with_snapshot_summaries(&[&[
                ("total-records", "100"),
                ("total-position-deletes", position_deletes),
                ("total-equality-deletes", equality_deletes),
            ]]);
            let statistics = id_scan_statistics(table, None, None);
            assert_eq!(
                statistics.num_rows,
                Precision::Inexact(100),
                "{position_deletes} position deletes"
            );
        }
    }

    #[tokio::test]
    async fn reports_unknown_rows_without_a_valid_record_count() {
        for summary in [
            &[][..],
            &[("total-records", "-1")],
            &[("total-records", "many")],
            // u64::MAX + 1
            &[("total-records", "18446744073709551616")],
        ] {
            let table = table_with_snapshot_summaries(&[summary]);
            let statistics = id_scan_statistics(table, None, None);
            assert_eq!(statistics.num_rows, Precision::Absent, "{summary:?}");
            assert_eq!(statistics.total_byte_size, Precision::Absent, "{summary:?}");
        }
    }

    #[tokio::test]
    async fn reports_unknown_rows_for_a_missing_snapshot() {
        let table = table_with_snapshot_summaries(&[&[("total-records", "100")]]);
        let statistics = id_scan_statistics(table, Some(2), None);
        assert_eq!(statistics.num_rows, Precision::Absent);
    }

    #[tokio::test]
    async fn uses_the_summary_of_the_scanned_snapshot() {
        let table = table_with_snapshot_summaries(&[
            &[("total-records", "100")],
            &[("total-records", "200")],
        ]);
        assert_eq!(
            id_scan_statistics(table.clone(), Some(1), None).num_rows,
            Precision::Inexact(100)
        );
        assert_eq!(
            id_scan_statistics(table, None, None).num_rows,
            Precision::Inexact(200)
        );
    }

    #[tokio::test]
    async fn caps_rows_at_the_fetch() {
        let table = table_with_snapshot_summaries(&[&[("total-records", "100")]]);
        let statistics = id_scan_statistics(table, None, Some(10));
        assert_eq!(statistics.num_rows, Precision::Inexact(10));
        assert_eq!(statistics.total_byte_size, Precision::Inexact(80));
    }

    #[tokio::test]
    async fn omits_the_byte_size_of_variable_width_columns() {
        let table = table_with_snapshot_summaries(&[&[("total-records", "100")]]);
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let source = IcebergDataSource::builder(table, schema).build().unwrap();
        let statistics = source.partition_statistics(None).unwrap();
        assert_eq!(statistics.num_rows, Precision::Inexact(100));
        assert_eq!(statistics.total_byte_size, Precision::Absent);
    }
}
