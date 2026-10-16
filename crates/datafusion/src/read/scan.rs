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
use datafusion::arrow::compute::SortOptions;
use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Statistics, plan_err};
use datafusion::datasource::source::DataSource;
use datafusion::error::Result;
use datafusion::execution::memory_pool::MemoryConsumer;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, PhysicalExpr, PhysicalSortExpr,
};
use datafusion::physical_plan::common::spawn_buffered;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion::physical_plan::sorts::streaming_merge::StreamingMergeBuilder;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayFormatType, Partitioning};
use datafusion::prelude::Expr;
use futures::{Stream, StreamExt, TryStreamExt};
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::expr::Predicate;
use iceberg::scan::{FileScanTask, TableScan};
use iceberg::spec::{
    NullOrder, PrimitiveType, Schema, SortDirection, SortOrder, Transform, Type,
};
use iceberg::table::Table;

use super::predicate::convert_filters_to_predicate;
use crate::{IcebergDataFusionConfig, to_datafusion_error};

/// Manages the scanning process of an Iceberg [`Table`], encapsulating the
/// necessary details and computed properties required for execution planning.
#[derive(Debug, Clone)]
pub struct IcebergDataSource {
    /// A table in the catalog.
    table: Table,
    /// Snapshot of the table to scan.
    snapshot_id: Option<i64>,
    /// Schema returned by this data source after projection.
    schema: ArrowSchemaRef,
    /// Projection column names in the output schema.
    projection: Vec<String>,
    /// Filter to apply to the table scan
    predicate: Option<Predicate>,
    /// Optional limit on the number of rows to return
    fetch: Option<usize>,
    /// The data files to merge in the order the source reports, listed while
    /// the scan was planned. `None` lists the files when the source runs and
    /// reports no order.
    sorted: Option<SortedTasks>,
    metrics: ExecutionPlanMetricsSet,
}

/// Data files that all record the same sort order, and the part of that order
/// the source reports, over the columns of its output.
#[derive(Debug, Clone)]
struct SortedTasks {
    tasks: Vec<FileScanTask>,
    ordering: LexOrdering,
}

impl IcebergDataSource {
    /// Starts building a source that scans `table` and returns the columns of
    /// `schema`, all of them unless [`IcebergDataSourceBuilder::with_projection`]
    /// selects some. Columns are read from Iceberg by name.
    ///
    /// `sorted_tasks` are the data files to merge in the sort order they all
    /// record, which the source then reports to DataFusion. With `None`, the
    /// source lists its files when it runs and reports no order, as a table
    /// provider's source does. A source rebuilt from another, such as in
    /// another process, must be given what that source's
    /// [`sorted_tasks`](Self::sorted_tasks) returns, because a plan built over a
    /// sorted source may rely on its order. [`IcebergDataSourceBuilder::build`]
    /// fails if the tasks don't share a sort order the source can report.
    pub fn builder(
        table: Table,
        schema: ArrowSchemaRef,
        sorted_tasks: Option<Vec<FileScanTask>>,
    ) -> IcebergDataSourceBuilder {
        IcebergDataSourceBuilder {
            table,
            schema,
            projection: None,
            snapshot_id: None,
            predicate: None,
            fetch: None,
            sorted_tasks,
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

    pub fn projection(&self) -> &[String] {
        &self.projection
    }

    pub fn predicate(&self) -> Option<&Predicate> {
        self.predicate.as_ref()
    }

    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    /// The data files the source merges in the order it reports, listed while
    /// the scan was planned, or `None` if it reports no order and lists its
    /// files when it runs.
    pub fn sorted_tasks(&self) -> Option<&[FileScanTask]> {
        self.sorted.as_ref().map(|sorted| sorted.tasks.as_slice())
    }

    /// Lists the source's data files and reports the sort order they all
    /// record, if the session enables `iceberg.execution.preserve_data_ordering`
    /// and they are few enough to merge. Otherwise returns the source unchanged.
    pub(crate) async fn plan_sort_order(self, state: &dyn Session) -> Result<Self> {
        let config = state
            .config()
            .options()
            .extensions
            .get::<IcebergDataFusionConfig>()
            .map(|config| config.execution.clone())
            .unwrap_or_default();
        if !config.preserve_data_ordering {
            return Ok(self);
        }

        let metadata = self.table.metadata();
        let snapshot = match self.snapshot_id {
            Some(snapshot_id) => metadata.snapshot_by_id(snapshot_id),
            None => metadata.current_snapshot(),
        };
        let Some(snapshot) = snapshot else {
            return Ok(self);
        };
        let schema = snapshot.schema(metadata).map_err(to_datafusion_error)?;
        let output = self.schema();
        if !metadata
            .sort_orders_iter()
            .any(|order| lex_ordering(order, &schema, &output).is_some())
        {
            return Ok(self);
        }

        let tasks: Vec<FileScanTask> = build_table_scan(
            &self.table,
            self.snapshot_id,
            Some(self.projection.clone()),
            self.predicate.clone(),
        )?
        .plan_files()
        .await
        .map_err(to_datafusion_error)?
        .take(config.max_merge_files_per_partition.saturating_add(1))
        .try_collect()
        .await
        .map_err(to_datafusion_error)?;
        if tasks.len() > config.max_merge_files_per_partition {
            return Ok(self);
        }
        let mut source = self;
        source.sorted = shared_ordering(&tasks, &output)
            .map(|ordering| SortedTasks { tasks, ordering });
        Ok(source)
    }

    /// Reads each sorted data file as its own stream and merges the streams in
    /// the order the source reports.
    fn merge_sorted(
        &self,
        sorted: &SortedTasks,
        partition: usize,
        context: &TaskContext,
    ) -> Result<SendableRecordBatchStream> {
        // The builder defaults to the reader settings `TableScan::to_arrow` uses.
        let reader = self.table.reader_builder().build();
        let mut streams = sorted
            .tasks
            .iter()
            .map(|task| {
                let tasks = futures::stream::iter([Ok(task.clone())]).boxed();
                let batches = reader
                    .clone()
                    .read(tasks)
                    .map_err(to_datafusion_error)?
                    .stream()
                    .map_err(to_datafusion_error);
                Ok(
                    Box::pin(RecordBatchStreamAdapter::new(self.schema(), batches))
                        as SendableRecordBatchStream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if streams.len() == 1 {
            return Ok(streams.remove(0));
        }

        let reservation = MemoryConsumer::new(format!("IcebergDataSource[{partition}]"))
            .register(&context.runtime_env().memory_pool);
        StreamingMergeBuilder::new()
            .with_streams(
                streams
                    .into_iter()
                    .map(|stream| spawn_buffered(stream, 1))
                    .collect(),
            )
            .with_schema(self.schema())
            .with_expressions(&sorted.ordering)
            .with_metrics(BaselineMetrics::new(&self.metrics, partition))
            .with_batch_size(context.session_config().batch_size())
            .with_fetch(self.fetch)
            .with_reservation(reservation)
            .build()
    }
}

/// Builds an [`IcebergDataSource`]. Created by [`IcebergDataSource::builder`].
///
/// A table provider passes the table's schema and selects columns with
/// [`Self::with_projection`]. To rebuild a source, such as in another process,
/// pass its [`IcebergDataSource::schema`], which is already projected, and its
/// [`IcebergDataSource::sorted_tasks`] to [`IcebergDataSource::builder`], and no
/// projection.
#[derive(Debug, Clone)]
pub struct IcebergDataSourceBuilder {
    table: Table,
    schema: ArrowSchemaRef,
    projection: Option<Vec<usize>>,
    snapshot_id: Option<i64>,
    predicate: Option<Predicate>,
    fetch: Option<usize>,
    sorted_tasks: Option<Vec<FileScanTask>>,
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
    /// Returns an error if the projection is out of bounds for the schema, or
    /// if the sorted tasks do not all record a sort order the source can
    /// report over its schema, or are read with a snapshot schema whose
    /// columns do not match it.
    pub fn build(self) -> Result<IcebergDataSource> {
        let schema = match &self.projection {
            None => self.schema,
            Some(projection) => Arc::new(self.schema.project(projection)?),
        };
        let projection = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let sorted = match self.sorted_tasks {
            None => None,
            Some(tasks) => {
                let Some(ordering) = shared_ordering(&tasks, &schema) else {
                    return plan_err!(
                        "IcebergDataSource cannot report an order for the given data \
                         files: they do not all record the same sort order, or are \
                         read with a schema that does not match the source's"
                    );
                };
                Some(SortedTasks { tasks, ordering })
            }
        };
        Ok(IcebergDataSource {
            table: self.table,
            snapshot_id: self.snapshot_id,
            schema,
            projection,
            predicate: self.predicate,
            fetch: self.fetch,
            sorted,
            metrics: ExecutionPlanMetricsSet::new(),
        })
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
        let stream: Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>> =
            match &self.sorted {
                Some(sorted) => self.merge_sorted(sorted, partition, &context)?,
                None => {
                    let fut = get_batch_stream(
                        self.table.clone(),
                        self.snapshot_id,
                        Some(self.projection.clone()),
                        self.predicate.clone(),
                    );
                    Box::pin(futures::stream::once(fut).try_flatten())
                }
            };

        // Apply the fetch if specified.
        let limited_stream: Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>> =
            if let Some(limit) = self.fetch {
                let mut remaining = limit;
                Box::pin(stream.try_filter_map(move |batch| {
                    futures::future::ready(if remaining == 0 {
                        Ok(None)
                    } else if batch.num_rows() <= remaining {
                        remaining -= batch.num_rows();
                        Ok(Some(batch))
                    } else {
                        let limited_batch = batch.slice(0, remaining);
                        remaining = 0;
                        Ok(Some(limited_batch))
                    })
                }))
            } else {
                Box::pin(stream)
            };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            limited_stream,
        )))
    }

    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "format=iceberg",)?;
        write!(f, ", projection=[{}]", self.projection.join(", "))?;
        if let Some(predicate) = &self.predicate {
            write!(f, ", predicate={predicate}")?;
        }
        if let Some(sorted) = &self.sorted {
            write!(
                f,
                ", output_ordering=[{}], files={}",
                sorted.ordering,
                sorted.tasks.len()
            )?;
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
        match &self.sorted {
            Some(sorted) => EquivalenceProperties::new_with_orderings(
                self.schema.clone(),
                [sorted.ordering.clone()],
            ),
            None => EquivalenceProperties::new(self.schema.clone()),
        }
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        Ok(Arc::new(Statistics::new_unknown(&self.schema)))
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
) -> Result<Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>> {
    let table_scan = build_table_scan(&table, snapshot_id, column_names, predicate)?;

    let stream = table_scan
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?
        .map_err(to_datafusion_error);
    Ok(Box::pin(stream))
}

/// Builds the [`TableScan`] of the columns `column_names` of a snapshot of
/// `table`, or of its current snapshot, filtered by `predicate`.
fn build_table_scan(
    table: &Table,
    snapshot_id: Option<i64>,
    column_names: Option<Vec<String>>,
    predicate: Option<Predicate>,
) -> Result<TableScan> {
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
    scan_builder.build().map_err(to_datafusion_error)
}

/// The order of the columns of `output` that every one of `tasks` is sorted
/// in, or `None` if they record different sort orders, one records none, or
/// the schema they are read with does not match `output`.
fn shared_ordering(tasks: &[FileScanTask], output: &ArrowSchema) -> Option<LexOrdering> {
    let first = tasks.first()?;
    let order = first.sort_order()?;
    if !tasks.iter().all(|task| task.sort_order() == Some(order)) {
        return None;
    }
    if !reads_as_declared(first.schema(), output) {
        return None;
    }
    lex_ordering(order, first.schema(), output)
}

/// Whether reading `schema`, the snapshot schema the tasks are read with,
/// yields every column of `output` with the type it declares, and no nulls
/// where it declares none. A provider's schema can be older than the
/// snapshot's, and DataFusion ignores the null order of a non-nullable
/// column, while the merge requires its input to match `output`.
fn reads_as_declared(schema: &Schema, output: &ArrowSchema) -> bool {
    let Ok(read) = schema_to_arrow_schema(schema) else {
        return false;
    };
    output.fields().iter().all(|field| {
        read.field_with_name(field.name()).is_ok_and(|read| {
            read.data_type() == field.data_type()
                && (field.is_nullable() || !read.is_nullable())
        })
    })
}

/// The longest leading part of `order` that DataFusion can merge and report as
/// an order of the columns of `output`, or `None` if that is empty. `schema`
/// is the Iceberg schema the scan reads.
fn lex_ordering(
    order: &SortOrder,
    schema: &Schema,
    output: &ArrowSchema,
) -> Option<LexOrdering> {
    let sort_exprs = order.fields.iter().map_while(|field| {
        if field.transform != Transform::Identity {
            return None;
        }
        let source = schema.as_struct().field_by_id(field.source_id)?;
        if !orders_like_writers(&source.field_type) {
            return None;
        }
        let index = output.index_of(&source.name).ok()?;
        Some(PhysicalSortExpr::new(
            Arc::new(Column::new(&source.name, index)),
            SortOptions {
                descending: field.direction == SortDirection::Descending,
                nulls_first: field.null_order == NullOrder::First,
            },
        ))
    });
    LexOrdering::new(sort_exprs)
}

/// Whether DataFusion orders values of `field_type` as the engines that write
/// sorted Iceberg files do. For floating point, DataFusion uses the order the
/// spec defines (<https://iceberg.apache.org/spec/#sort-orders>), with -0.0
/// before 0.0 and negative NaN first. Spark treats the zeros as equal and
/// every NaN as the largest value, so files it writes may not be sorted in the
/// spec's order. The spec leaves the order of UUIDs undefined
/// (apache/iceberg#14216).
fn orders_like_writers(field_type: &Type) -> bool {
    matches!(
        field_type,
        Type::Primitive(
            PrimitiveType::Boolean
                | PrimitiveType::Int
                | PrimitiveType::Long
                | PrimitiveType::Decimal { .. }
                | PrimitiveType::Date
                | PrimitiveType::Time
                | PrimitiveType::Timestamp
                | PrimitiveType::Timestamptz
                | PrimitiveType::TimestampNs
                | PrimitiveType::TimestamptzNs
                | PrimitiveType::String
                | PrimitiveType::Fixed(_)
                | PrimitiveType::Binary
        )
    )
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{NestedField, SortField, StructType};

    use super::*;

    fn schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                    .into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
                    .into(),
                NestedField::optional(3, "ts", Type::Primitive(PrimitiveType::Timestamp))
                    .into(),
                NestedField::optional(
                    4,
                    "amount",
                    Type::Primitive(PrimitiveType::Decimal {
                        precision: 10,
                        scale: 2,
                    }),
                )
                .into(),
                NestedField::optional(5, "price", Type::Primitive(PrimitiveType::Double))
                    .into(),
                NestedField::optional(6, "ratio", Type::Primitive(PrimitiveType::Float))
                    .into(),
                NestedField::optional(7, "uid", Type::Primitive(PrimitiveType::Uuid))
                    .into(),
                NestedField::optional(
                    8,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(
                            9,
                            "inner",
                            Type::Primitive(PrimitiveType::Int),
                        )
                        .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap()
    }

    fn field(
        source_id: i32,
        transform: Transform,
        direction: SortDirection,
        null_order: NullOrder,
    ) -> SortField {
        SortField::builder()
            .source_id(source_id)
            .transform(transform)
            .direction(direction)
            .null_order(null_order)
            .build()
    }

    fn asc(source_id: i32) -> SortField {
        field(
            source_id,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        )
    }

    /// The order reported for `fields` over the columns `projection` of
    /// [`schema`], rendered as DataFusion displays it.
    fn ordering(fields: Vec<SortField>, projection: &[&str]) -> Option<String> {
        let schema = schema();
        let mut builder = SortOrder::builder();
        builder.with_order_id(1);
        for field in fields {
            builder.with_sort_field(field);
        }
        let order = builder.build(&schema).unwrap();
        let arrow = schema_to_arrow_schema(&schema).unwrap();
        let indices: Vec<usize> = projection
            .iter()
            .map(|name| arrow.index_of(name).unwrap())
            .collect();
        let output = arrow.project(&indices).unwrap();
        lex_ordering(&order, &schema, &output).map(|ordering| ordering.to_string())
    }

    const ALL: &[&str] = &["id", "name", "ts", "amount", "price", "ratio", "uid", "s"];

    #[test]
    fn test_identity_fields_keep_direction_and_null_order() {
        let fields = vec![
            field(
                1,
                Transform::Identity,
                SortDirection::Ascending,
                NullOrder::First,
            ),
            field(
                2,
                Transform::Identity,
                SortDirection::Ascending,
                NullOrder::Last,
            ),
            field(
                3,
                Transform::Identity,
                SortDirection::Descending,
                NullOrder::First,
            ),
            field(
                4,
                Transform::Identity,
                SortDirection::Descending,
                NullOrder::Last,
            ),
        ];
        assert_eq!(
            ordering(fields, ALL).as_deref(),
            Some("id@0 ASC, name@1 ASC NULLS LAST, ts@2 DESC, amount@3 DESC NULLS LAST")
        );
    }

    #[test]
    fn test_ordering_stops_at_non_identity_transform() {
        for transform in [Transform::Bucket(4), Transform::Truncate(2), Transform::Day] {
            let source_id = if transform == Transform::Day { 3 } else { 2 };
            let fields = vec![
                asc(1),
                field(
                    source_id,
                    transform,
                    SortDirection::Ascending,
                    NullOrder::First,
                ),
                asc(4),
            ];
            assert_eq!(
                ordering(fields, ALL).as_deref(),
                Some("id@0 ASC"),
                "{transform}"
            );
        }
        let fields = vec![field(
            1,
            Transform::Bucket(4),
            SortDirection::Ascending,
            NullOrder::First,
        )];
        assert_eq!(ordering(fields, ALL), None);
    }

    #[test]
    fn test_ordering_stops_at_floating_point_and_uuid() {
        for source_id in [5, 6, 7] {
            assert_eq!(
                ordering(vec![asc(1), asc(source_id), asc(2)], ALL).as_deref(),
                Some("id@0 ASC")
            );
            assert_eq!(
                ordering(vec![asc(1), asc(source_id)], ALL).as_deref(),
                Some("id@0 ASC")
            );
            assert_eq!(ordering(vec![asc(source_id), asc(1)], ALL), None);
        }
    }

    #[test]
    fn test_ordering_stops_at_nested_field() {
        assert_eq!(
            ordering(vec![asc(1), asc(9), asc(2)], ALL).as_deref(),
            Some("id@0 ASC")
        );
        assert_eq!(ordering(vec![asc(9)], ALL), None);
    }

    #[test]
    fn test_ordering_stops_at_unprojected_field() {
        assert_eq!(
            ordering(vec![asc(1), asc(2), asc(3)], &["ts", "id"]).as_deref(),
            Some("id@1 ASC")
        );
        assert_eq!(ordering(vec![asc(2), asc(1)], &["id"]), None);
    }

    #[test]
    fn test_ordering_stops_at_field_missing_from_schema() {
        let mut builder = SortOrder::builder();
        builder.with_order_id(1);
        builder.with_sort_field(asc(1));
        builder.with_sort_field(asc(42));
        builder.with_sort_field(asc(2));
        let order = builder.build_unbound().unwrap();
        let schema = schema();
        let output = schema_to_arrow_schema(&schema).unwrap();
        assert_eq!(
            lex_ordering(&order, &schema, &output).map(|o| o.to_string()),
            Some("id@0 ASC".to_string())
        );
    }
}
