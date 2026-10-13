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
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Statistics, internal_err};
use datafusion::datasource::source::DataSource;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayFormatType, Partitioning};
use datafusion::prelude::Expr;
use futures::{Stream, StreamExt, TryStreamExt};
use iceberg::arrow::ArrowReader;
use iceberg::expr::Predicate;
use iceberg::scan::TableScan;
use iceberg::table::Table;

use super::file_planning::PlannedFileScan;
use super::predicate::convert_filters_to_predicate;
use crate::to_datafusion_error;

// A reader is created once per eager scan. Its clones share the delete loader's
// cache while each read() call still owns its own scan metrics.
#[derive(Clone)]
struct ScanReader(ArrowReader);

impl std::fmt::Debug for ScanReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ScanReader")
    }
}

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
    planned_files: Option<Arc<PlannedFileScan>>,
    output_partitions: usize,
    reader: Option<Arc<ScanReader>>,
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

    /// Plan file tasks during physical planning; read them only at execution.
    pub(crate) async fn with_planned_files(
        mut self,
        context: Arc<TaskContext>,
        max_files: usize,
        target_partitions: usize,
    ) -> Result<Self> {
        let scan = build_table_scan(
            &self.table,
            self.snapshot_id,
            Some(&self.projection),
            self.predicate.as_ref(),
        )?;
        let planned = PlannedFileScan::collect(scan, &context, max_files).await?;
        let partitions = planned.tasks().len().max(1).min(target_partitions.max(1));

        // Table::reader_builder retains the table's FileIO and custom runtime.
        // This scan currently changes only snapshot, projection, and predicate;
        // those values are already carried by FileScanTask. A future scan-specific
        // reader setting must be applied here and in TableScan::to_arrow().
        let builder = self.table.reader_builder();
        let builder = if partitions > 1 {
            builder.with_data_file_concurrency_limit(1)
        } else {
            builder
        };
        self.reader = Some(Arc::new(ScanReader(builder.build())));
        self.output_partitions = partitions;
        self.planned_files = Some(Arc::new(planned));
        Ok(self)
    }

    /// Planned, ungrouped tasks, if file discovery was enabled for this scan.
    /// A distributed consumer may assign these tasks to its own workers instead
    /// of using the local output partitions of this source.
    pub fn planned_files(&self) -> Option<&Arc<PlannedFileScan>> {
        self.planned_files.as_ref()
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
}

/// Builds an [`IcebergDataSource`]. Created by [`IcebergDataSource::builder`].
///
/// A table provider passes the table's schema and selects columns with
/// [`Self::with_projection`]. To rebuild a source, such as in another process,
/// pass its [`IcebergDataSource::schema`], which is already projected, and no
/// projection. This rebuilds a lazy single-partition source; planned file tasks
/// must be transferred separately when reconstructing a distributed scan.
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
    /// Returns an error if the projection is out of bounds for the schema.
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
        Ok(IcebergDataSource {
            table: self.table,
            snapshot_id: self.snapshot_id,
            schema,
            projection,
            predicate: self.predicate,
            fetch: self.fetch,
            planned_files: None,
            output_partitions: 1,
            reader: None,
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
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let stream: Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>> =
            if let Some(planned) = &self.planned_files {
                if partition >= self.output_partitions {
                    return internal_err!(
                        "Iceberg scan partition {partition} does not exist"
                    );
                }
                let Some(reader) = &self.reader else {
                    return internal_err!("Planned Iceberg scan has no reader");
                };
                let planned = Arc::clone(planned);
                let partitions = self.output_partitions;
                let tasks = futures::stream::iter(
                    (partition..planned.tasks().len()).step_by(partitions),
                )
                .map(move |index| {
                    Ok(Arc::unwrap_or_clone(Arc::clone(&planned.tasks()[index])))
                });
                Box::pin(
                    reader
                        .0
                        .clone()
                        .read(Box::pin(tasks))
                        .map_err(to_datafusion_error)?
                        .stream()
                        .map_err(to_datafusion_error),
                )
            } else {
                let fut = get_batch_stream(
                    self.table.clone(),
                    self.snapshot_id,
                    Some(self.projection.clone()),
                    self.predicate.clone(),
                );
                Box::pin(futures::stream::once(fut).try_flatten())
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
        if let Some(fetch) = self.fetch {
            write!(f, ", fetch={fetch}")?;
        }
        if let Some(planned) = &self.planned_files {
            write!(f, ", planned_files={}", planned.tasks().len())?;
        }
        Ok(())
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.output_partitions)
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(self.schema.clone())
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
    let table_scan = build_table_scan(
        &table,
        snapshot_id,
        column_names.as_deref(),
        predicate.as_ref(),
    )?;

    let stream = table_scan
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?
        .map_err(to_datafusion_error);
    Ok(Box::pin(stream))
}

fn build_table_scan(
    table: &Table,
    snapshot_id: Option<i64>,
    column_names: Option<&[String]>,
    predicate: Option<&Predicate>,
) -> Result<TableScan> {
    let builder = match snapshot_id {
        Some(snapshot_id) => table.scan().snapshot_id(snapshot_id),
        None => table.scan(),
    };
    let mut builder = match column_names {
        Some(columns) => builder.select(columns),
        None => builder.select_all(),
    };
    if let Some(predicate) = predicate {
        builder = builder.with_filter(predicate.clone());
    }
    builder.build().map_err(to_datafusion_error)
}
