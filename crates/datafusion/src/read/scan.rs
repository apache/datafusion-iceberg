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

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Statistics, internal_err};
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
use futures::TryStreamExt;
use iceberg::arrow::ArrowReader;
use iceberg::expr::Predicate;
use iceberg::io::FileIO;
use iceberg::table::Table;

use super::predicate::convert_filters_to_predicate;
use super::reader::ScanReader;
use super::task_source::{FileScanTaskSource, TableScanTaskSource};
use crate::to_datafusion_error;

/// Reads Iceberg files supplied by a [`FileScanTaskSource`].
///
/// The default [`TableScanTaskSource`] discovers files from a table when executed.
/// Custom task sources can deliver the same native tasks without a table or
/// catalog; file reading, deletes and per-partition fetch remain here.
#[derive(Debug)]
pub struct IcebergDataSource<T: FileScanTaskSource = TableScanTaskSource> {
    task_source: T,
    schema: ArrowSchemaRef,
    reader: ScanReader,
    fetch: Option<usize>,
    metrics: ExecutionPlanMetricsSet,
}

impl<T: FileScanTaskSource> Clone for IcebergDataSource<T> {
    fn clone(&self) -> Self {
        Self {
            task_source: self.task_source.clone(),
            schema: self.schema.clone(),
            reader: self.reader.clone(),
            fetch: self.fetch,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
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
            reader: None,
        }
    }

    pub fn table(&self) -> &Table {
        self.task_source.table()
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.task_source.snapshot_id()
    }

    pub fn projection(&self) -> &[String] {
        self.task_source.projection()
    }

    pub fn predicate(&self) -> Option<&Predicate> {
        self.task_source.predicate()
    }
}

impl<T: FileScanTaskSource> IcebergDataSource<T> {
    /// Creates a reader over tasks supplied without a table or catalog.
    ///
    /// `schema` must match the tasks' projected columns. `file_io` must describe
    /// the same storage configuration used by `reader`; it is retained for
    /// consumers that reconstruct this source in another process. Reader options
    /// are used as supplied, rather than replaced by session defaults. Clones of
    /// this source share the reader's cached delete-file loader.
    ///
    /// `fetch` limits rows per local partition, not across the whole source.
    /// Construction performs no discovery or file I/O.
    ///
    /// # Errors
    ///
    /// Returns an error if the task source has no partitions.
    pub fn from_task_source(
        schema: ArrowSchemaRef,
        task_source: T,
        file_io: FileIO,
        reader: ArrowReader,
        fetch: Option<usize>,
    ) -> Result<Self> {
        validate_partition_count(&task_source)?;
        Ok(Self {
            task_source,
            schema,
            reader: ScanReader::Provided { file_io, reader },
            fetch,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Replaces task delivery while preserving schema, reader options and fetch.
    ///
    /// The replacement must supply tasks with the same projection and storage
    /// access requirements. It may change the number of local partitions. This
    /// operation does not discover files or open task streams.
    ///
    /// # Errors
    ///
    /// Returns an error if the replacement has no partitions.
    pub fn with_task_source<U: FileScanTaskSource>(
        self,
        task_source: U,
    ) -> Result<IcebergDataSource<U>> {
        validate_partition_count(&task_source)?;
        Ok(IcebergDataSource {
            task_source,
            schema: self.schema,
            reader: self.reader,
            fetch: self.fetch,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Task delivery or discovery used by this source.
    pub fn task_source(&self) -> &T {
        &self.task_source
    }

    pub fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    /// Storage configuration matching this source's reader.
    pub fn file_io(&self) -> &FileIO {
        self.reader.file_io()
    }

    /// Per-partition row limit, if any.
    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }
}

/// Builds an [`IcebergDataSource`]. Created by [`IcebergDataSource::builder`].
///
/// A table provider passes the table's schema and selects columns with
/// [`Self::with_projection`]. To rebuild a source, such as in another process,
/// pass its [`IcebergDataSource::schema`], which is already projected, and no
/// projection. To rebuild without a table, use
/// [`IcebergDataSource::from_task_source`] with native file tasks.
#[derive(Clone)]
pub struct IcebergDataSourceBuilder {
    table: Table,
    schema: ArrowSchemaRef,
    projection: Option<Vec<usize>>,
    snapshot_id: Option<i64>,
    predicate: Option<Predicate>,
    fetch: Option<usize>,
    reader: Option<ArrowReader>,
}

impl Debug for IcebergDataSourceBuilder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergDataSourceBuilder")
            .field("table", &self.table)
            .field("schema", &self.schema)
            .field("projection", &self.projection)
            .field("snapshot_id", &self.snapshot_id)
            .field("predicate", &self.predicate)
            .field("fetch", &self.fetch)
            .field("has_reader", &self.reader.is_some())
            .finish()
    }
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

    /// Stops each partition after `fetch` rows, or reads every row if `None`.
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        self
    }

    /// Uses this reader without changing its settings to session defaults.
    ///
    /// The reader must use FileIO matching the table. Its clones share the cached
    /// delete-file loader. Without this override, a reader is built at execution
    /// with the table's runtime and the session's batch size and page-index flag.
    pub fn with_reader(mut self, reader: ArrowReader) -> Self {
        self.reader = Some(reader);
        self
    }

    /// Builds the source without discovering files or performing I/O.
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
        let reader = match self.reader {
            Some(reader) => ScanReader::Provided {
                file_io: self.table.file_io().clone(),
                reader,
            },
            None => ScanReader::Table(self.table.clone()),
        };
        Ok(IcebergDataSource {
            task_source: TableScanTaskSource {
                table: self.table,
                snapshot_id: self.snapshot_id,
                projection,
                predicate: self.predicate,
            },
            schema,
            reader,
            fetch: self.fetch,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl<T: FileScanTaskSource> DataSource for IcebergDataSource<T> {
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
        if partition >= self.task_source.partition_count() {
            return internal_err!("Iceberg scan partition {partition} does not exist");
        }
        if self.fetch == Some(0) {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema.clone(),
                futures::stream::empty(),
            )));
        }
        let tasks = self.task_source.tasks(partition, context.clone())?;
        let stream = self
            .reader
            .reader(&context)
            .read(tasks)
            .map_err(to_datafusion_error)?
            .stream()
            .map_err(to_datafusion_error);
        let stream = Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), stream));
        // Unlike filtering out batches after the limit, LimitStream drops its
        // input as soon as the limit is reached, releasing task delivery too.
        Ok(Box::pin(LimitStream::new(
            stream,
            0,
            self.fetch,
            BaselineMetrics::new(&self.metrics, partition),
        )))
    }

    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "format=iceberg")?;
        self.task_source.fmt_as(f)?;
        if let Some(fetch) = self.fetch {
            write!(f, ", fetch={fetch}")?;
        }
        Ok(())
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.task_source.partition_count())
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(self.schema.clone())
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        if let Some(partition) = partition
            && partition >= self.task_source.partition_count()
        {
            return internal_err!("Iceberg scan partition {partition} does not exist");
        }
        Ok(Arc::new(Statistics::new_unknown(&self.schema)))
    }

    fn metrics(&self) -> ExecutionPlanMetricsSet {
        self.metrics.clone()
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

fn validate_partition_count(task_source: &impl FileScanTaskSource) -> Result<()> {
    if task_source.partition_count() == 0 {
        return internal_err!("Iceberg task source requires at least one partition");
    }
    Ok(())
}
