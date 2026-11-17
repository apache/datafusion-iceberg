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

use std::sync::Arc;

use super::file_scan_task_provider::{
    FileScanTaskProvider, FileScanTaskProviderContext, LazyFileScanTaskProvider,
};
use super::predicate::convert_filters_to_predicate;
use crate::read::EagerFileScanTaskProvider;
use crate::to_datafusion_error;

use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::catalog::Session;
use datafusion::common::Statistics;
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
use futures::{StreamExt, TryStreamExt};
use iceberg::expr::Predicate;
use iceberg::scan::FileScanTask;
use iceberg::table::Table;

/// Reads an Iceberg [`Table`]: the file tasks supplied by its
/// [`FileScanTaskProvider`], as batches of its projected schema.
///
/// Built with [`IcebergDataSourceBuilder`],
/// which discovers files lazily during execution. Swap in another provider
/// with [`Self::with_file_scan_task_provider`].
#[derive(Debug, Clone)]
pub struct IcebergDataSource {
    /// Table the files belong to, used to build the reader.
    table: Table,
    /// Schema returned by this data source after projection.
    schema: ArrowSchemaRef,
    /// Optional limit on the number of rows to return
    fetch: Option<usize>,
    /// Files to read, by output partition.
    file_scan_task_provider: Arc<dyn FileScanTaskProvider>,
    /// Output rows and compute time recorded while reading.
    metrics: ExecutionPlanMetricsSet,
}

impl IcebergDataSource {
    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    /// Supplies the files this source reads. Downcast to inspect a concrete
    /// provider.
    pub fn file_scan_task_provider(&self) -> &dyn FileScanTaskProvider {
        self.file_scan_task_provider.as_ref()
    }

    /// Returns a source that reads the files of `file_scan_task_provider`
    /// instead, with fresh metrics. The
    /// tasks must read this source's projected columns.
    pub fn with_file_scan_task_provider(
        &self,
        file_scan_task_provider: Arc<dyn FileScanTaskProvider>,
    ) -> Self {
        Self {
            file_scan_task_provider,
            metrics: Default::default(),
            ..self.clone()
        }
    }
}

/// Builds an [`IcebergDataSource`].
///
/// [`Self::new`] takes what every scan needs; everything else is optional and
/// set with the `with_*` methods, each defaulting to `None`.
///
/// A table provider passes the table's schema and selects columns with
/// [`Self::with_projection`]. To rebuild a source, such as in another process,
/// pass its [`IcebergDataSource::schema`], which is already projected, and no
/// projection.
#[derive(Clone)]
pub struct IcebergDataSourceBuilder<'a> {
    table: &'a Table,
    schema: ArrowSchemaRef,
    projection: Option<Vec<usize>>,
    snapshot_id: Option<i64>,
    predicate: Option<Predicate>,
    fetch: Option<usize>,
    collected_file_scan_tasks: Option<Vec<Vec<FileScanTask>>>,
}

impl<'a> IcebergDataSourceBuilder<'a> {
    /// Starts building a source that scans `table` and returns the columns of
    /// `schema`, all of them unless [`Self::with_projection`] selects some.
    /// Columns are read from Iceberg by name.
    pub fn new(table: &'a Table, schema: ArrowSchemaRef) -> Self {
        Self {
            table,
            schema,
            projection: None,
            snapshot_id: None,
            predicate: None,
            fetch: None,
            collected_file_scan_tasks: None,
        }
    }

    /// Returns only the columns of the schema at these indices, or all of them
    /// if `None`. Indices are into the schema passed to [`Self::new`].
    pub fn with_projection(mut self, projection: Option<&Vec<usize>>) -> Self {
        self.projection = projection.cloned();
        self
    }

    /// Scans this snapshot, or the table's current one if `None`.
    pub fn with_snapshot_id(mut self, snapshot_id: Option<i64>) -> Self {
        self.snapshot_id = snapshot_id;
        self
    }

    /// Filters the scan with `predicate`, or not at all if `None`.
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

    /// Stops after `fetch` rows, or reads every row if `None`.
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        self
    }

    /// Reads these already planned file tasks, one output partition per
    /// group, instead of discovering files. Use this to rebuild an eagerly
    /// planned source, such as in another process.
    pub fn with_file_scan_tasks(
        mut self,
        file_scan_tasks: Vec<Vec<FileScanTask>>,
    ) -> Self {
        self.collected_file_scan_tasks = Some(file_scan_tasks);
        self
    }

    /// Gather all the [FileScanTask] in-memory, split round-robin into up to
    /// `session`'s target partitions.
    ///
    /// Discovery uses the projection, snapshot and predicate set so far, so
    /// call this after the other `with_*` methods.
    ///
    /// # Errors
    ///
    /// Returns an error if discovery fails.
    pub async fn try_with_eagerly_planned_file_scan_tasks(
        mut self,
        session: &dyn Session,
    ) -> Result<Self> {
        let scan = match self.snapshot_id {
            Some(snapshot_id) => self.table.scan().snapshot_id(snapshot_id),
            None => self.table.scan(),
        };

        // Read only the projected columns, which are what `build` reports in
        // the source's schema.
        let projection: Vec<String> = self
            .projected_schema()?
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let mut scan = scan.select(&projection);
        if let Some(predicate) = &self.predicate {
            scan = scan.with_filter(predicate.clone());
        }
        let scan = scan.build().map_err(to_datafusion_error)?;
        let mut stream = futures::stream::once(async move { scan.plan_files().await })
            .try_flatten()
            .boxed();

        let target_partitions = session.config().target_partitions();
        let mut groups: Vec<Vec<FileScanTask>> = Vec::new();
        let mut partition = 0;
        while let Some(task) = stream.try_next().await.map_err(to_datafusion_error)? {
            if partition == groups.len() {
                groups.push(Vec::new());
            }
            groups[partition].push(task);
            partition = (partition + 1) % target_partitions;
        }

        if groups.is_empty() {
            groups.push(Vec::new());
        }

        self.collected_file_scan_tasks = Some(groups);
        Ok(self)
    }

    /// Builds the source, which reads the columns of its projected schema by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the projection is out of bounds for the schema, or
    /// if [`Self::with_file_scan_tasks`] was given no groups.
    pub fn build(self) -> Result<IcebergDataSource> {
        let schema = self.projected_schema()?;
        let projection: Vec<String> = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();

        Ok(IcebergDataSource {
            table: self.table.clone(),
            schema,
            fetch: self.fetch,
            file_scan_task_provider: match self.collected_file_scan_tasks {
                Some(file_scan_tasks) => {
                    Arc::new(EagerFileScanTaskProvider { file_scan_tasks })
                }
                None => Arc::new(LazyFileScanTaskProvider {
                    snapshot_id: self.snapshot_id,
                    projection,
                    predicate: self.predicate,
                }),
            },
            metrics: Default::default(),
        })
    }

    /// The schema passed to [`Self::new`], narrowed to the projection.
    fn projected_schema(&self) -> Result<ArrowSchemaRef> {
        Ok(match &self.projection {
            None => Arc::clone(&self.schema),
            Some(projection) => Arc::new(self.schema.project(projection)?),
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
        let tasks = Arc::clone(&self.file_scan_task_provider).open(
            partition,
            FileScanTaskProviderContext {
                task_context: &context,
                table: &self.table,
            },
        )?;
        let config = context.session_config();
        let reader = self
            .table
            .reader_builder()
            .with_batch_size(config.batch_size())
            .with_row_selection_enabled(
                config.options().execution.parquet.enable_page_index,
            );
        let stream = reader
            // DataFusion already runs partitions concurrently, so each one reads a
            // single file at a time.
            .with_data_file_concurrency_limit(1)
            .build()
            .read(tasks)
            .map_err(to_datafusion_error)?
            .stream()
            .map_err(to_datafusion_error);
        Ok(Box::pin(LimitStream::new(
            Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), stream)),
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
        // The schema is already projected, so its fields are the columns read.
        let projection: Vec<&str> = self
            .schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect();
        write!(f, "format=iceberg, projection=[{}]", projection.join(", "))?;
        self.file_scan_task_provider.fmt_as(f)?;
        if let Some(fetch) = self.fetch {
            write!(f, ", fetch={fetch}")?;
        }
        Ok(())
    }

    fn output_partitioning(&self) -> Partitioning {
        self.file_scan_task_provider.partitioning()
    }

    fn metrics(&self) -> ExecutionPlanMetricsSet {
        self.metrics.clone()
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
