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

use super::file_scan_task_provider::{FileScanTaskProvider, FileScanTaskProviderContext};
use crate::to_datafusion_error;

use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Statistics, exec_datafusion_err, exec_err};
use datafusion::datasource::source::DataSource;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::limit::LimitStream;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayFormatType, Partitioning};
use datafusion::prelude::SessionConfig;
use futures::TryStreamExt;
use iceberg::arrow::{ArrowReader, ArrowReaderBuilder};
use iceberg::io::FileIO;

/// Reads the tasks of a [`FileScanTaskProvider`] using a process-local reader.
///
/// The source does not require a table. Lazy discovery retains its table in the
/// provider; eager and remote providers can supply tasks without one. The reader
/// is shared across partitions, including its loaded delete files and settings.
#[derive(Clone)]
pub struct IcebergDataSource {
    schema: ArrowSchemaRef,
    file_io: FileIO,
    reader: ArrowReader,
    fetch: Option<usize>,
    file_scan_task_provider: Arc<dyn FileScanTaskProvider>,
    metrics: ExecutionPlanMetricsSet,
}

impl Debug for IcebergDataSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergDataSource")
            .field("schema", &self.schema)
            .field("fetch", &self.fetch)
            .field("file_scan_task_provider", &self.file_scan_task_provider)
            .finish_non_exhaustive()
    }
}

impl IcebergDataSource {
    /// Starts source construction with session reader settings and explicit local
    /// storage/runtime resources. `reader` must use the supplied `file_io`.
    /// Batch size and page-index settings come from `config`; file concurrency
    /// defaults to one because DataFusion runs partitions concurrently.
    /// A later [`IcebergDataSourceBuilder::with_reader`] overrides these defaults.
    /// No discovery or current-runtime lookup is performed, even with eager
    /// planning enabled: the caller supplies the task provider explicitly.
    pub fn builder(
        config: &SessionConfig,
        schema: ArrowSchemaRef,
        file_io: FileIO,
        reader: ArrowReaderBuilder,
    ) -> IcebergDataSourceBuilder {
        IcebergDataSourceBuilder {
            schema,
            file_io,
            reader: reader
                .with_batch_size(config.batch_size())
                .with_row_selection_enabled(
                    config.options().execution.parquet.enable_page_index,
                )
                .with_data_file_concurrency_limit(1)
                .build(),
            fetch: None,
            file_scan_task_provider: None,
        }
    }

    pub fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    /// Storage access used by the reader, retained for transport/reconstruction.
    pub fn file_io(&self) -> &FileIO {
        &self.file_io
    }

    /// Process-local reader shared by all output partitions.
    pub fn reader(&self) -> &ArrowReader {
        &self.reader
    }

    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    /// Supplies the files this source reads. Downcast to inspect a concrete provider.
    pub fn file_scan_task_provider(&self) -> &dyn FileScanTaskProvider {
        self.file_scan_task_provider.as_ref()
    }

    /// Replaces task delivery while retaining the reader, schema and fetch, and
    /// resetting metrics. Tasks must produce this source's projected schema.
    /// Returns an error if the provider advertises no output partitions.
    pub fn with_file_scan_task_provider(
        &self,
        file_scan_task_provider: Arc<dyn FileScanTaskProvider>,
    ) -> Result<Self> {
        validate_provider(file_scan_task_provider.as_ref())?;
        Ok(Self {
            file_scan_task_provider,
            metrics: Default::default(),
            ..self.clone()
        })
    }
}

/// Assembles a source from its output schema, storage, reader and task provider.
///
/// Construction is synchronous and performs no discovery, file I/O or runtime
/// lookup. Start with [`IcebergDataSource::builder`], which applies session reader
/// settings, or override them with [`Self::with_reader`]. The execution context
/// passed to `open` does not replace the retained reader's settings. FileIO must
/// match the reader's storage, and tasks must produce the supplied output schema.
///
/// ```
/// # use std::sync::Arc;
/// # use datafusion::arrow::datatypes::SchemaRef;
/// # use datafusion::common::Result;
/// # use datafusion::prelude::SessionConfig;
/// # use iceberg::{arrow::ArrowReaderBuilder, io::FileIO, scan::FileScanTask};
/// # use iceberg_datafusion::read::{EagerFileScanTaskProvider, IcebergDataSource};
/// fn reconstruct(
///     config: &SessionConfig,
///     schema: SchemaRef,
///     file_io: FileIO,
///     reader: ArrowReaderBuilder,
///     groups: Vec<Vec<FileScanTask>>,
/// ) -> Result<IcebergDataSource> {
///     let provider = EagerFileScanTaskProvider::try_new(groups)?;
///     IcebergDataSource::builder(config, schema, file_io, reader)
///         .with_file_scan_task_provider(Arc::new(provider))
///         .build()
/// }
/// ```
#[derive(Clone)]
pub struct IcebergDataSourceBuilder {
    schema: ArrowSchemaRef,
    file_io: FileIO,
    reader: ArrowReader,
    fetch: Option<usize>,
    file_scan_task_provider: Option<Arc<dyn FileScanTaskProvider>>,
}

impl IcebergDataSourceBuilder {
    /// Uses an explicitly configured, shared reader instead of session defaults.
    /// Its FileIO must match the FileIO supplied at builder construction. Retaining
    /// a reader also retains its cached delete files across output partitions.
    pub fn with_reader(mut self, reader: ArrowReader) -> Self {
        self.reader = reader;
        self
    }

    /// Supplies task delivery. Required before calling [`Self::build`].
    pub fn with_file_scan_task_provider(
        mut self,
        provider: Arc<dyn FileScanTaskProvider>,
    ) -> Self {
        self.file_scan_task_provider = Some(provider);
        self
    }

    /// Stops each partition after `fetch` rows, or reads every row for `None`.
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        self
    }

    /// Builds a source without discovering files. Returns an error if no provider
    /// was supplied or the provider advertises no output partitions.
    pub fn build(self) -> Result<IcebergDataSource> {
        let provider = self.file_scan_task_provider.ok_or_else(|| {
            exec_datafusion_err!("An Iceberg source requires a file task provider")
        })?;
        validate_provider(provider.as_ref())?;
        Ok(IcebergDataSource {
            schema: self.schema,
            file_io: self.file_io,
            reader: self.reader,
            fetch: self.fetch,
            file_scan_task_provider: provider,
            metrics: Default::default(),
        })
    }
}

fn validate_provider(provider: &dyn FileScanTaskProvider) -> Result<()> {
    if provider.partitioning().partition_count() == 0 {
        return exec_err!(
            "An Iceberg task provider requires at least one output partition"
        );
    }
    Ok(())
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
        if partition >= self.output_partitioning().partition_count() {
            return exec_err!("Invalid Iceberg scan partition {partition}");
        }
        let tasks = Arc::clone(&self.file_scan_task_provider).open(
            partition,
            FileScanTaskProviderContext {
                task_context: &context,
            },
        )?;
        let stream = self
            .reader
            .clone()
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

    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
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
