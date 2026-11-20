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
use datafusion::common::stats::Precision;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Statistics, exec_err};
use datafusion::datasource::source::DataSource;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::limit::LimitStream;
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricBuilder, MetricCategory, MetricType,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayFormatType, Partitioning};
use datafusion::prelude::SessionConfig;
use futures::stream::poll_fn;
use futures::{StreamExt, TryStreamExt};
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
    /// planning enabled: the caller supplies the task provider at construction.
    /// Omitting the provider is a compile-time error:
    ///
    /// ```compile_fail,E0061
    /// # use datafusion::{arrow::datatypes::SchemaRef, prelude::SessionConfig};
    /// # use iceberg::{arrow::ArrowReaderBuilder, io::FileIO};
    /// # use iceberg_datafusion::read::IcebergDataSource;
    /// # fn missing_provider(config: &SessionConfig, schema: SchemaRef, file_io: FileIO, reader: ArrowReaderBuilder) {
    /// IcebergDataSource::builder(config, schema, file_io, reader);
    /// # }
    /// ```
    pub fn builder(
        config: &SessionConfig,
        schema: ArrowSchemaRef,
        file_io: FileIO,
        reader: ArrowReaderBuilder,
        file_scan_task_provider: Arc<dyn FileScanTaskProvider>,
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
            file_scan_task_provider,
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
///     IcebergDataSource::builder(config, schema, file_io, reader, Arc::new(provider))
///         .build()
/// }
/// ```
#[derive(Clone)]
pub struct IcebergDataSourceBuilder {
    schema: ArrowSchemaRef,
    file_io: FileIO,
    reader: ArrowReader,
    fetch: Option<usize>,
    file_scan_task_provider: Arc<dyn FileScanTaskProvider>,
}

impl IcebergDataSourceBuilder {
    /// Uses an explicitly configured, shared reader instead of session defaults.
    /// Its FileIO must match the FileIO supplied at builder construction. Retaining
    /// a reader also retains its cached delete files across output partitions.
    pub fn with_reader(mut self, reader: ArrowReader) -> Self {
        self.reader = reader;
        self
    }

    /// Stops each partition after `fetch` rows, or reads every row for `None`.
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        self
    }

    /// Builds a source without discovering files. Returns an error if the
    /// provider advertises no output partitions.
    pub fn build(self) -> Result<IcebergDataSource> {
        validate_provider(self.file_scan_task_provider.as_ref())?;
        Ok(IcebergDataSource {
            schema: self.schema,
            file_io: self.file_io,
            reader: self.reader,
            fetch: self.fetch,
            file_scan_task_provider: self.file_scan_task_provider,
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
        let files_opened = MetricBuilder::new(&self.metrics)
            .with_category(MetricCategory::Rows)
            .counter("files_opened", partition);
        let bytes_scanned = MetricBuilder::new(&self.metrics)
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Bytes)
            .gauge("bytes_scanned", partition);

        // Count files when the reader pulls their tasks, including read-ahead for LIMIT.
        let tasks = tasks.inspect_ok(move |_| files_opened.add(1)).boxed();
        let result = self
            .reader
            .clone()
            .read(tasks)
            .map_err(to_datafusion_error)?;
        // Reader bytes include data and delete files, but not manifest discovery.
        let reader_metrics = result.metrics().clone();
        let mut batches = result.stream().map_err(to_datafusion_error);
        let stream = poll_fn(move |cx| {
            let poll = batches.poll_next_unpin(cx);
            if poll.is_ready() {
                bytes_scanned.set(reader_metrics.bytes_read() as usize);
            }
            poll
        });
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

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        let partitions = self.output_partitioning().partition_count();
        if partition.is_some_and(|partition| partition >= partitions) {
            return exec_err!("Invalid Iceberg scan partition {partition:?}");
        }
        let mut statistics = self
            .file_scan_task_provider
            .partition_statistics(partition, &self.schema)?;
        if statistics.column_statistics.len() != self.schema.fields().len() {
            return exec_err!(
                "Iceberg task provider returned statistics for {} columns, expected {}",
                statistics.column_statistics.len(),
                self.schema.fields().len()
            );
        }
        let fetch = if partition.is_none() && partitions > 1 {
            // Fetch is enforced independently in each partition. Without its row
            // distribution we can only bound the sum, even for an exact input.
            if self.fetch.is_some() {
                statistics = statistics.to_inexact();
            }
            self.fetch.and_then(|fetch| fetch.checked_mul(partitions))
        } else {
            self.fetch
        };
        statistics = statistics.with_fetch(fetch, 0, 1)?;
        // Keep provider-supplied byte estimates. For unknown sizes, estimate
        // projected in-memory widths, not compressed Iceberg file sizes. Do this
        // after fetch so its row bound can also supply a missing byte estimate.
        if statistics.total_byte_size == Precision::Absent {
            statistics.calculate_total_byte_size(&self.schema);
        }
        Ok(Arc::new(statistics))
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::common::Statistics;
    use datafusion::common::stats::Precision;
    use datafusion::datasource::source::DataSource;
    use datafusion::prelude::SessionContext;
    use iceberg::io::FileIO;
    use iceberg::spec::{
        FormatVersion, MAIN_BRANCH, NestedField, Operation, PartitionSpec, PrimitiveType,
        Schema, Snapshot, SortOrder, Summary, TableMetadataBuilder, Type,
    };
    use iceberg::table::Table;
    use iceberg::{Runtime, TableIdent};

    use super::IcebergDataSource;
    use crate::read::FileScanTaskProviderBuilder;

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
    async fn id_scan_statistics(
        table: Table,
        snapshot_id: Option<i64>,
        fetch: Option<usize>,
    ) -> Statistics {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let context = SessionContext::new();
        let (schema, provider) =
            FileScanTaskProviderBuilder::new(context.task_ctx(), table.clone(), schema)
                .with_projection(Some(&vec![0]))
                .with_snapshot_id(snapshot_id)
                .build()
                .await
                .unwrap();
        let source = IcebergDataSource::builder(
            &context.copied_config(),
            schema,
            table.file_io().clone(),
            table.reader_builder(),
            provider,
        )
        .with_fetch(fetch)
        .build()
        .unwrap();
        Arc::unwrap_or_clone(source.partition_statistics(None).unwrap())
    }

    #[tokio::test]
    async fn reports_no_rows_for_a_table_without_snapshots() {
        let statistics =
            id_scan_statistics(table_with_snapshot_summaries(&[]), None, None).await;
        assert_eq!(statistics.num_rows, Precision::Exact(0));
        assert_eq!(statistics.total_byte_size, Precision::Exact(0));
    }

    #[tokio::test]
    async fn estimates_rows_from_the_snapshot_summary() {
        let table = table_with_snapshot_summaries(&[&[("total-records", "100")]]);
        let statistics = id_scan_statistics(table, None, None).await;
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
            let statistics = id_scan_statistics(table, None, None).await;
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
            let statistics = id_scan_statistics(table, None, None).await;
            assert_eq!(statistics.num_rows, Precision::Absent, "{summary:?}");
            assert_eq!(statistics.total_byte_size, Precision::Absent, "{summary:?}");
        }
    }

    #[tokio::test]
    async fn reports_unknown_rows_for_a_missing_snapshot() {
        let table = table_with_snapshot_summaries(&[&[("total-records", "100")]]);
        let statistics = id_scan_statistics(table, Some(2), None).await;
        assert_eq!(statistics.num_rows, Precision::Absent);
    }

    #[tokio::test]
    async fn uses_the_summary_of_the_scanned_snapshot() {
        let table = table_with_snapshot_summaries(&[
            &[("total-records", "100")],
            &[("total-records", "200")],
        ]);
        assert_eq!(
            id_scan_statistics(table.clone(), Some(1), None)
                .await
                .num_rows,
            Precision::Inexact(100)
        );
        assert_eq!(
            id_scan_statistics(table, None, None).await.num_rows,
            Precision::Inexact(200)
        );
    }

    #[tokio::test]
    async fn caps_rows_at_the_fetch() {
        let table = table_with_snapshot_summaries(&[&[("total-records", "100")]]);
        let statistics = id_scan_statistics(table, None, Some(10)).await;
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
        let context = SessionContext::new();
        let (schema, provider) =
            FileScanTaskProviderBuilder::new(context.task_ctx(), table.clone(), schema)
                .build()
                .await
                .unwrap();
        let source = IcebergDataSource::builder(
            &context.copied_config(),
            schema,
            table.file_io().clone(),
            table.reader_builder(),
            provider,
        )
        .build()
        .unwrap();
        let statistics = source.partition_statistics(None).unwrap();
        assert_eq!(statistics.num_rows, Precision::Inexact(100));
        assert_eq!(statistics.total_byte_size, Precision::Absent);
    }
}
