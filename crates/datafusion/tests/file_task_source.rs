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

use std::fs;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll};

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::test_util::batches_to_sort_string;
use datafusion::common::{Result, exec_err};
use datafusion::datasource::source::DataSource;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::common::collect;
use datafusion::prelude::{SessionContext, col, lit};
use futures::{FutureExt, Stream, StreamExt, TryStreamExt};
use iceberg::arrow::{
    ArrowReaderBuilder, arrow_schema_to_schema, schema_to_arrow_schema,
};
use iceberg::io::FileIO;
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_DELETE_FILE_PATH, RESERVED_COL_NAME_DELETE_FILE_POS,
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use iceberg::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
use iceberg::spec::{DataContentType, DataFileFormat};
use iceberg::table::Table;
use iceberg_datafusion::read::{FileScanTaskSource, IcebergDataSource};
use iceberg_datafusion::test_utils::IcebergTestHarness;
use iceberg_datafusion::{
    IcebergCatalogTableProvider, from_datafusion_error, to_datafusion_error,
};
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use tempfile::TempDir;

#[tokio::test]
async fn delivered_partitions_preserve_projection_predicates_and_fetch() -> Result<()> {
    let harness = IcebergTestHarness::new().await?;
    let context = harness.context().task_ctx();
    let table = fixture_table(harness.context()).await?;
    let schema = Arc::new(
        schema_to_arrow_schema(table.metadata().current_schema())
            .map_err(to_datafusion_error)?,
    );
    let projection = vec![schema.index_of("pickup_date")?];
    // The predicate refers to a column outside the output projection.
    let local = IcebergDataSource::builder(table.clone(), schema)
        .with_projection(Some(&projection))
        .with_filters(&[col("trip_distance").gt(lit(1.0))])
        .build()?;
    let tasks: Vec<FileScanTask> = local
        .task_source()
        .tasks(0, context.clone())?
        .try_collect()
        .await
        .map_err(to_datafusion_error)?;
    let mut groups = vec![Vec::new(); 3];
    for (index, task) in tasks.into_iter().enumerate() {
        groups[index % 3].push(task);
    }
    let polled = Arc::new(AtomicUsize::new(0));
    // Reconstruct without a table or catalog, using native tasks and a reader.
    let source = IcebergDataSource::from_task_source(
        local.schema(),
        PartitionedTasks {
            groups: Arc::new(groups),
            polled: polled.clone(),
        },
        table.file_io().clone(),
        table.reader_builder().build(),
        None,
    )?;
    assert_eq!(source.output_partitioning().partition_count(), 3);
    assert_eq!(polled.load(Ordering::SeqCst), 0);
    assert!(source.open(3, context.clone()).is_err());
    assert!(source.partition_statistics(Some(3)).is_err());
    assert_eq!(polled.load(Ordering::SeqCst), 0);

    let mut actual = Vec::new();
    for partition in 0..3 {
        let stream = source.open(partition, context.clone())?;
        let batches = collect(stream).await?;
        assert!(!batches.is_empty());
        assert!(
            batches
                .iter()
                .all(|batch| batch.schema() == source.schema())
        );
        actual.extend(batches);

        let limited = source.with_fetch(Some(2)).unwrap();
        let batches = collect(limited.open(partition, context.clone())?).await?;
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            2
        );
    }
    let expected = collect(local.open(0, context)?).await?;
    assert_eq!(
        batches_to_sort_string(&actual),
        batches_to_sort_string(&expected)
    );
    assert!(polled.load(Ordering::SeqCst) > 0);
    Ok(())
}

#[tokio::test]
async fn replacing_discovery_preserves_explicit_reader_schema_and_fetch() -> Result<()> {
    let harness = IcebergTestHarness::new().await?;
    let context = harness.context().task_ctx();
    let table = fixture_table(harness.context()).await?;
    let schema = Arc::new(
        schema_to_arrow_schema(table.metadata().current_schema())
            .map_err(to_datafusion_error)?,
    );
    let projection = vec![schema.index_of("pickup_date")?];
    let snapshot_id = table.metadata().current_snapshot_id();
    let source = IcebergDataSource::builder(table.clone(), schema)
        .with_projection(Some(&projection))
        .with_snapshot_id(snapshot_id)
        .with_filters(&[col("trip_distance").gt(lit(1.0))])
        .with_reader(table.reader_builder().with_batch_size(7).build())
        .with_fetch(Some(19))
        .build()?;
    assert_eq!(source.snapshot_id(), snapshot_id);
    let projected_schema = source.schema();
    let tasks: Vec<FileScanTask> = source
        .task_source()
        .tasks(0, context.clone())?
        .try_collect()
        .await
        .map_err(to_datafusion_error)?;
    let mut groups = vec![Vec::new(); 2];
    for (index, task) in tasks.into_iter().enumerate() {
        groups[index % 2].push(task);
    }
    let polled = Arc::new(AtomicUsize::new(0));
    let source = source.with_task_source(PartitionedTasks {
        groups: Arc::new(groups),
        polled: polled.clone(),
    })?;
    assert_eq!(source.schema(), projected_schema);
    assert_eq!(source.fetch(), Some(19));
    assert_eq!(source.output_partitioning().partition_count(), 2);
    assert_eq!(polled.load(Ordering::SeqCst), 0);
    for partition in 0..2 {
        let batches = collect(source.open(partition, context.clone())?).await?;
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            19
        );
        assert!(batches.iter().all(|batch| batch.num_rows() <= 7));
    }
    Ok(())
}

#[tokio::test]
async fn task_delivery_errors_reach_the_caller_without_table_discovery() -> Result<()> {
    let file_io = FileIO::new_with_fs();
    let source = IcebergDataSource::from_task_source(
        Arc::new(Schema::empty()),
        FailingTasks,
        file_io.clone(),
        ArrowReaderBuilder::new(file_io, iceberg::Runtime::current()).build(),
        None,
    )?;
    let error = collect(source.open(0, Arc::new(TaskContext::default()))?)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("task transport failed"),
        "{error}"
    );
    Ok(())
}

#[tokio::test]
async fn construction_and_replacement_reject_zero_partitions() -> Result<()> {
    let file_io = FileIO::new_with_fs();
    let reader =
        ArrowReaderBuilder::new(file_io.clone(), iceberg::Runtime::current()).build();
    let empty = PartitionedTasks {
        groups: Arc::new(Vec::new()),
        polled: Arc::new(AtomicUsize::new(0)),
    };
    let error = IcebergDataSource::from_task_source(
        Arc::new(Schema::empty()),
        empty.clone(),
        file_io.clone(),
        reader.clone(),
        None,
    )
    .unwrap_err();
    insta::assert_snapshot!(error.strip_backtrace(), @"
    Internal error: Iceberg task source requires at least one partition.
    This issue was likely caused by a bug in DataFusion's code. Please help us to resolve this by filing a bug report in our issue tracker: https://github.com/apache/datafusion/issues
    ");

    let source = IcebergDataSource::from_task_source(
        Arc::new(Schema::empty()),
        FailingTasks,
        file_io,
        reader,
        None,
    )?;
    assert!(source.with_task_source(empty).is_err());
    Ok(())
}

#[tokio::test]
async fn zero_fetch_does_not_open_task_delivery() -> Result<()> {
    let file_io = FileIO::new_with_fs();
    let dropped = Arc::new(AtomicBool::new(false));
    let source = IcebergDataSource::from_task_source(
        Arc::new(Schema::empty()),
        PendingTasks {
            dropped: dropped.clone(),
        },
        file_io.clone(),
        ArrowReaderBuilder::new(file_io, iceberg::Runtime::current()).build(),
        Some(0),
    )?;
    let batches = collect(source.open(0, Arc::new(TaskContext::default()))?).await?;
    assert!(batches.is_empty());
    drop(source);
    // Opening this delivery creates a stream whose drop would set the flag.
    assert!(!dropped.load(Ordering::SeqCst));
    Ok(())
}

#[tokio::test]
async fn dropping_the_reader_releases_its_pending_task_stream() -> Result<()> {
    let file_io = FileIO::new_with_fs();
    let dropped = Arc::new(AtomicBool::new(false));
    let source = IcebergDataSource::from_task_source(
        Arc::new(Schema::empty()),
        PendingTasks {
            dropped: dropped.clone(),
        },
        file_io.clone(),
        ArrowReaderBuilder::new(file_io, iceberg::Runtime::current()).build(),
        None,
    )?;
    let mut stream = source.open(0, Arc::new(TaskContext::default()))?;
    drop(source);
    assert!(!dropped.load(Ordering::SeqCst));
    assert!(stream.next().now_or_never().is_none());
    drop(stream);
    assert!(dropped.load(Ordering::SeqCst));
    Ok(())
}

#[tokio::test]
async fn reaching_fetch_stops_before_requesting_another_task() -> Result<()> {
    let directory = TempDir::new()?;
    let fixture = files_with_shared_deletes(&directory)?;
    let file_io = FileIO::new_with_fs();
    let source = IcebergDataSource::from_task_source(
        fixture.schema,
        OneFileThenError {
            task: fixture.groups[0][0].clone(),
        },
        file_io.clone(),
        // No task prefetch: asking for the next task after this file is an error.
        ArrowReaderBuilder::new(file_io, iceberg::Runtime::current())
            .with_data_file_concurrency_limit(1)
            .build(),
        Some(1),
    )?;
    let batches = collect(source.open(0, Arc::new(TaskContext::default()))?).await?;
    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @"
    +-------+
    | value |
    +-------+
    | 2     |
    +-------+
    ");
    Ok(())
}

#[tokio::test]
async fn delivered_partitions_and_source_clones_share_position_deletes() -> Result<()> {
    let directory = TempDir::new()?;
    let fixture = files_with_shared_deletes(&directory)?;
    let file_io = FileIO::new_with_fs();
    let context = Arc::new(TaskContext::default());
    let source = IcebergDataSource::from_task_source(
        fixture.schema,
        PartitionedTasks {
            groups: Arc::new(fixture.groups),
            polled: Arc::new(AtomicUsize::new(0)),
        },
        file_io.clone(),
        ArrowReaderBuilder::new(file_io, iceberg::Runtime::current()).build(),
        None,
    )?;
    let mut batches = collect(source.open(0, context.clone())?).await?;
    // Reading the second file through a clone must reuse the loaded deletes,
    // including its own row's deletion, rather than reopen the deleted file.
    let cloned = source.clone();
    drop(source);
    fs::remove_file(fixture.delete_path)?;
    batches.extend(collect(cloned.open(1, context)?).await?);
    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @"
    +-------+
    | value |
    +-------+
    | 2     |
    | 4     |
    +-------+
    ");
    Ok(())
}

async fn fixture_table(context: &SessionContext) -> Result<Table> {
    let provider = context.table_provider("taxi").await?;
    let catalog = provider
        .downcast_ref::<IcebergCatalogTableProvider>()
        .unwrap();
    catalog
        .catalog()
        .load_table(catalog.table_ident())
        .await
        .map_err(to_datafusion_error)
}

#[derive(Debug, Clone)]
struct PartitionedTasks {
    groups: Arc<Vec<Vec<FileScanTask>>>,
    polled: Arc<AtomicUsize>,
}

impl FileScanTaskSource for PartitionedTasks {
    fn partition_count(&self) -> usize {
        self.groups.len()
    }

    fn tasks(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<FileScanTaskStream> {
        let Some(group) = self.groups.get(partition) else {
            return exec_err!("Invalid task partition {partition}");
        };
        let polled = self.polled.clone();
        Ok(futures::stream::iter(group.clone().into_iter().map(Ok))
            .inspect(move |_| {
                polled.fetch_add(1, Ordering::SeqCst);
            })
            .boxed())
    }
}

#[derive(Debug, Clone)]
struct FailingTasks;

impl FileScanTaskSource for FailingTasks {
    fn partition_count(&self) -> usize {
        1
    }

    fn tasks(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<FileScanTaskStream> {
        Ok(
            futures::stream::once(async { exec_err!("task transport failed") })
                .map_err(from_datafusion_error)
                .boxed(),
        )
    }
}

#[derive(Debug, Clone)]
struct OneFileThenError {
    task: FileScanTask,
}

impl FileScanTaskSource for OneFileThenError {
    fn partition_count(&self) -> usize {
        1
    }

    fn tasks(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<FileScanTaskStream> {
        let file = futures::stream::iter([Ok(self.task.clone())]);
        let error = futures::stream::once(async {
            exec_err!("Task delivery continued after reaching fetch")
        })
        .map_err(from_datafusion_error);
        Ok(file.chain(error).boxed())
    }
}

#[derive(Debug, Clone)]
struct PendingTasks {
    dropped: Arc<AtomicBool>,
}

impl FileScanTaskSource for PendingTasks {
    fn partition_count(&self) -> usize {
        1
    }

    fn tasks(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<FileScanTaskStream> {
        Ok(Box::pin(PendingTaskStream {
            dropped: self.dropped.clone(),
        }))
    }
}

struct PendingTaskStream {
    dropped: Arc<AtomicBool>,
}

impl Stream for PendingTaskStream {
    type Item = iceberg::Result<FileScanTask>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }
}

impl Drop for PendingTaskStream {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

struct DeleteFixture {
    schema: SchemaRef,
    groups: Vec<Vec<FileScanTask>>,
    delete_path: String,
}

/// Two data files, each with its first row removed by one shared delete file.
fn files_with_shared_deletes(directory: &TempDir) -> Result<DeleteFixture> {
    let schema = Arc::new(Schema::new(vec![field("value", DataType::Int64, 1)]));
    let iceberg_schema =
        Arc::new(arrow_schema_to_schema(&schema).map_err(to_datafusion_error)?);
    let paths = ["first.parquet", "second.parquet"]
        .map(|name| directory.path().join(name).to_str().unwrap().to_owned());
    for (path, values) in paths.iter().zip([[1, 2], [3, 4]]) {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(values.to_vec()))],
        )?;
        write_parquet(path, &batch)?;
    }
    let delete_schema = Arc::new(Schema::new(vec![
        field(
            RESERVED_COL_NAME_DELETE_FILE_PATH,
            DataType::Utf8,
            RESERVED_FIELD_ID_DELETE_FILE_PATH,
        ),
        field(
            RESERVED_COL_NAME_DELETE_FILE_POS,
            DataType::Int64,
            RESERVED_FIELD_ID_DELETE_FILE_POS,
        ),
    ]));
    let deletes = RecordBatch::try_new(
        delete_schema,
        vec![
            Arc::new(StringArray::from(
                paths.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(vec![0, 0])),
        ],
    )?;
    let delete_path = directory
        .path()
        .join("deletes.parquet")
        .to_str()
        .unwrap()
        .to_owned();
    write_parquet(&delete_path, &deletes)?;
    let delete = FileScanTaskDeleteFile::builder()
        .with_file_path(delete_path.clone())
        .with_file_size_in_bytes(fs::metadata(&delete_path)?.len())
        .with_file_type(DataContentType::PositionDeletes)
        .with_file_format(DataFileFormat::Parquet)
        .with_partition_spec_id(0)
        .build()
        .map_err(to_datafusion_error)?;
    let mut groups = Vec::new();
    for path in paths {
        let size = fs::metadata(&path)?.len();
        groups.push(vec![
            FileScanTask::builder()
                .with_file_size_in_bytes(size)
                .with_start(0)
                .with_length(size)
                .with_record_count(Some(2))
                .with_data_file_path(path)
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(iceberg_schema.clone())
                .with_project_field_ids(vec![1])
                .with_deletes(vec![delete.clone()])
                .with_case_sensitive(true)
                .build()
                .map_err(to_datafusion_error)?,
        ]);
    }
    Ok(DeleteFixture {
        schema,
        groups,
        delete_path,
    })
}

fn field(name: &str, data_type: DataType, id: i32) -> Field {
    Field::new(name, data_type, false).with_metadata(
        [(PARQUET_FIELD_ID_META_KEY.to_owned(), id.to_string())]
            .into_iter()
            .collect(),
    )
}

fn write_parquet(path: &str, batch: &RecordBatch) -> Result<()> {
    let mut writer = ArrowWriter::try_new(fs::File::create(path)?, batch.schema(), None)?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}
