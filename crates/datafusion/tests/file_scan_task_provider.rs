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

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::Result;
use datafusion::datasource::source::{DataSource, DataSourceExec};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::SessionConfig;
use futures::TryStreamExt;
use iceberg::arrow::ArrowReaderBuilder;
use iceberg::io::FileIO;
use iceberg::table::Table;
use iceberg_datafusion::IcebergDataFusionConfig;
use iceberg_datafusion::read::{
    EagerFileScanTaskProvider, FileScanTaskProvider, FileScanTaskProviderContext,
    IcebergDataSource, IcebergDataSourceBuilder, LazyFileScanTaskProvider,
};
use iceberg_datafusion::test_utils::IcebergTestHarness;

#[tokio::test]
async fn lazy_discovery_does_not_read_manifests_until_polled() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_config(SessionConfig::new().with_target_partitions(3))
        .build()
        .await?;
    let plan = harness.plan("SELECT pickup_date FROM taxi").await?;
    let source = iceberg_source(&plan);
    let lazy = (source.file_scan_task_provider() as &dyn Any)
        .downcast_ref::<LazyFileScanTaskProvider>()
        .unwrap();
    assert_eq!(lazy.partitioning().partition_count(), 1);
    assert_eq!(lazy.projection(), &["pickup_date"]);

    // Keep the loaded metadata but deliberately remove its S3 fixture storage.
    // LocalFs cannot read these manifest paths. Opening the lazy stream must
    // still succeed; only polling it should attempt that I/O and fail.
    let inaccessible_table = Table::builder()
        .identifier(lazy.table().identifier().clone())
        .metadata(lazy.table().metadata_ref())
        .file_io(FileIO::new_with_fs())
        .runtime(iceberg::Runtime::current())
        .build()
        .map_err(iceberg_datafusion::to_datafusion_error)?;
    let discovery = Arc::new(LazyFileScanTaskProvider::new(
        inaccessible_table,
        lazy.projection().to_vec(),
    ));
    let context = harness.context().task_ctx();
    let mut tasks = discovery.open(
        0,
        FileScanTaskProviderContext {
            task_context: &context,
        },
    )?;
    assert!(tasks.try_next().await.is_err());
    Ok(())
}

#[tokio::test]
async fn eager_table_planning_selects_files_before_execution() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_config(
            SessionConfig::new()
                .with_target_partitions(3)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;
    let plan = harness
        .plan(
            "SELECT pickup_date, COUNT(*) AS trips FROM taxi \
         WHERE pickup_date IN (DATE '2024-01-10', DATE '2024-01-11') \
         GROUP BY pickup_date ORDER BY pickup_date",
        )
        .await?;
    let eager = (iceberg_source(&plan).file_scan_task_provider() as &dyn Any)
        .downcast_ref::<EagerFileScanTaskProvider>()
        .unwrap();
    // Only two selected files, so eager discovery caps the requested three partitions.
    assert_eq!(
        eager.groups().iter().map(Vec::len).collect::<Vec<_>>(),
        vec![1, 1]
    );
    let batches = collect(plan, harness.context().task_ctx()).await?;
    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @"
    +-------------+-------+
    | pickup_date | trips |
    +-------------+-------+
    | 2024-01-10  | 25000 |
    | 2024-01-11  | 25000 |
    +-------------+-------+
    ");
    Ok(())
}

#[tokio::test]
async fn eager_planning_consumes_the_lazy_providers_discovery() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_config(SessionConfig::new().with_target_partitions(3))
        .build()
        .await?;
    let plan = harness.plan("SELECT pickup_date FROM taxi").await?;
    let lazy = (iceberg_source(&plan).file_scan_task_provider() as &dyn Any)
        .downcast_ref::<LazyFileScanTaskProvider>()
        .unwrap()
        .clone();
    let eager =
        EagerFileScanTaskProvider::plan(Arc::new(lazy), harness.context().task_ctx())
            .await?;
    assert_eq!(
        eager.groups().iter().map(Vec::len).collect::<Vec<_>>(),
        vec![3, 2, 2]
    );
    assert_eq!(eager.tasks().count(), 7);
    Ok(())
}

#[tokio::test]
async fn reconstruction_needs_no_table_or_current_runtime() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_config(
            SessionConfig::new()
                .with_target_partitions(3)
                .with_batch_size(1)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;
    let plan = harness
        .plan("SELECT pickup_date FROM taxi WHERE pickup_date = DATE '2024-01-10'")
        .await?;
    let source = iceberg_source(&plan);
    let eager = (source.file_scan_task_provider() as &dyn Any)
        .downcast_ref::<EagerFileScanTaskProvider>()
        .unwrap();
    let file = eager.tasks().next().unwrap().clone();
    // Explicit assignment: preserve an empty partition and a repeated file.
    let groups = vec![vec![file.clone()], vec![], vec![file.clone(), file]];
    let (schema, file_io, reader) = (
        source.schema(),
        source.file_io().clone(),
        source.reader().clone(),
    );
    drop(plan);
    drop(harness);

    let source = std::thread::spawn(move || {
        assert!(tokio::runtime::Handle::try_current().is_err());
        IcebergDataSourceBuilder::new(schema, file_io, reader)
            .with_file_scan_task_provider(Arc::new(EagerFileScanTaskProvider::try_new(
                groups,
            )?))
            .with_fetch(Some(2))
            .build()
    })
    .join()
    .unwrap()?;
    let groups = (source.file_scan_task_provider() as &dyn Any)
        .downcast_ref::<EagerFileScanTaskProvider>()
        .unwrap()
        .groups();
    assert_eq!(
        groups.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![1, 0, 2]
    );
    assert_eq!(source.output_partitioning().partition_count(), 3);

    // This execution context has a different batch size. It must not replace
    // the supplied reader's size of one. Streams also outlive the source.
    let context = Arc::new(TaskContext::default());
    let streams = (0..3)
        .map(|p| source.open(p, Arc::clone(&context)))
        .collect::<Result<Vec<_>>>()?;
    drop(source);
    let mut batches = Vec::new();
    for stream in streams {
        batches.extend(stream.try_collect::<Vec<_>>().await?);
    }
    assert!(batches.iter().all(|batch| batch.num_rows() <= 1));
    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @"
    +-------------+
    | pickup_date |
    +-------------+
    | 2024-01-10  |
    | 2024-01-10  |
    | 2024-01-10  |
    | 2024-01-10  |
    +-------------+
    ");
    Ok(())
}

#[tokio::test]
async fn replacing_assignments_preserves_reader_schema_and_fetch() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_config(
            SessionConfig::new()
                .with_batch_size(1)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;
    let plan = harness
        .plan("SELECT pickup_date FROM taxi WHERE pickup_date = DATE '2024-01-11'")
        .await?;
    let source = iceberg_source(&plan);
    let eager = (source.file_scan_task_provider() as &dyn Any)
        .downcast_ref::<EagerFileScanTaskProvider>()
        .unwrap();
    let file = eager.tasks().next().unwrap().clone();
    let original = source.with_fetch(Some(2)).unwrap();
    let original = original.downcast_ref::<IcebergDataSource>().unwrap();
    let replacement = original.with_file_scan_task_provider(Arc::new(
        EagerFileScanTaskProvider::try_new(vec![vec![], vec![file]])?,
    ))?;
    assert_eq!(original.output_partitioning().partition_count(), 1);
    assert_eq!(replacement.output_partitioning().partition_count(), 2);
    assert_eq!(replacement.schema(), original.schema());
    assert_eq!(replacement.fetch(), Some(2));
    let context = harness.context().task_ctx();
    assert!(
        replacement
            .open(0, Arc::clone(&context))?
            .try_collect::<Vec<_>>()
            .await?
            .is_empty()
    );
    let batches = replacement
        .open(1, context)?
        .try_collect::<Vec<_>>()
        .await?;
    assert!(batches.iter().all(|batch| batch.num_rows() <= 1));
    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @"
    +-------------+
    | pickup_date |
    +-------------+
    | 2024-01-11  |
    | 2024-01-11  |
    +-------------+
    ");
    Ok(())
}

#[tokio::test]
async fn empty_eager_discovery_keeps_one_partition() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_config(
            SessionConfig::new()
                .with_target_partitions(3)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;
    let plan = harness
        .plan("SELECT pickup_date FROM taxi WHERE pickup_date = DATE '1900-01-01'")
        .await?;
    let source = iceberg_source(&plan);
    let eager = (source.file_scan_task_provider() as &dyn Any)
        .downcast_ref::<EagerFileScanTaskProvider>()
        .unwrap();
    assert_eq!(
        eager.groups().iter().map(Vec::len).collect::<Vec<_>>(),
        vec![0]
    );
    assert!(
        collect(plan, harness.context().task_ctx())
            .await?
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn reconstruction_requires_a_task_provider() -> Result<()> {
    let file_io = FileIO::new_with_fs();
    let reader =
        ArrowReaderBuilder::new(file_io.clone(), iceberg::Runtime::current()).build();
    let error = IcebergDataSourceBuilder::new(Arc::new(Schema::empty()), file_io, reader)
        .build()
        .unwrap_err();
    insta::assert_snapshot!(error.to_string(), @"Execution error: An Iceberg source requires a file task provider");
    Ok(())
}

#[test]
fn eager_provider_rejects_zero_groups() {
    let error = EagerFileScanTaskProvider::try_new(vec![]).unwrap_err();
    insta::assert_snapshot!(error.to_string(), @"Execution error: An Iceberg task provider requires at least one file group");
}

fn iceberg_source(plan: &Arc<dyn ExecutionPlan>) -> &IcebergDataSource {
    find_source(plan).expect("query reads the Iceberg fixture")
}

fn find_source(plan: &Arc<dyn ExecutionPlan>) -> Option<&IcebergDataSource> {
    plan.downcast_ref::<DataSourceExec>()
        .and_then(|exec| exec.data_source().downcast_ref::<IcebergDataSource>())
        .or_else(|| plan.children().into_iter().find_map(find_source))
}
