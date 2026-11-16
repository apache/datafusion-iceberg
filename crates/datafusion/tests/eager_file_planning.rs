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

use std::error::Error;
use std::path::Path;
use std::sync::Arc;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::test_util::batches_to_sort_string;
use datafusion::common::{Result, stats::Precision};
use datafusion::datasource::TableProvider;
use datafusion::datasource::source::{DataSource, DataSourceExec};
use datafusion::error::DataFusionError;
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::common::collect;
use datafusion::prelude::{SessionConfig, SessionContext, col, lit};
use datafusion::scalar::ScalarValue;
use iceberg::scan::FileScanTask;
use iceberg_datafusion::read::{
    EagerIcebergDataSource, IcebergDataSource, IcebergDataSourceBuilder,
};
use iceberg_datafusion::test_utils::IcebergTestHarness;
use iceberg_datafusion::{IcebergCatalogTableProvider, IcebergDataFusionConfig};

mod common;
use common::scan_fixture::ScanFixture;

type TestResult = std::result::Result<(), Box<dyn Error>>;

#[tokio::test]
async fn eager_planning_exposes_files_and_balanced_partitions() -> TestResult {
    let fixture = ScanFixture::builder()
        .with_file([0])
        .with_file([1])
        .with_file([2])
        .with_file([3])
        .with_file([4])
        .with_file([5])
        .with_file([6])
        .build()
        .await?;
    let harness = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(4)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;

    let plan = harness.plan("SELECT * FROM items").await?;
    let source = eager_source(plan.as_ref());
    assert_eq!(source.planned_files().len(), 7);
    let loads = (0..4)
        .map(|p| source.planned_partition_files(p).map(|files| files.len()))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(loads, vec![2, 2, 2, 1]);
    let mut names = source.planned_files().map(file_name).collect::<Vec<_>>();
    names.sort();
    insta::assert_snapshot!(names.join("\n"), @"
    data-0.parquet
    data-1.parquet
    data-2.parquet
    data-3.parquet
    data-4.parquet
    data-5.parquet
    data-6.parquet
    ");
    // Planning descriptors does not claim exact query statistics.
    assert_eq!(
        source.partition_statistics(None)?.num_rows,
        Precision::Absent
    );
    assert_eq!(
        source.partition_statistics(None)?.total_byte_size,
        Precision::Absent
    );
    Ok(())
}

#[tokio::test]
async fn eager_partition_count_is_bounded_by_selected_files() -> TestResult {
    let fixture = ScanFixture::builder()
        .with_file([0, 1])
        .with_file([2, 3])
        .build()
        .await?;
    let harness = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(16)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;

    let plan = harness.plan("SELECT * FROM items").await?;
    let source = eager_source(plan.as_ref());
    assert_eq!(source.output_partitioning().partition_count(), 2);
    assert!(source.planned_partition_files(2).is_err());
    assert!(source.open(2, harness.context().task_ctx()).is_err());
    Ok(())
}

#[tokio::test]
async fn eager_partition_pruning_selects_only_matching_files() -> TestResult {
    // The bundled Taxi table has seven daily files, 25,000 rows per day.
    let harness = IcebergTestHarness::builder()
        .with_config(
            SessionConfig::new()
                .with_target_partitions(4)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;

    let (plan, rows) = harness
        .query(
            "SELECT pickup_date, COUNT(*) AS trips FROM taxi \
         WHERE pickup_date = DATE '2024-01-10' \
         GROUP BY pickup_date ORDER BY pickup_date",
        )
        .await?;
    insta::assert_snapshot!(format!("{plan}\n{rows}"), @"
    CoalescePartitionsExec
      ProjectionExec: expr=[pickup_date@0 as pickup_date, count(Int64(1))@1 as trips]
        AggregateExec: mode=FinalPartitioned, gby=[pickup_date@0 as pickup_date], aggr=[count(Int64(1))], ordering_mode=Sorted
          RepartitionExec: partitioning=Hash([pickup_date@0], 4), input_partitions=4
            AggregateExec: mode=Partial, gby=[pickup_date@0 as pickup_date], aggr=[count(Int64(1))], ordering_mode=Sorted
              FilterExec: pickup_date@0 = 2024-01-10
                RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1
                  DataSourceExec: format=iceberg, projection=[pickup_date], planned_files=1

    +-------------+-------+
    | pickup_date | trips |
    +-------------+-------+
    | 2024-01-10  | 25000 |
    +-------------+-------+
    ");
    Ok(())
}

#[tokio::test]
async fn eager_scan_retains_the_residual_filter() -> TestResult {
    let harness = IcebergTestHarness::builder()
        .with_config(
            SessionConfig::new()
                .with_target_partitions(4)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;

    let (plan, rows) = harness
        .query(
            "SELECT COUNT(*) AS trips FROM taxi \
         WHERE pickup_date = DATE '2024-01-08' AND trip_distance > 50",
        )
        .await?;
    insta::assert_snapshot!(format!("{plan}\n{rows}"), @"
    ProjectionExec: expr=[count(Int64(1))@0 as trips]
      AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
        CoalescePartitionsExec
          AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
            FilterExec: pickup_date@1 = 2024-01-08 AND trip_distance@0 > 50, projection=[]
              RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1
                DataSourceExec: format=iceberg, projection=[trip_distance, pickup_date], planned_files=1

    +-------+
    | trips |
    +-------+
    | 2     |
    +-------+
    ");
    Ok(())
}

#[tokio::test]
async fn pruning_all_files_keeps_one_empty_scan_partition() -> TestResult {
    let harness = IcebergTestHarness::builder()
        .with_config(
            SessionConfig::new()
                .with_target_partitions(4)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;

    let (plan, rows) = harness
        .query("SELECT COUNT(*) AS trips FROM taxi WHERE pickup_date = DATE '2025-01-01'")
        .await?;
    insta::assert_snapshot!(format!("{plan}\n{rows}"), @"
    ProjectionExec: expr=[count(Int64(1))@0 as trips]
      AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
        CoalescePartitionsExec
          AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
            FilterExec: pickup_date@0 = 2025-01-01, projection=[]
              RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1
                DataSourceExec: format=iceberg, projection=[pickup_date], planned_files=0

    +-------+
    | trips |
    +-------+
    | 0     |
    +-------+
    ");
    Ok(())
}

#[tokio::test]
async fn eager_multi_partition_limit_is_global() -> TestResult {
    let fixture = ScanFixture::builder()
        .with_file([0, 1])
        .with_file([2, 3])
        .with_file([4, 5])
        .build()
        .await?;
    let harness = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(3)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;

    // Select a constant: a global limit promises a count, not which unordered IDs win.
    let (plan, rows) = harness.query("SELECT 1 AS n FROM items LIMIT 3").await?;
    insta::assert_snapshot!(format!("{plan}\n{rows}"), @"
    ProjectionExec: expr=[1 as n]
      CoalescePartitionsExec: fetch=3
        DataSourceExec: format=iceberg, projection=[], planned_files=3, fetch=3

    +---+
    | n |
    +---+
    | 1 |
    | 1 |
    | 1 |
    +---+
    ");
    Ok(())
}

#[tokio::test]
async fn eager_ordered_limit_matches_lazy_results() -> TestResult {
    let fixture = ScanFixture::builder()
        .with_file([5, 1])
        .with_file([4, 0])
        .with_file([3, 2])
        .build()
        .await?;
    let eager = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(3)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;
    let lazy = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(SessionConfig::new().with_target_partitions(3))
        .build()
        .await?;

    let sql = "SELECT id FROM items ORDER BY id LIMIT 3";
    let (plan, rows) = eager.query(sql).await?;
    let (_, lazy_rows) = lazy.query(sql).await?;
    insta::assert_snapshot!(format!("{plan}\n{rows}"), @"
    SortPreservingMergeExec: [id@0 ASC NULLS LAST], fetch=3
      SortExec: TopK(fetch=3), expr=[id@0 ASC NULLS LAST], preserve_partitioning=[true]
        DataSourceExec: format=iceberg, projection=[id], planned_files=3

    +----+
    | id |
    +----+
    | 0  |
    | 1  |
    | 2  |
    +----+
    ");
    assert_eq!(rows, lazy_rows);
    Ok(())
}

#[tokio::test]
async fn eager_and_lazy_scans_apply_position_deletes() -> TestResult {
    let fixture = ScanFixture::builder()
        .with_file([0, 1])
        .with_file([2, 3])
        .with_position_delete(0, 0)
        .build()
        .await?;
    let eager = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(2)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;
    let lazy = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(SessionConfig::new().with_target_partitions(2))
        .build()
        .await?;

    let (_, rows) = eager.query("SELECT id FROM items ORDER BY id").await?;
    let (_, lazy_rows) = lazy.query("SELECT id FROM items ORDER BY id").await?;
    insta::assert_snapshot!(rows, @"
    +----+
    | id |
    +----+
    | 1  |
    | 2  |
    | 3  |
    +----+
    ");
    assert_eq!(rows, lazy_rows);
    Ok(())
}

#[test]
fn eager_scan_can_execute_outside_the_table_creation_runtime() -> TestResult {
    let table_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let fixture = table_runtime.block_on(
        ScanFixture::builder()
            .with_file([0, 1])
            .with_file([2, 3])
            .build(),
    )?;
    let query_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    query_runtime.block_on(async {
        let harness = IcebergTestHarness::builder()
            .with_table("items", fixture.table.clone())
            .with_config(
                SessionConfig::new()
                    .with_target_partitions(2)
                    .with_option_extension(IcebergDataFusionConfig::default())
                    .set_bool("iceberg.execution.eager_file_planning", true),
            )
            .build()
            .await?;
        let (_, rows) = harness.query("SELECT id FROM items ORDER BY id").await?;
        insta::assert_snapshot!(rows, @"
        +----+
        | id |
        +----+
        | 0  |
        | 1  |
        | 2  |
        | 3  |
        +----+
        ");
        Ok(())
    })
}

#[tokio::test]
async fn eager_discovery_fails_instead_of_exceeding_query_memory() -> TestResult {
    let fixture = ScanFixture::builder().with_file([0, 1]).build().await?;
    let runtime = Arc::new(
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(1)))
            .build()?,
    );
    let harness = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(2)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .with_runtime_env(runtime.clone())
        .build()
        .await?;

    let error = harness.plan("SELECT * FROM items").await.unwrap_err();
    assert!(
        matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
        "{error}"
    );
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn source_clones_and_streams_retain_descriptor_memory() -> TestResult {
    let fixture = ScanFixture::builder()
        .with_file([0, 1])
        .with_file([2, 3])
        .build()
        .await?;
    let runtime = Arc::new(
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(64 * 1024)))
            .build()?,
    );
    let harness = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(2)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .with_runtime_env(runtime.clone())
        .build()
        .await?;

    let plan = harness.plan("SELECT * FROM items").await?;
    let reserved = runtime.memory_pool.reserved();
    assert!(reserved > 0);
    let source = eager_source(plan.as_ref()).clone();
    drop(plan);
    assert_eq!(runtime.memory_pool.reserved(), reserved);
    let stream = source.open(0, harness.context().task_ctx())?;
    drop(source);
    assert_eq!(runtime.memory_pool.reserved(), reserved);
    drop(stream);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn replacement_assignments_have_independent_memory_reservations() -> TestResult {
    let fixture = ScanFixture::builder()
        .with_file([0, 1])
        .with_file([2, 3])
        .build()
        .await?;
    let runtime = Arc::new(
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(64 * 1024)))
            .build()?,
    );
    let harness = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(2)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .with_runtime_env(runtime.clone())
        .build()
        .await?;

    let plan = harness.plan("SELECT * FROM items").await?;
    let source = eager_source(plan.as_ref());
    let original_reserved = runtime.memory_pool.reserved();
    let task = source.planned_files().next().unwrap().clone();
    let replacement = source.with_planned_files(vec![vec![task.clone()]])?;
    let replacement_reserved = runtime.memory_pool.reserved() - original_reserved;
    assert!(replacement_reserved > 0);
    // The original and the successful replacement must survive a failed replacement.
    let error = source
        .with_planned_files(vec![vec![task; 4096]])
        .unwrap_err();
    assert!(matches!(error, DataFusionError::ResourcesExhausted(_)));
    assert_eq!(
        runtime.memory_pool.reserved(),
        original_reserved + replacement_reserved
    );
    drop(plan);
    assert_eq!(runtime.memory_pool.reserved(), replacement_reserved);
    drop(replacement);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn public_builder_preserves_projection_filters_and_an_explicit_reader() -> TestResult
{
    let harness = IcebergTestHarness::builder()
        .with_config(
            SessionConfig::new()
                .with_target_partitions(4)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;
    let table = harness.context().table_provider("taxi").await?;
    let table = table.downcast_ref::<IcebergCatalogTableProvider>().unwrap();
    let iceberg_table = table.catalog().load_table(table.table_ident()).await?;
    let projection = vec![table.schema().index_of("pickup_date")?];
    let reader = iceberg_table.reader_builder().with_batch_size(2).build();
    let context = harness.context().task_ctx();

    let lazy = IcebergTestHarness::builder()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .build()
        .await?;
    let builder = IcebergDataSourceBuilder::new(&context)
        .with_schema(table.schema())
        .with_table(iceberg_table)
        .with_projection(Some(&projection))
        // 19732 is 2024-01-10, selecting one daily Taxi file.
        .with_filters(&[col("pickup_date").eq(lit(ScalarValue::Date32(Some(19732))))])
        .with_reader(reader)
        .with_fetch(Some(3));
    let eager = builder.clone().build().await?;
    let lazy_source = builder
        .with_context(&lazy.context().task_ctx())
        .build()
        .await?;
    assert!(lazy_source.downcast_ref::<IcebergDataSource>().is_some());
    assert_eq!(
        eager
            .downcast_ref::<EagerIcebergDataSource>()
            .unwrap()
            .planned_files()
            .len(),
        1
    );
    let batches = collect(eager.open(0, context)?).await?;
    let lazy_batches = collect(lazy_source.open(0, lazy.context().task_ctx())?).await?;
    assert!(
        batches
            .iter()
            .chain(&lazy_batches)
            .all(|batch| batch.num_rows() <= 2)
    );
    let rows = pretty_format_batches(&batches)?.to_string();
    insta::assert_snapshot!(rows, @"
    +-------------+
    | pickup_date |
    +-------------+
    | 2024-01-10  |
    | 2024-01-10  |
    | 2024-01-10  |
    +-------------+
    ");
    assert_eq!(rows, pretty_format_batches(&lazy_batches)?.to_string());
    Ok(())
}

#[tokio::test]
async fn builder_preserves_session_reader_settings_at_execution() -> TestResult {
    let fixture = ScanFixture::builder().with_file([0, 1, 2]).build().await?;
    let harness = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(1)
                .with_batch_size(1),
        )
        .build()
        .await?;

    let plan = harness.plan("SELECT * FROM items").await?;
    let source = scan_source(plan.as_ref());
    assert!(source.downcast_ref::<IcebergDataSource>().is_some());
    // A different execution context must not silently replace the construction settings.
    let execution_context = SessionContext::new().task_ctx();
    assert!(source.open(1, execution_context.clone()).is_err());
    let batches = collect(source.open(0, execution_context)?).await?;
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.num_rows())
            .collect::<Vec<_>>(),
        vec![1, 1, 1]
    );
    Ok(())
}

#[tokio::test]
async fn replacement_groups_preserve_order_empty_partitions_and_repeated_tasks()
-> TestResult {
    let fixture = ScanFixture::builder()
        .with_file([0])
        .with_file([1])
        .with_file([2])
        .build()
        .await?;
    let harness = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(2)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;
    let plan = harness.plan("SELECT * FROM items").await?;
    let source = eager_source(plan.as_ref());
    let files = files_by_name(source);

    let assigned = source.with_planned_files(vec![
        vec![files[2].clone(), files[0].clone(), files[2].clone()],
        vec![],
    ])?;
    insta::assert_snapshot!(file_groups(&assigned)?, @"
    partition 0: [data-2.parquet, data-0.parquet, data-2.parquet]
    partition 1: []
    ");
    assert_eq!(assigned.planned_files().len(), 3);
    let mut all_files = assigned.planned_files();
    all_files.next();
    assert_eq!(all_files.len(), 2);
    let narrowed = assigned.with_planned_files(vec![vec![], vec![files[0].clone()]])?;
    insta::assert_snapshot!(file_groups(&narrowed)?, @"
    partition 0: []
    partition 1: [data-0.parquet]
    ");
    assert_eq!(source.planned_files().len(), 3);
    let batches = collect(assigned.open(0, harness.context().task_ctx())?).await?;
    // Show the selected values as well as their multiplicity, not just a row count.
    insta::assert_snapshot!(batches_to_sort_string(&batches), @"
    +----+
    | id |
    +----+
    | 0  |
    | 2  |
    | 2  |
    +----+
    ");
    assert!(
        collect(assigned.open(1, harness.context().task_ctx())?)
            .await?
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn reconstruction_preserves_groups_reader_settings_fetch_and_deletes() -> TestResult
{
    let fixture = ScanFixture::builder()
        .with_file([0, 1])
        .with_file([2, 3])
        .with_position_delete(0, 0)
        .build()
        .await?;
    let harness = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(2)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;
    let plan = harness.plan("SELECT * FROM items").await?;
    let original = eager_source(plan.as_ref());
    let files = files_by_name(original);
    let schema = original.schema().clone();
    let file_io = fixture.table.file_io().clone();
    let reader = fixture
        .table
        .reader_builder()
        .with_batch_size(1)
        .with_data_file_concurrency_limit(1)
        .build();
    let context = harness.context().task_ctx();
    // Reconstruction has no table/catalog argument and cannot rediscover files.
    drop(plan);
    drop(harness);

    let rebuilt = IcebergDataSourceBuilder::new(&context)
        .with_schema(schema)
        .with_planned_files(vec![vec![], files])
        .with_file_io(file_io)
        .with_reader(reader)
        .with_fetch(Some(2))
        .build()?;
    let source = rebuilt.downcast_ref::<EagerIcebergDataSource>().unwrap();
    insta::assert_snapshot!(file_groups(source)?, @"
    partition 0: []
    partition 1: [data-0.parquet, data-1.parquet]
    ");
    assert!(collect(source.open(0, context.clone())?).await?.is_empty());
    assert!(source.open(2, context.clone()).is_err());
    let batches = collect(source.open(1, context.clone())?).await?;
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.num_rows())
            .collect::<Vec<_>>(),
        vec![1, 1]
    );
    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @"
    +----+
    | id |
    +----+
    | 1  |
    | 2  |
    +----+
    ");
    // Re-executing a reconstructed source must not consume its retained descriptors.
    let again = collect(source.open(1, context)?).await?;
    assert_eq!(
        pretty_format_batches(&batches)?.to_string(),
        pretty_format_batches(&again)?.to_string()
    );
    Ok(())
}

#[tokio::test]
async fn reconstruction_requires_groups_but_allows_empty_partitions() -> TestResult {
    let fixture = ScanFixture::builder().with_file([0, 1]).build().await?;
    let harness = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(1)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;
    let plan = harness.plan("SELECT * FROM items").await?;
    let source = eager_source(plan.as_ref());
    let context = harness.context().task_ctx();
    let reader = fixture.table.reader_builder().build();

    let error = source.with_planned_files(vec![]).unwrap_err();
    insta::assert_snapshot!(error.to_string(), @"Execution error: Iceberg scan requires at least one file group");
    let empty = IcebergDataSourceBuilder::new(&context)
        .with_schema(source.schema().clone())
        .with_planned_files(vec![])
        .with_file_io(fixture.table.file_io().clone())
        .with_reader(reader.clone())
        .build()
        .unwrap_err();
    assert_eq!(empty.to_string(), error.to_string());
    let empty_partitions = IcebergDataSourceBuilder::new(&context)
        .with_schema(source.schema().clone())
        .with_planned_files(vec![vec![], vec![]])
        .with_file_io(fixture.table.file_io().clone())
        .with_reader(reader)
        .build()?;
    let empty_partitions = empty_partitions
        .downcast_ref::<EagerIcebergDataSource>()
        .unwrap();
    insta::assert_snapshot!(file_groups(empty_partitions)?, @"
    partition 0: []
    partition 1: []
    ");
    assert_eq!(empty_partitions.planned_files().len(), 0);
    assert!(
        collect(empty_partitions.open(0, context.clone())?)
            .await?
            .is_empty()
    );
    assert!(
        collect(empty_partitions.open(1, context)?)
            .await?
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn reconstruction_uses_the_replacement_context_memory_pool() -> TestResult {
    let fixture = ScanFixture::builder().with_file([0, 1]).build().await?;
    let harness = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(
            SessionConfig::new()
                .with_target_partitions(1)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true),
        )
        .build()
        .await?;
    let plan = harness.plan("SELECT * FROM items").await?;
    let source = eager_source(plan.as_ref());
    let context = harness.context().task_ctx();
    let reader = fixture.table.reader_builder().build();
    let limited_runtime = Arc::new(
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(1)))
            .build()?,
    );
    let limited = IcebergTestHarness::builder()
        .with_table("items", fixture.table.clone())
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_runtime_env(limited_runtime.clone())
        .build()
        .await?;
    let rebuilt = IcebergDataSourceBuilder::new(&context)
        .with_schema(source.schema().clone())
        .with_planned_files(vec![source.planned_files().cloned().collect()])
        .with_file_io(fixture.table.file_io().clone())
        .with_reader(reader)
        .with_context(&limited.context().task_ctx())
        .build()
        .unwrap_err();
    assert!(matches!(rebuilt, DataFusionError::ResourcesExhausted(_)));
    assert_eq!(limited_runtime.memory_pool.reserved(), 0);
    Ok(())
}

fn scan_source(plan: &dyn ExecutionPlan) -> &dyn DataSource {
    if let Some(scan) = plan.downcast_ref::<DataSourceExec>() {
        return scan.data_source().as_ref();
    }
    let children = plan.children();
    assert_eq!(children.len(), 1, "expected a query over one Iceberg scan");
    scan_source(children[0].as_ref())
}

fn eager_source(plan: &dyn ExecutionPlan) -> &EagerIcebergDataSource {
    scan_source(plan)
        .downcast_ref::<EagerIcebergDataSource>()
        .expect("expected eager Iceberg source")
}

fn file_name(task: &FileScanTask) -> String {
    Path::new(task.data_file_path())
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

fn files_by_name(source: &EagerIcebergDataSource) -> Vec<FileScanTask> {
    let mut files = source.planned_files().cloned().collect::<Vec<_>>();
    files.sort_by_key(file_name);
    files
}

fn file_groups(source: &EagerIcebergDataSource) -> Result<String> {
    (0..source.output_partitioning().partition_count())
        .map(|partition| {
            let names = source
                .planned_partition_files(partition)?
                .map(file_name)
                .collect::<Vec<_>>();
            Ok(format!("partition {partition}: [{}]", names.join(", ")))
        })
        .collect::<Result<Vec<_>>>()
        .map(|lines| lines.join("\n"))
}
