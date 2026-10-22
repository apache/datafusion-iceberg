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
use std::result::Result as StdResult;
use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::common::{Result, stats::Precision};
use datafusion::datasource::TableProvider;
use datafusion::datasource::source::{DataSource, DataSourceExec};
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::common::collect;
use datafusion::physical_plan::{ExecutionPlanProperties, collect as collect_plan};
use datafusion::prelude::{SessionConfig, SessionContext, col, lit};
use datafusion::scalar::ScalarValue;
use iceberg_datafusion::read::IcebergDataSource;
use iceberg_datafusion::test_utils::IcebergTestHarness;
use iceberg_datafusion::{
    IcebergCatalogTableProvider, IcebergPlanningConfig, IcebergStaticTableProvider,
};

mod common;

use common::position_deletes::table_with_position_delete;

fn eager_config() -> SessionConfig {
    SessionConfig::new()
        .with_target_partitions(4)
        .with_option_extension(IcebergPlanningConfig::default())
        .set_bool("iceberg.eager_file_planning", true)
}

fn iceberg_source(
    plan: &dyn datafusion::physical_plan::ExecutionPlan,
) -> &IcebergDataSource {
    plan.downcast_ref::<DataSourceExec>()
        .expect("the Iceberg provider creates a DataSourceExec")
        .data_source()
        .downcast_ref::<IcebergDataSource>()
        .expect("the plan contains the Iceberg data source")
}

#[tokio::test]
async fn plans_file_tasks_before_execution_without_changing_statistics() -> Result<()> {
    let eager = IcebergTestHarness::with_config(eager_config()).await?;
    let lazy = IcebergTestHarness::new().await?;
    let eager_provider = eager.context().table_provider("taxi").await?;
    let lazy_provider = lazy.context().table_provider("taxi").await?;
    let eager_plan = eager_provider
        .scan(&eager.context().state(), None, &[], None)
        .await?;
    let lazy_plan = lazy_provider
        .scan(&lazy.context().state(), None, &[], None)
        .await?;
    let eager_source = iceberg_source(eager_plan.as_ref());
    let lazy_source = iceberg_source(lazy_plan.as_ref());

    assert!(lazy_source.planned_files().is_none());
    assert!(lazy_source.planned_partition_tasks(0)?.is_none());
    assert!(lazy_source.planned_partition_tasks(1).is_err());
    assert!(eager_source.planned_partition_tasks(4).is_err());
    let tasks = eager_source.planned_files().unwrap();
    for (partition, indices) in [vec![0, 4], vec![1, 5], vec![2, 6], vec![3]]
        .into_iter()
        .enumerate()
    {
        let assigned: Vec<_> = eager_source
            .planned_partition_tasks(partition)?
            .unwrap()
            .map(std::ptr::from_ref)
            .collect();
        let expected: Vec<_> = indices
            .into_iter()
            .map(|index| std::ptr::from_ref(&tasks[index]))
            .collect();
        assert_eq!(assigned, expected);
    }

    assert_eq!(lazy_plan.output_partitioning().partition_count(), 1);
    assert_eq!(eager_source.planned_files().unwrap().len(), 7);
    assert_eq!(eager_plan.output_partitioning().partition_count(), 4);
    assert!(eager_source.open(4, eager.context().task_ctx()).is_err());
    assert!(lazy_source.open(1, lazy.context().task_ctx()).is_err());
    assert_eq!(
        eager_source.partition_statistics(None)?.num_rows,
        Precision::Absent
    );
    assert_eq!(
        eager_source.partition_statistics(None)?.total_byte_size,
        lazy_source.partition_statistics(None)?.total_byte_size
    );

    let empty = eager_provider
        .scan(
            &eager.context().state(),
            None,
            &[col("pickup_date").eq(lit(ScalarValue::Date32(Some(0))))],
            None,
        )
        .await?;
    assert!(
        iceberg_source(empty.as_ref())
            .planned_files()
            .unwrap()
            .is_empty()
    );
    assert_eq!(empty.output_partitioning().partition_count(), 1);

    Ok(())
}

#[tokio::test]
async fn eager_partition_count_is_bounded_by_selected_files() -> Result<()> {
    let eager =
        IcebergTestHarness::with_config(eager_config().with_target_partitions(16))
            .await?;
    let provider = eager.context().table_provider("taxi").await?;
    let plan = provider
        .scan(&eager.context().state(), None, &[], None)
        .await?;
    assert_eq!(plan.output_partitioning().partition_count(), 7);
    Ok(())
}

#[tokio::test]
async fn filtered_and_empty_eager_scans_match_lazy_results() -> Result<()> {
    let eager = IcebergTestHarness::with_config(eager_config()).await?;
    let lazy = IcebergTestHarness::new().await?;
    for (name, query) in [
        (
            "partition_pruning",
            "SELECT pickup_date, COUNT(*) AS trips FROM taxi \
             WHERE pickup_date = DATE '2024-01-10' \
             GROUP BY pickup_date ORDER BY pickup_date",
        ),
        (
            "residual_filter",
            "SELECT COUNT(*) AS trips FROM taxi \
             WHERE pickup_date = DATE '2024-01-08' AND trip_distance > 50",
        ),
        (
            "empty_scan",
            "SELECT COUNT(*) AS trips FROM taxi \
             WHERE pickup_date = DATE '2025-01-01'",
        ),
    ] {
        let (plan, eager_batches) = eager.query(query).await?;
        let (_, lazy_batches) = lazy.query(query).await?;
        insta::assert_snapshot!(name, format!("{plan}\n{eager_batches}"));
        assert_eq!(eager_batches, lazy_batches);
    }
    Ok(())
}

#[tokio::test]
async fn eager_multi_partition_limit_is_global() -> Result<()> {
    let eager = IcebergTestHarness::with_config(eager_config()).await?;
    let (plan, batches) = eager.query("SELECT 1 AS n FROM taxi LIMIT 3").await?;
    insta::assert_snapshot!(format!("{plan}\n{batches}"));
    Ok(())
}

#[tokio::test]
async fn eager_ordered_limit_matches_lazy_results() -> Result<()> {
    let eager = IcebergTestHarness::with_config(eager_config()).await?;
    let lazy = IcebergTestHarness::new().await?;
    let sql = "SELECT vendor_id, pickup_at FROM taxi \
               ORDER BY pickup_at, vendor_id, pickup_location_id LIMIT 3";
    let (plan, eager_batches) = eager.query(sql).await?;
    let (_, lazy_batches) = lazy.query(sql).await?;
    insta::assert_snapshot!(format!("{plan}\n{eager_batches}"));
    assert_eq!(eager_batches, lazy_batches);
    Ok(())
}

#[tokio::test]
async fn eager_and_lazy_scans_apply_position_deletes() -> StdResult<(), Box<dyn Error>> {
    let fixture = table_with_position_delete().await?;
    let eager = SessionContext::new_with_config(eager_config());
    let table = fixture.table.clone();
    let provider = IcebergStaticTableProvider::try_new_from_table(table).await?;
    let lazy = SessionContext::new();

    let state = eager.state();
    let eager_plan = provider.scan(&state, None, &[], None).await?;
    // Both unpartitioned files receive the delete task; the reader matches its path.
    assert_eq!(
        iceberg_source(eager_plan.as_ref())
            .planned_files()
            .expect("eager plan has files")
            .iter()
            .filter(|task| !task.deletes().is_empty())
            .count(),
        2
    );
    assert_eq!(
        iceberg_source(eager_plan.as_ref())
            .planned_files()
            .expect("eager plan has files")
            .len(),
        2
    );
    let lazy_plan = provider.scan(&lazy.state(), None, &[], None).await?;
    let eager_batches = collect_plan(eager_plan, state.task_ctx()).await?;
    let lazy_batches = collect_plan(lazy_plan, lazy.task_ctx()).await?;
    let ids = |batches: &[datafusion::arrow::array::RecordBatch]| {
        let mut ids: Vec<i64> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        ids.sort_unstable();
        ids
    };
    assert_eq!(ids(&eager_batches), vec![1, 2, 3]);
    assert_eq!(ids(&eager_batches), ids(&lazy_batches));
    Ok(())
}

#[test]
fn eager_scan_uses_the_tables_custom_runtime() -> Result<()> {
    let io_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let (harness, provider) = io_runtime.block_on(async {
        let harness = IcebergTestHarness::with_config(eager_config()).await?;
        let provider = harness.context().table_provider("taxi").await?;
        let catalog_provider = provider
            .downcast_ref::<IcebergCatalogTableProvider>()
            .expect("the fixture creates a catalog Iceberg provider");
        let table = catalog_provider
            .catalog()
            .load_table(catalog_provider.table_ident())
            .await
            .map_err(iceberg_datafusion::to_datafusion_error)?;
        let provider = IcebergStaticTableProvider::try_new_from_table(table).await?;
        Ok::<_, datafusion::error::DataFusionError>((harness, provider))
    })?;
    // The table retains its own runtime while planning is called from another runtime.
    let planning_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    planning_runtime.block_on(async {
        let state = harness.context().state();
        let plan = provider.scan(&state, None, &[], None).await?;
        assert_eq!(
            iceberg_source(plan.as_ref()).planned_files().unwrap().len(),
            7
        );
        let mut rows = 0;
        for partition in 0..plan.output_partitioning().partition_count() {
            let batches = collect(plan.execute(partition, state.task_ctx())?).await?;
            rows += batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
        }
        assert_eq!(rows, 175_000);
        Ok(())
    })
}

#[tokio::test]
async fn eager_tasks_respect_the_query_memory_pool() -> Result<()> {
    let runtime = Arc::new(
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(1)))
            .build()?,
    );
    let eager =
        IcebergTestHarness::with_config_and_runtime(eager_config(), runtime.clone())
            .await?;
    let provider = eager.context().table_provider("taxi").await?;
    let error = provider
        .scan(&eager.context().state(), None, &[], None)
        .await
        .expect_err("file metadata cannot fit in one byte");
    assert!(
        matches!(
            error,
            datafusion::error::DataFusionError::ResourcesExhausted(_)
        ),
        "{error}"
    );
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn retained_tasks_follow_plan_and_stream_lifetimes() -> Result<()> {
    let runtime = Arc::new(
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(1024 * 1024)))
            .build()?,
    );
    let harness =
        IcebergTestHarness::with_config_and_runtime(eager_config(), runtime.clone())
            .await?;
    let provider = harness.context().table_provider("taxi").await?;
    let plan = provider
        .scan(&harness.context().state(), None, &[], None)
        .await?;
    let reserved = runtime.memory_pool.reserved();
    assert!(reserved > 0);
    let source = iceberg_source(plan.as_ref()).clone();
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
async fn public_builder_plans_projected_filtered_scans() -> Result<()> {
    let harness = IcebergTestHarness::new().await?;
    let provider = harness.context().table_provider("taxi").await?;
    let provider = provider
        .downcast_ref::<IcebergCatalogTableProvider>()
        .unwrap();
    let table = provider
        .catalog()
        .load_table(provider.table_ident())
        .await
        .map_err(iceberg_datafusion::to_datafusion_error)?;
    let schema = provider.schema();
    let projection = vec![schema.index_of("pickup_date")?];
    let pickup_date = ScalarValue::Date32(Some(19732)); // 2024-01-10
    let source = IcebergDataSource::builder(table, schema)
        .with_projection(Some(&projection))
        .with_filters(&[col("pickup_date").eq(lit(pickup_date))])
        .with_eager_planning(true)
        .build(harness.context().task_ctx())
        .await?;
    assert_eq!(source.projection(), &["pickup_date"]);
    assert_eq!(source.planned_files().unwrap().len(), 1);
    assert_eq!(source.output_partitioning().partition_count(), 1);
    let batches = collect(source.open(0, harness.context().task_ctx())?).await?;
    assert!(!batches.is_empty());
    assert!(
        batches
            .iter()
            .all(|batch| batch.schema() == source.schema())
    );
    Ok(())
}
