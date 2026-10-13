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

use datafusion::common::{Result, exec_datafusion_err, stats::Precision};
use datafusion::datasource::TableProvider;
use datafusion::datasource::source::{DataSource, DataSourceExec};
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::ExecutionPlanProperties;
use datafusion::physical_plan::common::collect;
use datafusion::prelude::{SessionConfig, col, lit};
use datafusion::scalar::ScalarValue;
use iceberg::Runtime;
use iceberg::table::Table;
use iceberg_datafusion::read::IcebergDataSource;
use iceberg_datafusion::test_utils::IcebergTestHarness;
use iceberg_datafusion::{
    IcebergCatalogTableProvider, IcebergPlanningConfig, IcebergStaticTableProvider,
};

fn eager_config(max_files: usize) -> SessionConfig {
    SessionConfig::new()
        .with_target_partitions(4)
        .with_option_extension(IcebergPlanningConfig::default())
        .set_bool("iceberg_local.plan_files", true)
        .set_usize("iceberg_local.planning_max_files", max_files)
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
    let eager = IcebergTestHarness::with_config(eager_config(100)).await?;
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
    assert_eq!(lazy_plan.output_partitioning().partition_count(), 1);
    assert_eq!(eager_source.planned_files().unwrap().tasks().len(), 7);
    assert_eq!(eager_plan.output_partitioning().partition_count(), 4);
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
            .tasks()
            .is_empty()
    );
    assert_eq!(empty.output_partitioning().partition_count(), 1);

    Ok(())
}

#[tokio::test]
async fn eager_partition_count_is_bounded_by_selected_files() -> Result<()> {
    let eager =
        IcebergTestHarness::with_config(eager_config(100).with_target_partitions(16))
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
    let eager = IcebergTestHarness::with_config(eager_config(100)).await?;
    let lazy = IcebergTestHarness::new().await?;
    for (query, planned_files) in [
        (
            "SELECT pickup_date, COUNT(*) AS trips FROM taxi \
             WHERE pickup_date = DATE '2024-01-10' \
             GROUP BY pickup_date ORDER BY pickup_date",
            1,
        ),
        (
            "SELECT COUNT(*) AS trips FROM taxi \
             WHERE pickup_date = DATE '2025-01-01'",
            0,
        ),
    ] {
        let (plan, eager_batches) = eager.query(query).await?;
        let (_, lazy_batches) = lazy.query(query).await?;
        assert!(
            plan.contains(&format!("planned_files={planned_files}")),
            "eager scan not visible in plan: {plan}"
        );
        assert_eq!(eager_batches, lazy_batches);
    }
    Ok(())
}

#[tokio::test]
async fn eager_multi_partition_limit_is_global() -> Result<()> {
    let eager = IcebergTestHarness::with_config(eager_config(100)).await?;
    let (plan, batches) = eager.query("SELECT pickup_date FROM taxi LIMIT 3").await?;
    assert!(plan.contains("planned_files=7"), "unexpected plan: {plan}");
    assert_eq!(batches.matches("2024-01-").count(), 3, "{batches}");
    Ok(())
}

#[tokio::test]
async fn eager_ordered_limit_matches_lazy_results() -> Result<()> {
    let eager = IcebergTestHarness::with_config(eager_config(100)).await?;
    let lazy = IcebergTestHarness::new().await?;
    let sql = "SELECT vendor_id, pickup_at FROM taxi \
               ORDER BY pickup_at, vendor_id, pickup_location_id LIMIT 3";
    let (plan, eager_batches) = eager.query(sql).await?;
    let (_, lazy_batches) = lazy.query(sql).await?;
    assert!(plan.contains("planned_files=7"), "unexpected plan: {plan}");
    assert_eq!(eager_batches, lazy_batches);
    Ok(())
}

#[tokio::test]
async fn eager_scan_uses_the_tables_custom_runtime() -> Result<()> {
    let harness = IcebergTestHarness::with_config(eager_config(100)).await?;
    let provider = harness.context().table_provider("taxi").await?;
    let catalog_provider = provider
        .downcast_ref::<IcebergCatalogTableProvider>()
        .expect("the fixture creates a catalog Iceberg provider");
    let table = catalog_provider
        .catalog()
        .load_table(catalog_provider.table_ident())
        .await
        .map_err(iceberg_datafusion::to_datafusion_error)?;
    let io_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let table = Table::builder()
        .file_io(table.file_io().clone())
        .metadata(table.metadata_ref())
        .identifier(table.identifier().clone())
        .runtime(Runtime::new(&io_runtime))
        .build()
        .map_err(iceberg_datafusion::to_datafusion_error)?;
    let provider = IcebergStaticTableProvider::try_new_from_table(table).await?;
    let state = harness.context().state();
    let plan = provider.scan(&state, None, &[], None).await?;
    assert_eq!(
        iceberg_source(plan.as_ref())
            .planned_files()
            .unwrap()
            .tasks()
            .len(),
        7
    );
    let mut rows = 0;
    for partition in 0..plan.output_partitioning().partition_count() {
        let batches = collect(plan.execute(partition, state.task_ctx())?).await?;
        rows += batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
    }
    assert_eq!(rows, 175_000);
    tokio::task::spawn_blocking(move || drop(io_runtime))
        .await
        .map_err(|error| {
            exec_datafusion_err!("failed to shut down test runtime: {error}")
        })?;
    Ok(())
}

#[tokio::test]
async fn eager_tasks_respect_the_query_memory_pool() -> Result<()> {
    let runtime = Arc::new(
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(1)))
            .build()?,
    );
    let eager =
        IcebergTestHarness::with_config_and_runtime(eager_config(100), runtime).await?;
    let provider = eager.context().table_provider("taxi").await?;
    let error = provider
        .scan(&eager.context().state(), None, &[], None)
        .await
        .expect_err("file metadata cannot fit in one byte");
    assert!(
        error.to_string().contains("Iceberg file planning"),
        "{error}"
    );
    Ok(())
}

#[tokio::test]
async fn planning_file_limit_rejects_instead_of_truncating() -> Result<()> {
    let eager = IcebergTestHarness::with_config(eager_config(2)).await?;
    let provider = eager.context().table_provider("taxi").await?;
    let error = provider
        .scan(&eager.context().state(), None, &[], None)
        .await
        .expect_err("seven files exceed the two-file planning limit");
    assert!(
        error.to_string().contains("planning_max_files=2"),
        "{error}"
    );
    Ok(())
}
