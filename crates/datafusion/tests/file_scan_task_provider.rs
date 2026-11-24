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

use datafusion::arrow::array::{ArrayRef, Int64Array, RecordBatch};
use datafusion::common::Result;
use datafusion::datasource::source::{DataSource, DataSourceExec};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use datafusion::physical_plan::{
    ExecutionPlan, ExecutionPlanProperties, collect, displayable,
};
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg_datafusion::read::{EagerFileScanTaskProvider, IcebergDataSource};
use iceberg_datafusion::{IcebergDataFusionConfig, IcebergTableProviderFactory};
use iceberg_datafusion_test_tools::{
    FixtureCatalog, FixtureTable, IcebergTestHarness, ReadCountingStorageFactory,
    create_taxi_table,
};

#[tokio::test]
async fn lazy_scan_filters_and_projects_taxi_rows() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(create_taxi_table().await?)
        .with_config_option("datafusion.execution.target_partitions", 2)
        .build()
        .await?;

    let (plan, rows) = harness
        .query(
            "SELECT pickup_date, COUNT(*) AS trips FROM taxi \
             WHERE pickup_date IN (DATE '2024-01-10', DATE '2024-01-11') \
             GROUP BY pickup_date ORDER BY pickup_date",
        )
        .await?;

    insta::assert_snapshot!(plan, @"
    SortPreservingMergeExec: [pickup_date@0 ASC NULLS LAST]
      ProjectionExec: expr=[pickup_date@0 as pickup_date, count(Int64(1))@1 as trips]
        SortExec: expr=[pickup_date@0 ASC NULLS LAST], preserve_partitioning=[true]
          AggregateExec: mode=FinalPartitioned, gby=[pickup_date@0 as pickup_date], aggr=[count(Int64(1))]
            RepartitionExec: partitioning=Hash([pickup_date@0], 2), input_partitions=2
              AggregateExec: mode=Partial, gby=[pickup_date@0 as pickup_date], aggr=[count(Int64(1))]
                FilterExec: pickup_date@0 = 2024-01-10 OR pickup_date@0 = 2024-01-11
                  DataSourceExec: format=iceberg, projection=[pickup_date], provider=lazy, predicate=(pickup_date = 2024-01-10) OR (pickup_date = 2024-01-11)
    ");
    insta::assert_snapshot!(rows, @"
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
async fn eager_scan_plans_only_the_matching_taxi_files() -> Result<()> {
    // Taxi has seven daily files; this query needs only two of them.
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(create_taxi_table().await?)
        .with_session_extension(|mut state| {
            state
                .config()
                .get_or_insert_default()
                .options_mut()
                .extensions
                .insert(IcebergDataFusionConfig::default());
            state
        })
        .with_config_option("datafusion.execution.target_partitions", 2)
        .with_config_option("iceberg.execution.eager_file_planning", true)
        .build()
        .await?;

    let (plan, rows) = harness
        .query(
            "SELECT pickup_date, COUNT(*) AS trips FROM taxi \
             WHERE pickup_date IN (DATE '2024-01-10', DATE '2024-01-11') \
             GROUP BY pickup_date ORDER BY pickup_date",
        )
        .await?;

    insta::assert_snapshot!(plan, @"
    SortPreservingMergeExec: [pickup_date@0 ASC NULLS LAST]
      ProjectionExec: expr=[pickup_date@0 as pickup_date, count(Int64(1))@1 as trips]
        SortExec: expr=[pickup_date@0 ASC NULLS LAST], preserve_partitioning=[true]
          AggregateExec: mode=FinalPartitioned, gby=[pickup_date@0 as pickup_date], aggr=[count(Int64(1))]
            RepartitionExec: partitioning=Hash([pickup_date@0], 2), input_partitions=2
              AggregateExec: mode=Partial, gby=[pickup_date@0 as pickup_date], aggr=[count(Int64(1))]
                FilterExec: pickup_date@0 = 2024-01-10 OR pickup_date@0 = 2024-01-11
                  DataSourceExec: format=iceberg, projection=[pickup_date], provider=eager, planned_files=2, predicates=[(pickup_date = 2024-01-10) OR (pickup_date = 2024-01-11)]
    ");
    insta::assert_snapshot!(rows, @"
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
async fn eager_count_skips_scan_when_no_files_match() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(create_taxi_table().await?)
        .with_session_extension(|mut state| {
            state
                .config()
                .get_or_insert_default()
                .options_mut()
                .extensions
                .insert(IcebergDataFusionConfig::default());
            state
        })
        .with_config_option("datafusion.execution.target_partitions", 2)
        .with_config_option("iceberg.execution.eager_file_planning", true)
        .build()
        .await?;

    let (plan, rows) = harness
        .query("SELECT COUNT(*) AS trips FROM taxi WHERE pickup_date = DATE '2025-01-01'")
        .await?;

    insta::assert_snapshot!(plan, @"
    ProjectionExec: expr=[0 as trips]
      PlaceholderRowExec
    ");
    insta::assert_snapshot!(rows, @"
    +-------+
    | trips |
    +-------+
    | 0     |
    +-------+
    ");
    Ok(())
}

#[tokio::test]
async fn replacing_the_provider_preserves_the_fetch_limit() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(create_taxi_table().await?)
        .with_session_extension(|mut state| {
            state
                .config()
                .get_or_insert_default()
                .options_mut()
                .extensions
                .insert(IcebergDataFusionConfig::default());
            state
        })
        .with_config_option("datafusion.execution.target_partitions", 2)
        .with_config_option("iceberg.execution.eager_file_planning", true)
        .build()
        .await?;
    let plan = harness
        .physical_plan("SELECT pickup_date FROM taxi LIMIT 2")
        .await?;
    let source = iceberg_source(&plan).unwrap();
    let assigned_file = eager_files(source).tasks().next().unwrap().clone();

    let assigned = source.with_file_scan_task_provider(Arc::new(
        EagerFileScanTaskProvider::try_new(vec![vec![assigned_file]])?,
    ))?;
    let assigned_plan = DataSourceExec::from_data_source(assigned);

    insta::assert_snapshot!(displayable(assigned_plan.as_ref()).indent(true).to_string(), @"DataSourceExec: format=iceberg, projection=[pickup_date], provider=eager, planned_files=1, fetch=2");
    Ok(())
}

#[tokio::test]
async fn eager_file_statistics_apply_fetch_per_partition() -> Result<()> {
    let mut table = empty_items().await?;
    table
        .append()
        .file(id_batch(vec![1, 2, 3])?)
        .file(id_batch(vec![4, 5, 6])?)
        .commit()
        .await?;
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(table)
        .with_session_extension(|mut state| {
            state
                .config()
                .get_or_insert_default()
                .options_mut()
                .extensions
                .insert(IcebergDataFusionConfig::default());
            state
        })
        .with_config_option("datafusion.execution.target_partitions", 2)
        .with_config_option("iceberg.execution.eager_file_planning", true)
        .build()
        .await?;

    // Each of the two file groups contributes three rows to the assignment.
    let plan = harness.physical_plan("SELECT id FROM items").await?;
    insta::assert_snapshot!(harness.source_statistics(&plan)?, @"
    all partitions (2): Rows=Inexact(6), Bytes=Inexact(48), [(Col[0]:)]
    partition 0: Rows=Inexact(3), Bytes=Inexact(24), [(Col[0]:)]
    partition 1: Rows=Inexact(3), Bytes=Inexact(24), [(Col[0]:)]
    ");

    // Fetch caps each source partition at two rows, not the whole assignment.
    let plan = harness
        .physical_plan("SELECT id FROM items LIMIT 2")
        .await?;
    insta::assert_snapshot!(harness.source_statistics(&plan)?, @"
    all partitions (2): Rows=Inexact(4), Bytes=Inexact(32), [(Col[0]:)]
    partition 0: Rows=Inexact(2), Bytes=Inexact(16), [(Col[0]:)]
    partition 1: Rows=Inexact(2), Bytes=Inexact(16), [(Col[0]:)]
    ");
    Ok(())
}

#[tokio::test]
async fn the_scan_keeps_its_construction_time_reader_settings() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(create_taxi_table().await?)
        .with_config_option("datafusion.execution.target_partitions", 1)
        .with_config_option("datafusion.execution.batch_size", 2000)
        .build()
        .await?;
    let plan = harness
        .physical_plan("SELECT pickup_date FROM taxi")
        .await?;
    let source = iceberg_source(&plan).unwrap();

    // Open with default settings, not the settings used to construct the scan.
    let batches = source
        .open(0, Arc::new(TaskContext::default()))?
        .try_collect::<Vec<_>>()
        .await?;

    assert_eq!(batches.iter().map(RecordBatch::num_rows).max(), Some(2000));
    Ok(())
}

#[tokio::test]
async fn reset_scans_read_the_same_files_independently() -> Result<()> {
    let mut table = empty_items().await?;
    table
        .append()
        .file(id_batch(vec![1, 2])?)
        .file(id_batch(vec![3, 4])?)
        .commit()
        .await?;
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(table)
        .with_config_option("datafusion.execution.target_partitions", 4)
        .build()
        .await?;
    let scan = harness.scan().await?;
    assert_eq!(scan.output_partitioning().partition_count(), 4);
    let next_run = reset_plan_states(scan.clone())?;
    let context = harness.state().task_ctx();
    let (original_rows, reset_rows) =
        tokio::try_join!(collect(scan, context.clone()), collect(next_run, context),)?;

    let expected = [
        "+----+", "| id |", "+----+", "| 1  |", "| 2  |", "| 3  |", "| 4  |", "+----+",
    ];
    datafusion::assert_batches_sorted_eq!(expected, &original_rows);
    datafusion::assert_batches_sorted_eq!(expected, &reset_rows);
    Ok(())
}

#[tokio::test]
async fn a_late_partition_reads_only_the_unclaimed_file() -> Result<()> {
    let mut table = empty_items().await?;
    table
        .append()
        .file(id_batch(vec![1])?)
        .file(id_batch(vec![2])?)
        .commit()
        .await?;
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(table)
        .with_config_option("datafusion.execution.target_partitions", 2)
        .build()
        .await?;
    let scan = harness.scan().await?;
    let context = harness.state().task_ctx();

    let mut active = scan.execute(0, context.clone())?;
    let first = active.try_next().await?.expect("one row in the first file");
    assert_eq!(first.num_rows(), 1);
    drop(active);

    let mut rows = scan.execute(1, context)?.try_collect::<Vec<_>>().await?;
    rows.push(first);
    datafusion::assert_batches_sorted_eq!(
        ["+----+", "| id |", "+----+", "| 1  |", "| 2  |", "+----+"],
        &rows
    );
    Ok(())
}

#[tokio::test]
async fn lazy_scan_shares_a_position_delete_file_across_data_files() -> Result<()> {
    let mut table = empty_items().await?;
    let files = table
        .append()
        .file(id_batch(vec![1, 2])?)
        .file(id_batch(vec![3, 4])?)
        .commit()
        .await?;
    let deletes = table
        .delete_positions(&[(&files[0], 0), (&files[1], 0)])
        .await?;
    let storage = ReadCountingStorageFactory::default();
    let (_, rows) = IcebergTestHarness::builder()
        .with_iceberg_table_factory(
            IcebergTableProviderFactory::new_with_storage_factory(Arc::new(
                storage.clone(),
            )),
        )
        .with_table(table)
        .with_config_option("datafusion.execution.target_partitions", 2)
        .query("SELECT id FROM items ORDER BY id")
        .await?;
    insta::assert_snapshot!(rows, @"
    +----+
    | id |
    +----+
    | 2  |
    | 4  |
    +----+
    ");
    assert_eq!(storage.file_opens(deletes.file_path()), 1);
    Ok(())
}

#[tokio::test]
async fn lazy_scan_without_a_snapshot_is_empty() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(empty_items().await?)
        .with_config_option("datafusion.execution.target_partitions", 4)
        .build()
        .await?;
    let plan = harness.scan().await?;
    assert_eq!(plan.output_partitioning().partition_count(), 4);
    assert!(collect(plan, harness.state().task_ctx()).await?.is_empty());
    Ok(())
}

async fn empty_items() -> Result<FixtureTable> {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .unwrap();
    FixtureCatalog::try_new()
        .await?
        .table(TableIdent::from_strs(["default", "items"]).unwrap(), schema)
        .create()
        .await
}

fn id_batch(values: Vec<i64>) -> Result<RecordBatch> {
    Ok(RecordBatch::try_from_iter([(
        "id",
        Arc::new(Int64Array::from(values)) as ArrayRef,
    )])?)
}

// Only the public source-composition tests need to inspect the scan below SQL.
fn iceberg_source(plan: &Arc<dyn ExecutionPlan>) -> Option<&IcebergDataSource> {
    plan.downcast_ref::<DataSourceExec>()
        .and_then(|exec| exec.data_source().downcast_ref::<IcebergDataSource>())
        .or_else(|| plan.children().into_iter().find_map(iceberg_source))
}

fn eager_files(source: &IcebergDataSource) -> &EagerFileScanTaskProvider {
    (source.file_scan_task_provider() as &dyn Any)
        .downcast_ref::<EagerFileScanTaskProvider>()
        .unwrap()
}
