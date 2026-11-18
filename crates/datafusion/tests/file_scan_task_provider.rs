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

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::Result;
use datafusion::datasource::source::{DataSource, DataSourceExec};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{ExecutionPlan, collect};
use futures::TryStreamExt;
use iceberg::Runtime;
use iceberg::arrow::ArrowReaderBuilder;
use iceberg_datafusion::IcebergDataFusionConfig;
use iceberg_datafusion::read::{EagerFileScanTaskProvider, IcebergDataSource};
use iceberg_datafusion::test_utils::IcebergTestHarness;

#[tokio::test]
async fn lazy_scan_filters_and_projects_taxi_rows() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_config(|config| config.with_target_partitions(2))
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
                  RepartitionExec: partitioning=RoundRobinBatch(2), input_partitions=1
                    DataSourceExec: format=iceberg, projection=[pickup_date], predicate=(pickup_date = 2024-01-10) OR (pickup_date = 2024-01-11)
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
        .with_config(|config| {
            config
                .with_target_partitions(2)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true)
        })
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
                  DataSourceExec: format=iceberg, projection=[pickup_date], planned_files=2
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
async fn eager_scan_with_no_matching_files_is_executable() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_config(|config| {
            config
                .with_target_partitions(2)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true)
        })
        .build()
        .await?;

    let (plan, rows) = harness
        .query("SELECT COUNT(*) AS trips FROM taxi WHERE pickup_date = DATE '2025-01-01'")
        .await?;

    insta::assert_snapshot!(plan, @"
    ProjectionExec: expr=[count(Int64(1))@0 as trips]
      AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
        CoalescePartitionsExec
          AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
            FilterExec: pickup_date@0 = 2025-01-01, projection=[]
              RepartitionExec: partitioning=RoundRobinBatch(2), input_partitions=1
                DataSourceExec: format=iceberg, projection=[pickup_date], planned_files=0
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
async fn assigned_tasks_can_be_reconstructed_without_a_table() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_config(|config| {
            config
                .with_target_partitions(1)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true)
        })
        .build()
        .await?;
    let plan = harness
        .plan("SELECT pickup_date FROM taxi WHERE pickup_date = DATE '2024-01-10'")
        .await?;
    let source = iceberg_source(&plan).unwrap();
    let schema = source.schema();
    let groups = eager_files(source).groups().to_vec();
    let file_io = source.file_io().clone();
    let reader = ArrowReaderBuilder::new(file_io.clone(), Runtime::current());
    let config = harness.context().copied_config();
    drop(plan);
    drop(harness);

    // A worker decoder receives assigned tasks and local reader resources, not
    // a table or catalog. Construction also works outside a Tokio runtime.
    let source = std::thread::spawn(move || {
        assert!(tokio::runtime::Handle::try_current().is_err());
        IcebergDataSource::builder(&config, schema, file_io, reader)
            .with_file_scan_task_provider(Arc::new(EagerFileScanTaskProvider::try_new(
                groups,
            )?))
            .build()
    })
    .join()
    .unwrap()?;
    let batches = collect(
        DataSourceExec::from_data_source(source),
        Arc::new(TaskContext::default()),
    )
    .await?;

    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        25_000
    );
    Ok(())
}

#[tokio::test]
async fn replacing_the_provider_reads_the_assignment_with_the_existing_limit()
-> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_config(|config| {
            config
                .with_target_partitions(2)
                .with_option_extension(IcebergDataFusionConfig::default())
                .set_bool("iceberg.execution.eager_file_planning", true)
        })
        .build()
        .await?;
    let plan = harness.plan("SELECT pickup_date FROM taxi LIMIT 2").await?;
    let source = iceberg_source(&plan).unwrap();
    let assigned_file = eager_files(source)
        .tasks()
        .find(|task| task.data_file_path().contains("pickup_date=2024-01-12/"))
        .unwrap()
        .clone();

    let assigned = source.with_file_scan_task_provider(Arc::new(
        EagerFileScanTaskProvider::try_new(vec![vec![assigned_file]])?,
    ))?;
    let batches = collect(
        DataSourceExec::from_data_source(assigned),
        harness.context().task_ctx(),
    )
    .await?;

    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @"
    +-------------+
    | pickup_date |
    +-------------+
    | 2024-01-12  |
    | 2024-01-12  |
    +-------------+
    ");
    Ok(())
}

#[tokio::test]
async fn the_scan_keeps_its_construction_time_reader_settings() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_config(|config| config.with_batch_size(2000))
        .build()
        .await?;
    let plan = harness.plan("SELECT pickup_date FROM taxi").await?;
    let source = iceberg_source(&plan).unwrap();

    // Open with default settings, not the settings used to construct the scan.
    let batches = source
        .open(0, Arc::new(TaskContext::default()))?
        .try_collect::<Vec<_>>()
        .await?;

    assert_eq!(batches.iter().map(RecordBatch::num_rows).max(), Some(2000));
    Ok(())
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
