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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::error::Result;
    use iceberg_datafusion::{IcebergDataFusionConfig, IcebergTableProviderFactory};
    use iceberg_datafusion_test_tools::{
        IcebergTestHarness, ReadCountingStorageFactory, create_taxi_table,
    };

    #[tokio::test]
    async fn reports_the_snapshot_row_count_to_datafusion() -> Result<()> {
        let (_, explain) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .with_config_option("datafusion.explain.show_statistics", true)
            .query("EXPLAIN SELECT vendor_id FROM taxi")
            .await?;

        insta::assert_snapshot!(explain, @r"
        +---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | plan_type     | plan                                                                                                                                           |
        +---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | logical_plan  | TableScan: taxi projection=[vendor_id]                                                                                                         |
        | physical_plan | CooperativeExec, statistics=[Rows=Inexact(175000), Bytes=Inexact(700000), [(Col[0]:)]]                                                         |
        |               |   DataSourceExec: format=iceberg, projection=[vendor_id], provider=lazy, statistics=[Rows=Inexact(175000), Bytes=Inexact(700000), [(Col[0]:)]] |
        |               |                                                                                                                                                |
        +---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn eager_statistics_count_only_the_selected_files() -> Result<()> {
        // Taxi has seven daily files of 25,000 rows; the predicate selects two.
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
            .with_config_option("iceberg.execution.eager_file_planning", true)
            .with_config_option("datafusion.execution.target_partitions", 2)
            .build()
            .await?;

        let plan = harness
            .physical_plan(
                "SELECT vendor_id FROM taxi \
                 WHERE pickup_date IN (DATE '2024-01-10', DATE '2024-01-11')",
            )
            .await?;

        insta::assert_snapshot!(harness.source_statistics(&plan)?, @"
        all partitions (2): Rows=Inexact(50000), Bytes=Inexact(400000), [(Col[0]:),(Col[1]:)]
        partition 0: Rows=Inexact(25000), Bytes=Inexact(200000), [(Col[0]:),(Col[1]:)]
        partition 1: Rows=Inexact(25000), Bytes=Inexact(200000), [(Col[0]:),(Col[1]:)]
        ");
        Ok(())
    }

    async fn eager_taxi_harness(
        storage: &ReadCountingStorageFactory,
    ) -> Result<IcebergTestHarness> {
        IcebergTestHarness::builder()
            .with_iceberg_table_factory(
                IcebergTableProviderFactory::new_with_storage_factory(Arc::new(
                    storage.clone(),
                )),
            )
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
            .with_config_option("iceberg.execution.eager_file_planning", true)
            .with_config_option("datafusion.execution.target_partitions", 2)
            .build()
            .await
    }

    #[tokio::test]
    async fn eager_count_star_is_answered_without_reading_data_files() -> Result<()> {
        let storage = ReadCountingStorageFactory::default();
        let harness = eager_taxi_harness(&storage).await?;

        // Taxi has seven daily files of 25,000 rows and no deletes.
        let (plan, _) = harness.query("SELECT count(*) FROM taxi").await?;
        insta::assert_snapshot!(plan, @"
        ProjectionExec: expr=[175000 as count(*)]
          PlaceholderRowExec
        ");
        assert_eq!(storage.take_reads(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn eager_count_star_with_a_predicate_still_reads_data_files() -> Result<()> {
        let storage = ReadCountingStorageFactory::default();
        let harness = eager_taxi_harness(&storage).await?;

        // A pushed-down predicate may drop rows, so the row count stays inexact and
        // DataFusion still aggregates what the scan returns.
        let sql = "SELECT count(*) FROM taxi WHERE pickup_date = DATE '2024-01-10'";
        let physical_plan = harness.physical_plan(sql).await?;
        insta::assert_snapshot!(harness.source_statistics(&physical_plan)?, @"
        all partitions (1): Rows=Inexact(25000), Bytes=Inexact(100000), [(Col[0]:)]
        partition 0: Rows=Inexact(25000), Bytes=Inexact(100000), [(Col[0]:)]
        ");

        let (plan, rows) = harness.query(sql).await?;
        insta::assert_snapshot!(plan, @"
        ProjectionExec: expr=[count(Int64(1))@0 as count(*)]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                FilterExec: pickup_date@0 = 2024-01-10, projection=[]
                  RepartitionExec: partitioning=RoundRobinBatch(2), input_partitions=1
                    DataSourceExec: format=iceberg, projection=[pickup_date], provider=eager, planned_files=1, predicates=[pickup_date = 2024-01-10]
        ");
        insta::assert_snapshot!(rows, @"
        +----------+
        | count(*) |
        +----------+
        | 25000    |
        +----------+
        ");
        assert_ne!(storage.take_reads(), 0);
        Ok(())
    }
}
