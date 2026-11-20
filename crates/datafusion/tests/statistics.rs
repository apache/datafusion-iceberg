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
    use datafusion::error::Result;
    use iceberg_datafusion::{IcebergDataFusionConfig, IcebergTableProviderFactory};
    use iceberg_datafusion_test_tools::{IcebergTestHarness, create_taxi_table};

    #[tokio::test]
    async fn lazy_snapshot_count_belongs_to_the_scan_not_either_reader() -> Result<()> {
        let harness = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .with_config_option("datafusion.execution.target_partitions", 2)
            .build()
            .await?;
        let plan = harness.physical_plan("SELECT vendor_id FROM taxi").await?;
        insta::assert_snapshot!(harness.source_statistics(&plan)?, @"
        all partitions (2): Rows=Inexact(175000), Bytes=Inexact(700000), [(Col[0]:)]
        partition 0: Rows=Absent, Bytes=Absent, [(Col[0]:)]
        partition 1: Rows=Absent, Bytes=Absent, [(Col[0]:)]
        ");
        Ok(())
    }

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
}
