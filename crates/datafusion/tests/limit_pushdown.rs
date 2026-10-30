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
    use iceberg_datafusion::IcebergTableProviderFactory;
    use iceberg_datafusion_test_tools::{IcebergTestHarness, create_taxi_table};

    #[tokio::test]
    async fn applies_sql_limit_to_the_iceberg_scan_output() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query(
                r"SELECT vendor_id, pickup_at, passenger_count, trip_distance, pickup_date
               FROM taxi
               WHERE pickup_date = DATE '2024-01-10'
               ORDER BY pickup_at
               LIMIT 3",
            )
            .await?;

        insta::assert_snapshot!(plan, @r"
        SortPreservingMergeExec: [pickup_at@1 ASC NULLS LAST], fetch=3
          SortExec: TopK(fetch=3), expr=[pickup_at@1 ASC NULLS LAST], preserve_partitioning=[true]
            FilterExec: pickup_date@4 = 2024-01-10
              RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                DataSourceExec: format=iceberg, projection=[vendor_id, pickup_at, passenger_count, trip_distance, pickup_date], predicate=pickup_date = 2024-01-10
        ");
        insta::assert_snapshot!(batches, @r"
        +-----------+---------------------+-----------------+---------------+-------------+
        | vendor_id | pickup_at           | passenger_count | trip_distance | pickup_date |
        +-----------+---------------------+-----------------+---------------+-------------+
        | 1         | 2024-01-10T00:00:04 | 1               | 0.68          | 2024-01-10  |
        | 2         | 2024-01-10T00:00:08 | 1               | 6.18          | 2024-01-10  |
        | 2         | 2024-01-10T00:00:11 |                 | 3.61          | 2024-01-10  |
        +-----------+---------------------+-----------------+---------------+-------------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn keeps_unordered_limits_above_the_iceberg_source() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query("SELECT pickup_date FROM taxi WHERE pickup_date = DATE '2024-01-10' LIMIT 3")
            .await?;

        insta::assert_snapshot!(plan, @r"
        CoalescePartitionsExec: fetch=3
          FilterExec: pickup_date@0 = 2024-01-10, fetch=3
            RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
              DataSourceExec: format=iceberg, projection=[pickup_date], predicate=pickup_date = 2024-01-10
        ");
        insta::assert_snapshot!(batches, @r"
    +-------------+
    | pickup_date |
    +-------------+
    | 2024-01-10  |
    | 2024-01-10  |
    | 2024-01-10  |
    +-------------+
    ");

        Ok(())
    }

    #[tokio::test]
    async fn passes_the_scan_limit_to_the_iceberg_source() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query("SELECT pickup_date FROM taxi LIMIT 3")
            .await?;

        insta::assert_snapshot!(plan, @r"
        CooperativeExec
          DataSourceExec: format=iceberg, projection=[pickup_date], fetch=3
        ");
        insta::assert_snapshot!(batches, @r"
    +-------------+
    | pickup_date |
    +-------------+
    | 2024-01-08  |
    | 2024-01-08  |
    | 2024-01-08  |
    +-------------+
    ");

        Ok(())
    }

    #[tokio::test]
    async fn avoids_scanning_for_limit_zero() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query("SELECT vendor_id FROM taxi WHERE pickup_date = DATE '2024-01-10' LIMIT 0")
            .await?;

        insta::assert_snapshot!(plan, @r"
    EmptyExec
    ");
        insta::assert_snapshot!(batches, @r"
    ++
    ++
    ");

        Ok(())
    }

    #[tokio::test]
    async fn applies_offset_before_limit() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query(
                r"SELECT vendor_id, pickup_at
                   FROM taxi
                   WHERE pickup_date = DATE '2024-01-10'
                   ORDER BY pickup_at, vendor_id, pickup_location_id
                   LIMIT 2 OFFSET 3",
            )
            .await?;

        insta::assert_snapshot!(plan, @r"
        ProjectionExec: expr=[vendor_id@0 as vendor_id, pickup_at@1 as pickup_at]
          GlobalLimitExec: skip=3, fetch=2
            SortPreservingMergeExec: [pickup_at@1 ASC NULLS LAST, vendor_id@0 ASC NULLS LAST, pickup_location_id@2 ASC NULLS LAST], fetch=5
              SortExec: TopK(fetch=5), expr=[pickup_at@1 ASC NULLS LAST, vendor_id@0 ASC NULLS LAST, pickup_location_id@2 ASC NULLS LAST], preserve_partitioning=[true]
                FilterExec: pickup_date@3 = 2024-01-10, projection=[vendor_id@0, pickup_at@1, pickup_location_id@2]
                  RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                    DataSourceExec: format=iceberg, projection=[vendor_id, pickup_at, pickup_location_id, pickup_date], predicate=pickup_date = 2024-01-10
        ");
        insta::assert_snapshot!(batches, @r"
        +-----------+---------------------+
        | vendor_id | pickup_at           |
        +-----------+---------------------+
        | 1         | 2024-01-10T00:00:15 |
        | 2         | 2024-01-10T00:00:18 |
        +-----------+---------------------+
        ");

        Ok(())
    }
}
