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
    async fn registers_the_fixture_with_the_iceberg_schema() -> Result<()> {
        let (_, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query("DESCRIBE taxi")
            .await?;

        insta::assert_snapshot!(batches, @r"
    +---------------------+---------------+-------------+
    | column_name         | data_type     | is_nullable |
    +---------------------+---------------+-------------+
    | vendor_id           | Int32         | YES         |
    | pickup_at           | Timestamp(µs) | YES         |
    | dropoff_at          | Timestamp(µs) | YES         |
    | passenger_count     | Int64         | YES         |
    | trip_distance       | Float64       | YES         |
    | pickup_location_id  | Int32         | YES         |
    | dropoff_location_id | Int32         | YES         |
    | payment_type        | Int64         | YES         |
    | fare_amount         | Float64       | YES         |
    | tip_amount          | Float64       | YES         |
    | tolls_amount        | Float64       | YES         |
    | total_amount        | Float64       | YES         |
    | pickup_date         | Date32        | YES         |
    +---------------------+---------------+-------------+
    ");

        Ok(())
    }

    #[tokio::test]
    async fn rejects_schema_definitions_for_existing_iceberg_tables() -> Result<()> {
        let harness = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .build()
            .await?;
        let error = harness
            .query(&format!(
                "CREATE EXTERNAL TABLE invalid (id INT) STORED AS ICEBERG \
                 LOCATION '{}'",
                harness.table().metadata_location()?
            ))
            .await
            .unwrap_err();

        insta::assert_snapshot!(error.to_string(), @"This feature is not implemented: Currently we only support reading existing icebergs tables in external table command. To create new table, please use catalog provider.");

        Ok(())
    }
}
