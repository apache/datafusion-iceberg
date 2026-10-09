#[cfg(test)]
mod tests {
    use datafusion::error::Result;
    use datafusion_iceberg::test_utils::{FIXTURE_URI, IcebergTestHarness};

    #[tokio::test]
    async fn registers_the_fixture_with_the_iceberg_schema() -> Result<()> {
        let harness = IcebergTestHarness::new().await?;
        let (_, batches) = harness.query("DESCRIBE taxi").await?;

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
        let harness = IcebergTestHarness::new().await?;
        let error = harness
            .query(&format!(
                "CREATE EXTERNAL TABLE invalid (id INT) STORED AS ICEBERG \
                 LOCATION '{FIXTURE_URI}/metadata/v1.metadata.json'"
            ))
            .await
            .unwrap_err();

        insta::assert_snapshot!(error.to_string(), @"This feature is not implemented: Currently we only support reading existing icebergs tables in external table command. To create new table, please use catalog provider.");

        Ok(())
    }
}
