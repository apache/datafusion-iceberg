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

use datafusion::error::Result;
use iceberg_datafusion::IcebergTableProviderFactory;
use iceberg_datafusion_test_tools::{IcebergTestHarness, create_taxi_table};

#[tokio::test]
async fn lazy_scan_reports_files_and_bytes_even_when_no_rows_match() -> Result<()> {
    let (_, explain) = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(create_taxi_table().await?)
        .with_config_option("datafusion.execution.target_partitions", 1)
        .with_config_option("datafusion.explain.analyze_categories", "rows,bytes")
        .query(
            "EXPLAIN ANALYZE SELECT fare_amount FROM taxi \
             WHERE pickup_date <= DATE '2024-01-10' AND fare_amount = 12.345",
        )
        .await?;

    insta::assert_snapshot!(explain, @"
    +-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | plan_type         | plan                                                                                                                                                                                                                                                                             |
    +-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Plan with Metrics | FilterExec: pickup_date@1 <= 2024-01-10 AND fare_amount@0 = 12.345, projection=[fare_amount@0], metrics=[output_rows=0, output_bytes=0.0 B, output_batches=0, selectivity=N/A (0/0)]                                                                                             |
    |                   |   CooperativeExec, metrics=[]                                                                                                                                                                                                                                                    |
    |                   |     DataSourceExec: format=iceberg, projection=[fare_amount, pickup_date], provider=lazy, predicate=(pickup_date <= 2024-01-10) AND (fare_amount = 12.345), metrics=[output_rows=0, output_bytes=0.0 B, output_batches=0, batches_split=0, files_opened=3, bytes_scanned=2.19 M] |
    |                   |                                                                                                                                                                                                                                                                                  |
    +-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    ");
    Ok(())
}
