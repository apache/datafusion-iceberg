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
    async fn reports_the_snapshot_row_count_to_datafusion() -> Result<()> {
        let (_, explain) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .with_config_option("datafusion.explain.show_statistics", true)
            .query("EXPLAIN SELECT vendor_id FROM taxi")
            .await?;

        insta::assert_snapshot!(explain, @r"
        +---------------+---------------------------------------------------------------------------------------------------------------------------------+
        | plan_type     | plan                                                                                                                            |
        +---------------+---------------------------------------------------------------------------------------------------------------------------------+
        | logical_plan  | TableScan: taxi projection=[vendor_id]                                                                                          |
        | physical_plan | CooperativeExec, statistics=[Rows=Inexact(175000), Bytes=Inexact(700000), [(Col[0]:)]]                                          |
        |               |   DataSourceExec: format=iceberg, projection=[vendor_id], statistics=[Rows=Inexact(175000), Bytes=Inexact(700000), [(Col[0]:)]] |
        |               |                                                                                                                                 |
        +---------------+---------------------------------------------------------------------------------------------------------------------------------+
        ");

        Ok(())
    }
}
