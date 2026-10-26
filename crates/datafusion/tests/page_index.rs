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

mod common;

use common::IcebergTestHarness;
use datafusion::error::Result;

#[tokio::test]
async fn page_index_setting_without_indexes() -> Result<()> {
    let harness = IcebergTestHarness::new().await?;

    // create_taxi_table writes no page indexes, so this covers reading files
    // without them.
    for enabled in [false, true] {
        harness
            .query(&format!(
                "SET datafusion.execution.parquet.enable_page_index = '{enabled}'"
            ))
            .await?;
        let (_, batches) = harness
            .query(
                "SELECT COUNT(*) AS trips FROM taxi \
                 WHERE pickup_date = DATE '2024-01-10' \
                   AND payment_type IN (1, 2) AND trip_distance >= 2.0",
            )
            .await?;
        insta::allow_duplicates! {
            insta::assert_snapshot!(batches, @r"
            +-------+
            | trips |
            +-------+
            | 10659 |
            +-------+
            ");
        }
    }
    Ok(())
}
