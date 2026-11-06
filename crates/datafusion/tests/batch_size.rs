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
use iceberg_datafusion::test_utils::IcebergTestHarness;

#[tokio::test]
async fn scan_honors_session_batch_size() -> Result<()> {
    let harness = IcebergTestHarness::new().await?;

    // Batches above the session's batch_size are split after the scan, so a
    // size below arrow-rs's default of 1024 would pass without the reader
    // seeing it. Use a larger one.
    harness
        .query("SET datafusion.execution.batch_size = 2000")
        .await?;
    let (plan, batches) = harness
        .query_batches("SELECT pickup_date FROM taxi")
        .await?;

    // Nothing above the scan may merge batches, or the assertion below would
    // not be about the scan.
    insta::assert_snapshot!(plan, @r"
    CooperativeExec
      DataSourceExec: format=iceberg, projection=[pickup_date]
    ");
    let max_rows = batches.iter().map(|batch| batch.num_rows()).max();
    assert_eq!(max_rows, Some(2000));
    Ok(())
}
