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

use std::fs::read_to_string;

use datafusion::common::Result;
use datafusion::physical_plan::collect;
use iceberg_datafusion::{IcebergDataFusionConfig, IcebergTableProviderFactory};
use iceberg_datafusion_test_tools::{IcebergTestHarness, create_taxi_table};
use tempfile::NamedTempFile;
use tracing::subscriber::DefaultGuard;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;

#[tokio::test]
async fn eager_discovery_is_a_child_of_scan_planning() -> Result<()> {
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
        .build()
        .await?;
    let (trace, _guard) = capture_planning_spans()?;
    harness.scan().await?;
    insta::assert_snapshot!(read_to_string(trace.path())?, @"
    iceberg.plan_scan{eager=true}:iceberg.plan_files{eager=true file_tasks=7}: close
    iceberg.plan_scan{eager=true}: close
    ");
    Ok(())
}

#[tokio::test]
async fn lazy_discovery_is_deferred_until_execution() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(create_taxi_table().await?)
        .build()
        .await?;
    let (trace, _guard) = capture_planning_spans()?;
    let plan = harness.scan().await?;
    insta::assert_snapshot!(read_to_string(trace.path())?, @"iceberg.plan_scan{eager=false}: close");
    collect(plan, harness.state().task_ctx()).await?;
    insta::assert_snapshot!(read_to_string(trace.path())?, @"
    iceberg.plan_scan{eager=false}: close
    iceberg.plan_files{eager=false file_tasks=7}: close
    ");
    Ok(())
}

fn capture_planning_spans() -> Result<(NamedTempFile, DefaultGuard)> {
    let trace = NamedTempFile::new()?;
    let layer = tracing_subscriber::fmt::layer()
        .without_time()
        .with_target(false)
        .with_level(false)
        .with_ansi(false)
        .with_span_events(FmtSpan::CLOSE)
        .with_writer(trace.reopen()?)
        .with_filter(filter_fn(|m| m.name().starts_with("iceberg.plan_")));
    let guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(layer));
    Ok((trace, guard))
}
