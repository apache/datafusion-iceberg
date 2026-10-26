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

// Each test binary compiles this module and uses only part of it.
#![allow(dead_code)]

use std::sync::Arc;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::dataframe::DataFrame;
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::displayable;
use datafusion::prelude::{SessionConfig, SessionContext};
use iceberg::TableIdent;
use iceberg_datafusion::{
    IcebergCatalogProvider, IcebergTableProviderFactory, to_datafusion_error,
};
use iceberg_datafusion_test_tools::{FixtureCatalog, FixtureTable, create_taxi_table};

/// A DataFusion session for integration tests, over a freshly generated taxi
/// table.
///
/// Its default catalog, `datafusion`, is an Iceberg catalog holding
/// `public.taxi` from [`create_taxi_table`], so SQL can say `FROM taxi`.
/// `CREATE EXTERNAL TABLE ... STORED AS ICEBERG` is also registered. It owns
/// the fixture catalog, so the table lives as long as the harness.
pub struct IcebergTestHarness {
    ctx: SessionContext,
    taxi: FixtureTable,
    _fixtures: FixtureCatalog,
}

impl IcebergTestHarness {
    /// Generates the taxi table and builds the session over it.
    pub async fn new() -> Result<Self> {
        let state = SessionStateBuilder::new()
            .with_default_features()
            // Plan snapshots must not vary with the runner's CPU count.
            .with_config(SessionConfig::new().with_target_partitions(16))
            // We register a factory in addition to the catalog below
            // so that tests can be written against the factory's behaviours
            .with_table_factory(
                "ICEBERG".to_string(),
                Arc::new(IcebergTableProviderFactory::new()),
            )
            .build();

        let fixtures = FixtureCatalog::new().await.map_err(to_datafusion_error)?;
        let ident =
            TableIdent::from_strs(["public", "taxi"]).map_err(to_datafusion_error)?;
        let taxi = create_taxi_table(&fixtures, ident)
            .await
            .map_err(to_datafusion_error)?;
        let provider = IcebergCatalogProvider::try_new(fixtures.catalog()).await?;

        let ctx = SessionContext::new_with_state(state);
        // Replaces DataFusion's default catalog, so `taxi` resolves unqualified.
        ctx.register_catalog("datafusion", Arc::new(provider));

        Ok(Self {
            ctx,
            taxi,
            _fixtures: fixtures,
        })
    }

    /// Table `taxi`, e.g. to inspect its files.
    pub fn taxi(&self) -> &FixtureTable {
        &self.taxi
    }

    /// The current metadata file of table `taxi`.
    pub fn taxi_metadata_location(&self) -> Result<&str> {
        self.taxi.metadata_location().map_err(to_datafusion_error)
    }

    /// Runs `sql` and returns its indented physical plan and its formatted
    /// rows, ready to snapshot.
    pub async fn query(&self, sql: &str) -> Result<(String, String)> {
        let dataframe: DataFrame = self.ctx.sql(sql).await?;
        let plan = dataframe.create_physical_plan().await?;
        let batches = dataframe.collect().await?;

        Ok((
            displayable(plan.as_ref()).indent(true).to_string(),
            pretty_format_batches(&batches)?.to_string(),
        ))
    }
}
