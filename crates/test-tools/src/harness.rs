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

use std::collections::BTreeMap;
use std::fmt::Display;
use std::str::FromStr;
use std::sync::Arc;

use datafusion::arrow::datatypes::Schema as ArrowSchema;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::{CatalogProvider, TableProviderFactory};
use datafusion::common::exec_err;
use datafusion::error::Result;
use datafusion::execution::{SessionState, SessionStateBuilder};
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::{SessionConfig, SessionContext};
use iceberg::MetadataLocation;
use iceberg::spec::{MAIN_BRANCH, TableMetadata, TableMetadataBuilder};

use crate::FixtureTable;
use crate::error::df_err;

/// A DataFusion session for integration tests, over the table given to its builder, if
/// any.
///
/// The table is registered under its name, e.g. `taxi` for `default.taxi`, with
/// `CREATE EXTERNAL TABLE ... STORED AS ICEBERG`, so the session needs an `ICEBERG`
/// table factory, given with [`IcebergTestHarnessBuilder::with_iceberg_table_factory`]. The harness owns the table, so its files live as long as
/// the harness.
pub struct IcebergTestHarness {
    ctx: SessionContext,
    table: Option<FixtureTable>,
}

impl IcebergTestHarness {
    pub fn builder() -> IcebergTestHarnessBuilder {
        IcebergTestHarnessBuilder::default()
    }

    /// The registered table, e.g. to inspect its files.
    pub fn table(&self) -> &FixtureTable {
        self.table.as_ref().expect("the harness has a table")
    }

    /// The session's state, e.g. to read its configuration.
    pub fn state(&self) -> SessionState {
        self.ctx.state()
    }

    /// Runs `sql` and returns its indented physical plan and its formatted rows, ready to
    /// snapshot. The plan shown is the plan that runs.
    pub async fn query(&self, sql: &str) -> Result<(String, String)> {
        let plan = self.physical_plan(sql).await?;
        let display = displayable(plan.as_ref()).indent(true).to_string();
        let batches = collect(plan, self.ctx.task_ctx()).await?;

        Ok((display, pretty_format_batches(&batches)?.to_string()))
    }

    pub async fn physical_plan(&self, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
        self.ctx.sql(sql).await?.create_physical_plan().await
    }

    /// Returns the registered table's scan without SQL optimization, including for an
    /// empty table.
    pub async fn scan(&self) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(table) = &self.table else {
            return exec_err!("the harness has no table to scan");
        };
        self.ctx
            .table_provider(table.ident().name())
            .await?
            .scan(&self.ctx.state(), None, &[], None)
            .await
    }
}

/// Derives the metadata to register a table from, given its own.
type MetadataTransform = Box<dyn FnOnce(&TableMetadata) -> TableMetadata>;

/// Extends the session, e.g. with a table factory.
type SessionExtension = Box<dyn FnOnce(SessionStateBuilder) -> SessionStateBuilder>;

pub struct IcebergTestHarnessBuilder {
    session_builder: SessionStateBuilder,
    session_extensions: Vec<SessionExtension>,
    config_options: BTreeMap<String, String>,
    catalogs: Vec<(String, Arc<dyn CatalogProvider>)>,
    table_options: BTreeMap<String, String>,
    table: Option<FixtureTable>,
    metadata: Option<MetadataTransform>,
}

impl Default for IcebergTestHarnessBuilder {
    fn default() -> Self {
        Self {
            session_builder: SessionStateBuilder::new()
                .with_default_features()
                // Plan snapshots must not vary with the runner's CPU count.
                .with_config(SessionConfig::new().with_target_partitions(16)),
            session_extensions: Vec::new(),
            config_options: BTreeMap::new(),
            catalogs: Vec::new(),
            table_options: BTreeMap::new(),
            table: None,
            metadata: None,
        }
    }
}

impl IcebergTestHarnessBuilder {
    /// Customizes the session builder, retaining defaults unless explicitly replaced.
    /// Session extensions are installed during [`Self::build`].
    pub fn configure_session(
        mut self,
        configure: impl FnOnce(SessionStateBuilder) -> Result<SessionStateBuilder>,
    ) -> Result<Self> {
        self.session_builder = configure(self.session_builder)?;
        Ok(self)
    }

    /// Extends the session with `extension`, e.g. to register a function or a config
    /// extension. Extensions run in the order they were added.
    pub fn with_session_extension(
        mut self,
        extension: impl FnOnce(SessionStateBuilder) -> SessionStateBuilder + 'static,
    ) -> Self {
        self.session_extensions.push(Box::new(extension));
        self
    }

    /// Registers `factory` as the `ICEBERG` table factory, which [`Self::with_table`]
    /// registers the table with, e.g. `IcebergTableProviderFactory::new()`.
    pub fn with_iceberg_table_factory(
        self,
        factory: impl TableProviderFactory + 'static,
    ) -> Self {
        self.with_session_extension(move |state| {
            state.with_table_factory("ICEBERG".to_string(), Arc::new(factory))
        })
    }

    /// Sets session option `key`, e.g. `datafusion.execution.parquet.enable_page_index`,
    /// to `value`, e.g. `true`. Options are set after the session extensions run, so they
    /// can belong to a config extension one of them registers. Later values replace
    /// earlier ones for the same key.
    pub fn with_config_option(
        mut self,
        key: impl Into<String>,
        value: impl Display,
    ) -> Self {
        self.config_options.insert(key.into(), value.to_string());
        self
    }

    /// Registers `catalog` under `name`, e.g. an Iceberg catalog provider over
    /// [`FixtureTable::catalog`], for tests that need the catalog rather than an external
    /// table. Catalogs are registered before the table, so naming one `datafusion`
    /// replaces the catalog the table would be registered in.
    pub fn with_catalog(
        mut self,
        name: impl Into<String>,
        catalog: Arc<dyn CatalogProvider>,
    ) -> Self {
        self.catalogs.push((name.into(), catalog));
        self
    }

    /// Registers `table` under its name.
    pub fn with_table(mut self, table: FixtureTable) -> Self {
        self.table = Some(table);
        self
    }

    /// Sets an option of the table's `CREATE EXTERNAL TABLE`; later values replace earlier
    /// ones for the same key.
    pub fn with_table_option(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.table_options.insert(key.into(), value.into());
        self
    }

    /// Registers the table from the metadata `metadata` returns for its own. The
    /// metadata is written next to the table's, so it can refer to the table's manifests
    /// and data files.
    pub fn with_metadata(
        mut self,
        metadata: impl FnOnce(&TableMetadata) -> TableMetadata + 'static,
    ) -> Self {
        self.metadata = Some(Box::new(metadata));
        self
    }

    /// Builds the harness and runs `sql` on it, returning its plan and formatted rows.
    pub async fn query(self, sql: &str) -> Result<(String, String)> {
        let harness = self.build().await?;
        harness.query(sql).await
    }

    /// Registers the catalogs, then the table, if any, from customized metadata if
    /// [`Self::with_metadata`] was given.
    pub async fn build(mut self) -> Result<IcebergTestHarness> {
        let table = match (self.table.take(), self.metadata.take()) {
            (None, None) => None,
            (None, Some(_)) => {
                return exec_err!("table metadata can only be customized for a table");
            }
            (Some(table), None) => {
                let location = table.metadata_location()?.to_string();
                Some((table, location))
            }
            (Some(table), Some(metadata)) => {
                let metadata = metadata(table.table().metadata());
                let location = MetadataLocation::from_str(table.metadata_location()?)
                    .map_err(df_err)?
                    .with_next_version();
                metadata
                    .write_to(table.table().file_io(), &location)
                    .await
                    .map_err(df_err)?;
                Some((table, location.to_string()))
            }
        };

        let mut state = self.session_builder;
        for extension in self.session_extensions {
            state = extension(state);
        }
        let options = state.config().get_or_insert_default().options_mut();
        for (key, value) in &self.config_options {
            options.set(key, value)?;
        }
        let ctx = SessionContext::new_with_state(state.build());
        for (name, catalog) in self.catalogs {
            ctx.register_catalog(name, catalog);
        }
        let Some((table, metadata_location)) = table else {
            return Ok(IcebergTestHarness { ctx, table: None });
        };
        let mut statement = format!(
            "CREATE EXTERNAL TABLE {} STORED AS ICEBERG LOCATION '{metadata_location}'",
            table.ident().name()
        );
        if !self.table_options.is_empty() {
            let options = self
                .table_options
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "'{}' '{}'",
                        key.replace('\'', "''"),
                        value.replace('\'', "''")
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            statement.push_str(&format!(" OPTIONS ({options})"));
        }
        ctx.sql(&statement).await?.collect().await?;
        Ok(IcebergTestHarness {
            ctx,
            table: Some(table),
        })
    }
}

/// Renders `schema` one field per line, with types, nullability and metadata, ready to
/// snapshot.
pub fn display_schema(schema: &ArrowSchema) -> String {
    schema
        .fields()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Returns a data file's path relative to its table's `data/` directory, which, unlike
/// the temp dir the table lives in, is the same on every run.
pub fn data_file_name(path: &str) -> &str {
    path.rsplit_once("/data/").map_or(path, |(_, name)| name)
}

/// Starts from `metadata` with its schema, partition spec and properties, but no
/// snapshots.
pub fn without_snapshots(metadata: &TableMetadata) -> TableMetadataBuilder {
    let snapshot_ids: Vec<_> = metadata.snapshots().map(|s| s.snapshot_id()).collect();
    metadata
        .clone()
        .into_builder(None)
        .remove_ref(MAIN_BRANCH)
        .remove_snapshots(&snapshot_ids)
}
