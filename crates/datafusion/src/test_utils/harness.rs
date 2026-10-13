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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::dataframe::DataFrame;
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::physical_plan::displayable;
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::StreamExt;
use futures::stream::BoxStream;
use iceberg::io::{
    FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorage, OutputFile, Storage,
    StorageConfig, StorageFactory,
};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::{
    Catalog, CatalogBuilder, Error, ErrorKind, NamespaceIdent, Result as IcebergResult,
    TableIdent,
};
use serde::{Deserialize, Serialize};

use crate::{IcebergCatalogProvider, IcebergTableProviderFactory};

pub const FIXTURE_URI: &str = "s3://iceberg-test/warehouse/taxi";
const WAREHOUSE_URI: &str = "s3://iceberg-test/warehouse/";
const CATALOG_NAME: &str = "datafusion";
const NAMESPACE_NAME: &str = "public";

pub struct IcebergTestHarness {
    ctx: SessionContext,
}

impl IcebergTestHarness {
    pub async fn new() -> Result<Self> {
        Self::with_config(SessionConfig::new().with_target_partitions(16)).await
    }

    pub async fn with_config(config: SessionConfig) -> Result<Self> {
        Self::build(config, None).await
    }

    pub async fn with_config_and_runtime(
        config: SessionConfig,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<Self> {
        Self::build(config, Some(runtime_env)).await
    }

    async fn build(
        config: SessionConfig,
        runtime_env: Option<Arc<RuntimeEnv>>,
    ) -> Result<Self> {
        let builder = SessionStateBuilder::new()
            .with_default_features()
            .with_config(config);
        let builder = match runtime_env {
            Some(runtime_env) => builder.with_runtime_env(runtime_env),
            None => builder,
        };
        // Register a factory in addition to the catalog below so tests can
        // exercise the factory's behavior.
        let state = builder
            .with_table_factory(
                "ICEBERG".to_string(),
                Arc::new(IcebergTableProviderFactory::new_with_storage_factory(
                    Arc::new(FixtureStorageFactory::default()),
                )),
            )
            .build();

        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(FixtureStorageFactory::default()))
            .load(
                CATALOG_NAME,
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    WAREHOUSE_URI.to_string(),
                )]),
            )
            .await
            .map_err(crate::to_datafusion_error)?;

        let namespace = NamespaceIdent::new(String::from(NAMESPACE_NAME));
        catalog
            .create_namespace(&namespace, HashMap::from([]))
            .await
            .map_err(crate::to_datafusion_error)?;

        let table = TableIdent::new(namespace, String::from("taxi"));
        catalog
            .register_table(&table, format!("{FIXTURE_URI}/metadata/v1.metadata.json"))
            .await
            .map_err(crate::to_datafusion_error)?;

        let provider = IcebergCatalogProvider::try_new(Arc::new(catalog)).await?;

        let ctx = SessionContext::new_with_state(state);
        ctx.register_catalog(CATALOG_NAME, Arc::new(provider));

        Ok(Self { ctx })
    }

    pub fn context(&self) -> &SessionContext {
        &self.ctx
    }

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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FixtureStorageFactory {
    root: PathBuf,
}

impl Default for FixtureStorageFactory {
    fn default() -> Self {
        Self {
            root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/iceberg"),
        }
    }
}

#[typetag::serde]
impl StorageFactory for FixtureStorageFactory {
    fn build(&self, _config: &StorageConfig) -> IcebergResult<Arc<dyn Storage>> {
        Ok(Arc::new(FixtureStorage {
            root: self.root.clone(),
        }))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FixtureStorage {
    root: PathBuf,
}

impl FixtureStorage {
    fn local_path(&self, path: &str) -> IcebergResult<String> {
        let relative = path.strip_prefix(WAREHOUSE_URI).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("unsupported fixture URI: {path}"),
            )
        })?;
        Ok(self.root.join(relative).display().to_string())
    }

    fn local(&self) -> LocalFsStorage {
        LocalFsStorage::new()
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for FixtureStorage {
    async fn exists(&self, path: &str) -> IcebergResult<bool> {
        self.local().exists(&self.local_path(path)?).await
    }

    async fn metadata(&self, path: &str) -> IcebergResult<FileMetadata> {
        self.local().metadata(&self.local_path(path)?).await
    }

    async fn read(&self, path: &str) -> IcebergResult<Bytes> {
        self.local().read(&self.local_path(path)?).await
    }

    async fn reader(&self, path: &str) -> IcebergResult<Box<dyn FileRead>> {
        self.local().reader(&self.local_path(path)?).await
    }

    async fn write(&self, path: &str, bytes: Bytes) -> IcebergResult<()> {
        self.local().write(&self.local_path(path)?, bytes).await
    }

    async fn writer(&self, path: &str) -> IcebergResult<Box<dyn FileWrite>> {
        self.local().writer(&self.local_path(path)?).await
    }

    async fn delete(&self, path: &str) -> IcebergResult<()> {
        self.local().delete(&self.local_path(path)?).await
    }

    async fn delete_prefix(&self, path: &str) -> IcebergResult<()> {
        self.local().delete_prefix(&self.local_path(path)?).await
    }

    async fn delete_stream(
        &self,
        paths: BoxStream<'static, String>,
    ) -> IcebergResult<()> {
        let mut paths = paths;
        while let Some(path) = paths.next().await {
            self.delete(&path).await?;
        }
        Ok(())
    }

    fn new_input(&self, path: &str) -> IcebergResult<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> IcebergResult<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}
