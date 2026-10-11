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
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::dataframe::DataFrame;
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::displayable;
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::StreamExt;
use futures::stream::BoxStream;
use iceberg::io::{
    FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorage, OutputFile, Storage,
    StorageConfig, StorageFactory,
};
use iceberg::{Error, ErrorKind, Result as IcebergResult};
use serde::{Deserialize, Serialize};

use crate::IcebergTableProviderFactory;

pub const FIXTURE_URI: &str = "s3://iceberg-test/warehouse/taxi";
const WAREHOUSE_URI: &str = "s3://iceberg-test/warehouse/";

pub struct IcebergTestHarness {
    ctx: SessionContext,
    #[cfg(test)]
    parquet_read_bytes: Arc<AtomicU64>,
}

impl IcebergTestHarness {
    pub async fn new() -> Result<Self> {
        let storage_factory = FixtureStorageFactory::default();
        #[cfg(test)]
        let parquet_read_bytes = storage_factory.parquet_read_bytes.clone();
        let state = SessionStateBuilder::new()
            .with_default_features()
            // Plan snapshots must not vary with the runner's CPU count.
            .with_config(SessionConfig::new().with_target_partitions(16))
            .with_table_factory(
                "ICEBERG".to_string(),
                Arc::new(IcebergTableProviderFactory::new_with_storage_factory(
                    Arc::new(storage_factory),
                )),
            )
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.sql(&format!(
            "CREATE EXTERNAL TABLE taxi STORED AS ICEBERG \
         LOCATION '{FIXTURE_URI}/metadata/v1.metadata.json'"
        ))
        .await?
        .collect()
        .await?;
        Ok(Self {
            ctx,
            #[cfg(test)]
            parquet_read_bytes,
        })
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
    #[cfg(test)]
    #[serde(skip)]
    parquet_read_bytes: Arc<AtomicU64>,
}

impl Default for FixtureStorageFactory {
    fn default() -> Self {
        Self {
            root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/iceberg"),
            #[cfg(test)]
            parquet_read_bytes: Arc::default(),
        }
    }
}

#[typetag::serde]
impl StorageFactory for FixtureStorageFactory {
    fn build(&self, _config: &StorageConfig) -> IcebergResult<Arc<dyn Storage>> {
        Ok(Arc::new(FixtureStorage {
            root: self.root.clone(),
            #[cfg(test)]
            parquet_read_bytes: self.parquet_read_bytes.clone(),
        }))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FixtureStorage {
    root: PathBuf,
    #[cfg(test)]
    #[serde(skip)]
    parquet_read_bytes: Arc<AtomicU64>,
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
        let reader = self.local().reader(&self.local_path(path)?).await?;
        #[cfg(test)]
        if path.ends_with(".parquet") {
            return Ok(Box::new(CountingFileRead {
                reader,
                bytes: self.parquet_read_bytes.clone(),
            }));
        }
        Ok(reader)
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

#[cfg(test)]
struct CountingFileRead {
    reader: Box<dyn FileRead>,
    bytes: Arc<AtomicU64>,
}

#[cfg(test)]
#[async_trait]
impl FileRead for CountingFileRead {
    async fn read(&self, range: Range<u64>) -> IcebergResult<Bytes> {
        let bytes = self.reader.read(range).await?;
        self.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }
}

#[cfg(test)]
impl IcebergTestHarness {
    pub(crate) fn context(&self) -> &SessionContext {
        &self.ctx
    }

    pub(crate) fn take_parquet_read_bytes(&self) -> u64 {
        self.parquet_read_bytes.swap(0, Ordering::Relaxed)
    }

    pub(crate) async fn with_parquet(
        schema: iceberg::spec::Schema,
        batch: &datafusion::arrow::array::RecordBatch,
        properties: parquet::file::properties::WriterProperties,
        catalog_backed: bool,
    ) -> Result<(Self, tempfile::TempDir)> {
        use std::collections::HashMap;

        use datafusion::datasource::TableProvider;
        use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
        use iceberg::spec::{DataContentType, DataFileBuilder, DataFileFormat, Struct};
        use iceberg::transaction::{ApplyTransactionAction, Transaction};
        use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
        use parquet::arrow::ArrowWriter;

        use crate::{
            IcebergCatalogTableProvider, IcebergStaticTableProvider, to_datafusion_error,
        };

        let directory = tempfile::TempDir::new()?;
        let factory = FixtureStorageFactory {
            root: directory.path().to_path_buf(),
            ..Default::default()
        };
        let parquet_read_bytes = factory.parquet_read_bytes.clone();
        let catalog = Arc::new(
            MemoryCatalogBuilder::default()
                .with_storage_factory(Arc::new(factory))
                .load(
                    "fixture",
                    HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        WAREHOUSE_URI.trim_end_matches('/').to_string(),
                    )]),
                )
                .await
                .map_err(to_datafusion_error)?,
        );
        let namespace = NamespaceIdent::new("default".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .map_err(to_datafusion_error)?;
        let table = catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("events".to_string())
                    .schema(schema)
                    .build(),
            )
            .await
            .map_err(to_datafusion_error)?;
        let path = format!("{}/data.parquet", table.metadata().location());
        let mut data = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut data, batch.schema(), Some(properties))?;
        writer.write(batch)?;
        let metadata = writer.close()?;
        assert_eq!(metadata.num_row_groups(), 1);
        let file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .partition_spec_id(0)
            .file_path(path.clone())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(data.len() as u64)
            .record_count(batch.num_rows() as u64)
            .partition(Struct::empty())
            .build()
            .unwrap();
        table
            .file_io()
            .new_output(&path)
            .map_err(to_datafusion_error)?
            .write(data.into())
            .await
            .map_err(to_datafusion_error)?;
        let transaction = Transaction::new(&table);
        let table = transaction
            .fast_append()
            .add_data_files([file])
            .apply(transaction)
            .map_err(to_datafusion_error)?
            .commit(catalog.as_ref())
            .await
            .map_err(to_datafusion_error)?;
        let provider: Arc<dyn TableProvider> = if catalog_backed {
            Arc::new(
                IcebergCatalogTableProvider::try_new(catalog, namespace, "events")
                    .await?,
            )
        } else {
            Arc::new(IcebergStaticTableProvider::try_new_from_table(table).await?)
        };
        let ctx = SessionContext::new();
        ctx.register_table("events", provider)?;
        Ok((
            Self {
                ctx,
                parquet_read_bytes,
            },
            directory,
        ))
    }
}
