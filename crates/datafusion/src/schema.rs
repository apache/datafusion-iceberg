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

use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use datafusion::catalog::SchemaProvider;
use datafusion::common::{exec_datafusion_err, exec_err, plan_datafusion_err};
use datafusion::datasource::TableProvider;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use futures::future::try_join_all;
use iceberg::arrow::arrow_schema_to_schema_auto_assign_ids;
use iceberg::inspect::MetadataTableType;
use iceberg::spec::FormatVersion;
use iceberg::{Catalog, NamespaceIdent, TableCreation, TableIdent};
use tokio::runtime::{Handle, RuntimeFlavor};

use crate::table::IcebergTableProvider;
use crate::to_datafusion_error;

/// Represents a [`SchemaProvider`] for the Iceberg [`Catalog`], managing
/// access to table providers within a specific namespace.
#[derive(Debug)]
pub(crate) struct IcebergSchemaProvider {
    /// Reference to the Iceberg catalog
    catalog: Arc<dyn Catalog>,
    /// The namespace this schema represents
    namespace: NamespaceIdent,
    /// A concurrent map where keys are table names
    /// and values are dynamic references to objects implementing the
    /// [`TableProvider`] trait.
    /// Wrapped in Arc to allow sharing across async boundaries in register_table.
    tables: Arc<DashMap<String, Arc<IcebergTableProvider>>>,
}

impl IcebergSchemaProvider {
    /// Asynchronously tries to construct a new [`IcebergSchemaProvider`]
    /// using the given client to fetch and initialize table providers for
    /// the provided namespace in the Iceberg [`Catalog`].
    ///
    /// This method retrieves a list of table names
    /// attempts to create a table provider for each table name, and
    /// collects these providers into a `HashMap`.
    pub(crate) async fn try_new(
        client: Arc<dyn Catalog>,
        namespace: NamespaceIdent,
    ) -> Result<Self> {
        // TODO:
        // Tables and providers should be cached based on table_name
        // if we have a cache miss; we update our internal cache & check again
        // As of right now; tables might become stale.
        let table_names: Vec<_> = client
            .list_tables(&namespace)
            .await
            .map_err(to_datafusion_error)?
            .iter()
            .map(|tbl| tbl.name().to_string())
            .collect();

        let providers = try_join_all(
            table_names
                .iter()
                .map(|name| {
                    IcebergTableProvider::try_new(client.clone(), namespace.clone(), name)
                })
                .collect::<Vec<_>>(),
        )
        .await?;

        let tables = Arc::new(DashMap::new());
        for (name, provider) in table_names.into_iter().zip(providers) {
            tables.insert(name, Arc::new(provider));
        }

        Ok(IcebergSchemaProvider {
            catalog: client,
            namespace,
            tables,
        })
    }
}

#[async_trait]
impl SchemaProvider for IcebergSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.tables
            .iter()
            .flat_map(|entry| {
                let table_name = entry.key().clone();
                [table_name.clone()].into_iter().chain(
                    MetadataTableType::all_types().map(move |metadata_table_name| {
                        format!("{}${}", table_name, metadata_table_name.as_str())
                    }),
                )
            })
            .collect()
    }

    fn table_exist(&self, name: &str) -> bool {
        if let Some((table_name, metadata_table_name)) = name.split_once('$') {
            self.tables.contains_key(table_name)
                && MetadataTableType::try_from(metadata_table_name).is_ok()
        } else {
            self.tables.contains_key(name)
        }
    }

    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        if let Some((table_name, metadata_table_name)) = name.split_once('$') {
            let metadata_table_type = MetadataTableType::try_from(metadata_table_name)
                .map_err(|e| plan_datafusion_err!("{e}"))?;
            if let Some(table) = self.tables.get(table_name) {
                let metadata_table = table.metadata_table(metadata_table_type).await?;
                return Ok(Some(Arc::new(metadata_table)));
            } else {
                return Ok(None);
            }
        }

        Ok(self
            .tables
            .get(name)
            .map(|entry| entry.value().clone() as Arc<dyn TableProvider>))
    }

    fn register_table(
        &self,
        name: String,
        table: Arc<dyn TableProvider>,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        // Check if table already exists
        if self.table_exist(name.as_str()) {
            return exec_err!("Table {name} already exists");
        }

        // Convert DataFusion schema to Iceberg schema
        // DataFusion schemas don't have field IDs, so we use the function that assigns them automatically
        let df_schema = table.schema();
        let iceberg_schema = arrow_schema_to_schema_auto_assign_ids(df_schema.as_ref())
            .map_err(to_datafusion_error)?;

        // Use at least V2, and upgrade to V3 if the schema requires it (e.g. timestamp_ns / variant).
        let format_version = iceberg_schema
            .calc_min_compatible_format()
            .max(FormatVersion::V2);

        // Create the table in the Iceberg catalog
        let table_creation = TableCreation::builder()
            .name(name.clone())
            .schema(iceberg_schema)
            .format_version(format_version)
            .build();

        let catalog = self.catalog.clone();
        let namespace = self.namespace.clone();
        let tables = self.tables.clone();
        let name_clone = name.clone();

        block_on_catalog("create-table", async move {
            // Verify the input table is empty - CREATE TABLE only accepts schema definition
            ensure_table_is_empty(&table).await?;

            catalog
                .create_table(&namespace, table_creation)
                .await
                .map_err(to_datafusion_error)?;

            // Create a new table provider using the catalog reference
            let table_provider = IcebergTableProvider::try_new(
                catalog.clone(),
                namespace.clone(),
                name_clone.clone(),
            )
            .await?;

            // Store the new table provider
            tables.insert(name_clone, Arc::new(table_provider));

            Ok(None)
        })
    }

    fn deregister_table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        // Check if table exists
        if !self.table_exist(name) {
            return Ok(None);
        }

        let catalog = self.catalog.clone();
        let namespace = self.namespace.clone();
        let tables = self.tables.clone();
        let table_name = name.to_string();

        block_on_catalog("drop-table", async move {
            let table_ident = TableIdent::new(namespace, table_name.clone());

            // Drop the table from the Iceberg catalog
            catalog
                .drop_table(&table_ident)
                .await
                .map_err(to_datafusion_error)?;

            // Remove from local cache and return the removed provider
            let removed = tables
                .remove(&table_name)
                .map(|(_, table)| table as Arc<dyn TableProvider>);

            Ok(removed)
        })
    }
}

/// Runs a synchronous catalog operation without blocking a runtime that must drive it.
///
/// A multi-thread runtime can continue driving I/O while this worker blocks, so run the
/// future on that runtime using `block_in_place`. A current-thread runtime cannot make
/// progress while its caller blocks and must return an error. Synchronous callers with
/// no current runtime can drive the future on a runtime created for the call.
fn block_on_catalog<T>(
    operation: &'static str,
    future: impl Future<Output = Result<T>>,
) -> Result<T> {
    match Handle::try_current() {
        Ok(handle) => match handle.runtime_flavor() {
            RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| handle.block_on(future))
            }
            RuntimeFlavor::CurrentThread => {
                exec_err!("{operation} requires a multi-thread Tokio runtime")
            }
            _ => exec_err!("{operation} requires a multi-thread Tokio runtime"),
        },
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| {
                exec_datafusion_err!(
                    "Failed to create Tokio runtime for {operation}: {error}"
                )
            })?
            .block_on(future),
    }
}

/// Verifies that a table provider contains no data by scanning with LIMIT 1.
/// Returns an error if the table has any rows.
async fn ensure_table_is_empty(table: &Arc<dyn TableProvider>) -> Result<()> {
    let session_ctx = SessionContext::new();
    let exec_plan = table.scan(&session_ctx.state(), None, &[], Some(1)).await?;

    let task_ctx = Arc::new(TaskContext::default());
    let stream = exec_plan.execute(0, task_ctx)?;

    let batches: Vec<_> = stream.try_collect().await?;
    let has_data = batches.iter().any(|batch| batch.num_rows() > 0);

    if has_data {
        return exec_err!("register_table does not support tables with data.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use datafusion::arrow::array::{Int32Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::datasource::MemTable;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::table::Table;
    use iceberg::{
        Catalog, CatalogBuilder, Namespace, NamespaceIdent, TableCommit, TableCreation,
        TableIdent,
    };
    use tempfile::TempDir;

    use super::*;

    async fn create_test_schema_provider() -> (IcebergSchemaProvider, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let warehouse_path = temp_dir.path().to_str().unwrap().to_string();

        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    warehouse_path.clone(),
                )]),
            )
            .await
            .unwrap();

        let namespace = NamespaceIdent::new("test_ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();

        let provider = IcebergSchemaProvider::try_new(Arc::new(catalog), namespace)
            .await
            .unwrap();

        (provider, temp_dir)
    }

    #[derive(Debug)]
    struct DelayedCatalog(Arc<dyn Catalog>, Handle);

    impl DelayedCatalog {
        async fn delay_on_owner_runtime(&self) {
            let (sender, receiver) = tokio::sync::oneshot::channel();
            self.1.spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                let _ = sender.send(());
            });
            receiver.await.unwrap();
        }
    }

    #[async_trait::async_trait]
    impl Catalog for DelayedCatalog {
        async fn list_namespaces(
            &self,
            parent: Option<&NamespaceIdent>,
        ) -> iceberg::Result<Vec<NamespaceIdent>> {
            self.0.list_namespaces(parent).await
        }

        async fn create_namespace(
            &self,
            namespace: &NamespaceIdent,
            properties: HashMap<String, String>,
        ) -> iceberg::Result<Namespace> {
            self.0.create_namespace(namespace, properties).await
        }

        async fn get_namespace(
            &self,
            namespace: &NamespaceIdent,
        ) -> iceberg::Result<Namespace> {
            self.0.get_namespace(namespace).await
        }

        async fn namespace_exists(
            &self,
            namespace: &NamespaceIdent,
        ) -> iceberg::Result<bool> {
            self.0.namespace_exists(namespace).await
        }

        async fn update_namespace(
            &self,
            namespace: &NamespaceIdent,
            properties: HashMap<String, String>,
        ) -> iceberg::Result<()> {
            self.0.update_namespace(namespace, properties).await
        }

        async fn drop_namespace(
            &self,
            namespace: &NamespaceIdent,
        ) -> iceberg::Result<()> {
            self.0.drop_namespace(namespace).await
        }

        async fn list_tables(
            &self,
            namespace: &NamespaceIdent,
        ) -> iceberg::Result<Vec<TableIdent>> {
            self.0.list_tables(namespace).await
        }

        async fn create_table(
            &self,
            namespace: &NamespaceIdent,
            creation: TableCreation,
        ) -> iceberg::Result<Table> {
            self.delay_on_owner_runtime().await;
            self.0.create_table(namespace, creation).await
        }

        async fn load_table(&self, table: &TableIdent) -> iceberg::Result<Table> {
            self.0.load_table(table).await
        }

        async fn drop_table(&self, table: &TableIdent) -> iceberg::Result<()> {
            self.delay_on_owner_runtime().await;
            self.0.drop_table(table).await
        }

        async fn purge_table(&self, table: &TableIdent) -> iceberg::Result<()> {
            self.0.purge_table(table).await
        }

        async fn table_exists(&self, table: &TableIdent) -> iceberg::Result<bool> {
            self.0.table_exists(table).await
        }

        async fn rename_table(
            &self,
            src: &TableIdent,
            dest: &TableIdent,
        ) -> iceberg::Result<()> {
            self.0.rename_table(src, dest).await
        }

        async fn register_table(
            &self,
            table: &TableIdent,
            metadata_location: String,
        ) -> iceberg::Result<Table> {
            self.0.register_table(table, metadata_location).await
        }

        async fn update_table(&self, commit: TableCommit) -> iceberg::Result<Table> {
            self.0.update_table(commit).await
        }
    }

    async fn create_delayed_test_schema_provider(
        with_existing_table: bool,
    ) -> (IcebergSchemaProvider, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let warehouse_path = temp_dir.path().to_str().unwrap().to_string();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path)]),
            )
            .await
            .unwrap();
        let namespace = NamespaceIdent::new("test_ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        if with_existing_table {
            let arrow_schema =
                ArrowSchema::new(vec![Field::new("id", DataType::Int32, false)]);
            let iceberg_schema =
                arrow_schema_to_schema_auto_assign_ids(&arrow_schema).unwrap();
            catalog
                .create_table(
                    &namespace,
                    TableCreation::builder()
                        .name("existing_table".to_string())
                        .schema(iceberg_schema)
                        .build(),
                )
                .await
                .unwrap();
        }
        let catalog: Arc<dyn Catalog> =
            Arc::new(DelayedCatalog(Arc::new(catalog), Handle::current()));
        let provider = IcebergSchemaProvider::try_new(catalog, namespace)
            .await
            .unwrap();
        (provider, temp_dir)
    }

    #[test]
    fn test_sync_catalog_operations_reject_current_thread_runtime() {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let (schema_provider, _temp_dir) =
                    create_delayed_test_schema_provider(true).await;
                let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
                    "id",
                    DataType::Int32,
                    false,
                )]));
                let empty_batch = RecordBatch::new_empty(arrow_schema.clone());
                let mem_table =
                    MemTable::try_new(arrow_schema, vec![vec![empty_batch]]).unwrap();

                let register_error = schema_provider
                    .register_table("async_table".to_string(), Arc::new(mem_table))
                    .unwrap_err()
                    .to_string();
                let deregister_error = schema_provider
                    .deregister_table("existing_table")
                    .unwrap_err()
                    .to_string();
                let _ = sender.send((register_error, deregister_error));
            });
        });

        let (register_error, deregister_error) = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("catalog calls must return instead of deadlocking a current-thread runtime");
        assert!(register_error.contains("requires a multi-thread"));
        assert!(deregister_error.contains("requires a multi-thread"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_sync_catalog_operations_on_multi_thread_runtime() {
        let (schema_provider, _temp_dir) =
            create_delayed_test_schema_provider(false).await;
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let empty_batch = RecordBatch::new_empty(arrow_schema.clone());
        let mem_table = MemTable::try_new(arrow_schema, vec![vec![empty_batch]]).unwrap();

        schema_provider
            .register_table("async_table".to_string(), Arc::new(mem_table))
            .unwrap();
        assert!(schema_provider.table_exist("async_table"));
        assert!(
            schema_provider
                .deregister_table("async_table")
                .unwrap()
                .is_some()
        );
        assert!(!schema_provider.table_exist("async_table"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_table_registration_without_caller_runtime() {
        let (schema_provider, _temp_dir) =
            create_delayed_test_schema_provider(false).await;

        std::thread::spawn(move || {
            let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
                "id",
                DataType::Int32,
                false,
            )]));
            let empty_batch = RecordBatch::new_empty(arrow_schema.clone());
            let mem_table =
                MemTable::try_new(arrow_schema, vec![vec![empty_batch]]).unwrap();

            schema_provider
                .register_table("thread_table".to_string(), Arc::new(mem_table))
                .unwrap();
            schema_provider.deregister_table("thread_table").unwrap();
        })
        .join()
        .expect("synchronous schema methods should work without a caller runtime");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_register_table_with_data_fails() {
        let (schema_provider, _temp_dir) = create_test_schema_provider().await;

        // Create a MemTable with data
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
            ],
        )
        .unwrap();

        let mem_table = MemTable::try_new(arrow_schema, vec![vec![batch]]).unwrap();

        // Attempt to register the table with data - should fail
        let result =
            schema_provider.register_table("test_table".to_string(), Arc::new(mem_table));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("register_table does not support tables with data."),
            "Expected error about tables with data, got: {err}",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_register_empty_table_succeeds() {
        let (schema_provider, _temp_dir) = create_test_schema_provider().await;

        // Create an empty MemTable (schema only, no data rows)
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        // Create an empty batch (0 rows) - MemTable requires at least one partition
        let empty_batch = RecordBatch::new_empty(arrow_schema.clone());
        let mem_table = MemTable::try_new(arrow_schema, vec![vec![empty_batch]]).unwrap();

        // Attempt to register the empty table - should succeed
        let result = schema_provider
            .register_table("empty_table".to_string(), Arc::new(mem_table));

        assert!(result.is_ok(), "Expected success, got: {result:?}");

        // Verify the table was registered
        assert!(schema_provider.table_exist("empty_table"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_register_duplicate_table_fails() {
        let (schema_provider, _temp_dir) = create_test_schema_provider().await;

        // Create empty MemTables
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let empty_batch1 = RecordBatch::new_empty(arrow_schema.clone());
        let empty_batch2 = RecordBatch::new_empty(arrow_schema.clone());
        let mem_table1 =
            MemTable::try_new(arrow_schema.clone(), vec![vec![empty_batch1]]).unwrap();
        let mem_table2 =
            MemTable::try_new(arrow_schema, vec![vec![empty_batch2]]).unwrap();

        // Register first table - should succeed
        let result1 =
            schema_provider.register_table("dup_table".to_string(), Arc::new(mem_table1));
        assert!(result1.is_ok());

        // Register second table with same name - should fail
        let result2 =
            schema_provider.register_table("dup_table".to_string(), Arc::new(mem_table2));
        assert!(result2.is_err());
        let err = result2.unwrap_err();
        assert!(
            err.to_string().contains("already exists"),
            "Expected error about table already existing, got: {err}",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_deregister_table_succeeds() {
        let (schema_provider, _temp_dir) = create_test_schema_provider().await;

        // Create and register an empty table
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let empty_batch = RecordBatch::new_empty(arrow_schema.clone());
        let mem_table = MemTable::try_new(arrow_schema, vec![vec![empty_batch]]).unwrap();

        // Register the table
        let result =
            schema_provider.register_table("drop_me".to_string(), Arc::new(mem_table));
        assert!(result.is_ok());
        assert!(schema_provider.table_exist("drop_me"));

        // Deregister the table
        let result = schema_provider.deregister_table("drop_me");
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());

        // Verify the table no longer exists
        assert!(!schema_provider.table_exist("drop_me"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_deregister_nonexistent_table_returns_none() {
        let (schema_provider, _temp_dir) = create_test_schema_provider().await;

        // Attempt to deregister a table that doesn't exist
        let result = schema_provider.deregister_table("nonexistent");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }
}
