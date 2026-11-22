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

use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::catalog::Session;
use datafusion::common::utils::project_schema;
use datafusion::common::{config_datafusion_err, not_impl_err};
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::inspect::MetadataTableType;
use iceberg::{Catalog, NamespaceIdent, TableIdent};

use crate::error::to_datafusion_error;
use crate::metadata::table::IcebergMetadataTableProvider;
use crate::read::{FileScanTaskProviderBuilder, IcebergDataSource};
use crate::write::commit::IcebergCommitExec;
use crate::write::exec::IcebergWriteExec;
use crate::write::project::project_with_partition;
use crate::write::repartition::repartition;

/// Catalog-backed table provider with automatic metadata refresh.
///
/// This provider loads fresh table metadata from the catalog on every scan and write
/// operation, ensuring you always see the latest table state. Use this when you need
/// write operations or want to see the most up-to-date data.
///
/// For read-only access to a specific snapshot without catalog overhead, use
/// [`IcebergStaticTableProvider`](super::IcebergStaticTableProvider) instead.
#[derive(Debug, Clone)]
pub struct IcebergCatalogTableProvider {
    /// The catalog that manages this table
    catalog: Arc<dyn Catalog>,
    /// The table identifier (namespace + name)
    table_ident: TableIdent,
    /// A reference-counted arrow `Schema` (cached at construction)
    schema: ArrowSchemaRef,
}

impl IcebergCatalogTableProvider {
    /// Creates a new catalog-backed table provider.
    ///
    /// Loads the table once to get the initial schema, then stores the catalog
    /// reference for future metadata refreshes on each operation.
    pub async fn try_new(
        catalog: Arc<dyn Catalog>,
        namespace: NamespaceIdent,
        name: impl Into<String>,
    ) -> Result<Self> {
        let table_ident = TableIdent::new(namespace, name.into());

        // Load table once to get initial schema
        let table = catalog
            .load_table(&table_ident)
            .await
            .map_err(to_datafusion_error)?;
        let schema = Arc::new(
            schema_to_arrow_schema(table.metadata().current_schema())
                .map_err(to_datafusion_error)?,
        );

        Ok(IcebergCatalogTableProvider {
            catalog,
            table_ident,
            schema,
        })
    }

    pub(crate) async fn metadata_table(
        &self,
        r#type: MetadataTableType,
    ) -> Result<IcebergMetadataTableProvider> {
        // Load fresh table metadata for metadata table access
        let table = self
            .catalog
            .load_table(&self.table_ident)
            .await
            .map_err(to_datafusion_error)?;
        IcebergMetadataTableProvider::try_new(table, r#type)
    }

    /// The catalog this provider loads its table from and commits through.
    pub fn catalog(&self) -> &Arc<dyn Catalog> {
        &self.catalog
    }

    /// The identifier of the table this provider reads and writes.
    pub fn table_ident(&self) -> &TableIdent {
        &self.table_ident
    }
}

#[async_trait]
impl TableProvider for IcebergCatalogTableProvider {
    fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Load fresh table metadata from catalog
        let table = self
            .catalog
            .load_table(&self.table_ident)
            .await
            .map_err(to_datafusion_error)?;

        // Create scan with fresh metadata (always use current snapshot).
        let schema = project_schema(&self.schema, projection)?;
        let tasks = FileScanTaskProviderBuilder::new(
            state.task_ctx(),
            table.clone(),
            Arc::clone(&schema),
        )
        .with_filters(filters)
        .build()
        .await?;
        let source = IcebergDataSource::builder(
            state.config(),
            schema,
            table.file_io().clone(),
            table.reader_builder(),
            tasks,
        )
        .with_fetch(limit)
        .build()?;
        Ok(DataSourceExec::from_data_source(source))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        // Push down all filters, as a single source of truth, the scanner will drop the filters which couldn't be push down
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn insert_into(
        &self,
        state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if insert_op != InsertOp::Append {
            return not_impl_err!(
                "IcebergCatalogTableProvider supports only append inserts, \
                 got {insert_op}"
            );
        }

        // Load fresh table metadata from catalog
        let table = self
            .catalog
            .load_table(&self.table_ident)
            .await
            .map_err(to_datafusion_error)?;

        let partition_spec = table.metadata().default_partition_spec();

        // Step 1: Project partition values for partitioned tables
        let plan_with_partition = if !partition_spec.is_unpartitioned() {
            project_with_partition(input, &table)?
        } else {
            input
        };

        // Step 2: Repartition for parallel processing
        let target_partitions = NonZeroUsize::new(state.config().target_partitions())
            .ok_or_else(|| {
                config_datafusion_err!("target_partitions must be greater than 0")
            })?;

        let repartitioned_plan =
            repartition(plan_with_partition, table.metadata_ref(), target_partitions)?;

        let write_plan =
            Arc::new(IcebergWriteExec::new(table.clone(), repartitioned_plan));

        // Merge the outputs of write_plan into one so we can commit all files together
        let coalesce_partitions = Arc::new(CoalescePartitionsExec::new(write_plan));

        Ok(Arc::new(IcebergCommitExec::new(
            table,
            self.catalog.clone(),
            coalesce_partitions,
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use datafusion::datasource::TableProvider;
    use datafusion::error::DataFusionError;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::prelude::SessionContext;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    use iceberg::{CatalogBuilder, TableCreation};
    use tempfile::TempDir;

    use super::*;
    use crate::read::IcebergDataSource;

    #[tokio::test]
    async fn test_catalog_backed_provider_creation() {
        let (catalog, namespace, table_name, _temp_dir) =
            get_test_catalog_and_table().await;

        // Test creating a catalog-backed provider
        let provider = IcebergCatalogTableProvider::try_new(
            catalog.clone(),
            namespace.clone(),
            table_name.clone(),
        )
        .await
        .unwrap();

        // Verify the schema is loaded correctly
        let schema = provider.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "name");
    }

    #[tokio::test]
    async fn test_catalog_backed_provider_scan() {
        let (catalog, namespace, table_name, _temp_dir) =
            get_test_catalog_and_table().await;

        let provider = IcebergCatalogTableProvider::try_new(
            catalog.clone(),
            namespace.clone(),
            table_name.clone(),
        )
        .await
        .unwrap();

        let ctx = SessionContext::new();
        ctx.register_table("test_table", Arc::new(provider))
            .unwrap();

        // Test that scan operations work correctly
        let df = ctx.sql("SELECT * FROM test_table").await.unwrap();

        // Verify the schema in the query result
        let df_schema = df.schema();
        assert_eq!(df_schema.fields().len(), 2);
        assert_eq!(df_schema.field(0).name(), "id");
        assert_eq!(df_schema.field(1).name(), "name");

        let physical_plan = df.create_physical_plan().await;
        assert!(physical_plan.is_ok());
    }

    #[tokio::test]
    async fn test_catalog_backed_provider_insert() {
        let (catalog, namespace, table_name, _temp_dir) =
            get_test_catalog_and_table().await;

        let provider = IcebergCatalogTableProvider::try_new(
            catalog.clone(),
            namespace.clone(),
            table_name.clone(),
        )
        .await
        .unwrap();

        let ctx = SessionContext::new();
        ctx.register_table("test_table", Arc::new(provider))
            .unwrap();

        // Test that insert operations work correctly
        let result = ctx.sql("INSERT INTO test_table VALUES (1, 'test')").await;

        // Insert should succeed (or at least not fail during planning)
        assert!(result.is_ok());

        // Try to execute the insert plan
        let df = result.unwrap();
        let execution_result = df.collect().await;

        // The execution should succeed
        assert!(execution_result.is_ok());
    }

    #[tokio::test]
    async fn test_physical_input_schema_consistent_with_logical_input_schema() {
        let (catalog, namespace, table_name, _temp_dir) =
            get_test_catalog_and_table().await;

        let provider = IcebergCatalogTableProvider::try_new(
            catalog.clone(),
            namespace.clone(),
            table_name.clone(),
        )
        .await
        .unwrap();

        let ctx = SessionContext::new();
        ctx.register_table("test_table", Arc::new(provider))
            .unwrap();

        // Create a query plan
        let df = ctx.sql("SELECT id, name FROM test_table").await.unwrap();

        // Get logical schema before consuming df
        let logical_schema = df.schema().clone();

        // Get physical plan (this consumes df)
        let physical_plan = df.create_physical_plan().await.unwrap();
        let physical_schema = physical_plan.schema();

        // Verify that logical and physical schemas are consistent
        assert_eq!(
            logical_schema.fields().len(),
            physical_schema.fields().len()
        );

        for (logical_field, physical_field) in logical_schema
            .fields()
            .iter()
            .zip(physical_schema.fields().iter())
        {
            assert_eq!(logical_field.name(), physical_field.name());
            assert_eq!(logical_field.data_type(), physical_field.data_type());
        }
    }

    #[tokio::test]
    async fn test_catalog_backed_provider_rejects_non_append_op() {
        use datafusion::physical_plan::empty::EmptyExec;

        let (catalog, namespace, table_name, _temp_dir) =
            get_test_catalog_and_table().await;
        let provider =
            IcebergCatalogTableProvider::try_new(catalog, namespace, table_name)
                .await
                .unwrap();
        let ctx = SessionContext::new();

        for (insert_op, expected_message) in [
            (
                InsertOp::Overwrite,
                "IcebergCatalogTableProvider supports only append inserts, got Insert Overwrite",
            ),
            (
                InsertOp::Replace,
                "IcebergCatalogTableProvider supports only append inserts, got Replace Into",
            ),
        ] {
            let input =
                Arc::new(EmptyExec::new(provider.schema())) as Arc<dyn ExecutionPlan>;
            let error = provider
                .insert_into(&ctx.state(), input, insert_op)
                .await
                .expect_err("non-append inserts should be rejected");

            assert!(
                matches!(
                    error,
                    DataFusionError::NotImplemented(ref message) if message == expected_message
                ),
                "unexpected error: {error}"
            );
        }
    }

    #[tokio::test]
    async fn test_limit_pushdown_catalog_backed_provider() {
        use datafusion::datasource::TableProvider;

        let (catalog, namespace, table_name, _temp_dir) =
            get_test_catalog_and_table().await;

        let provider = IcebergCatalogTableProvider::try_new(
            catalog.clone(),
            namespace.clone(),
            table_name.clone(),
        )
        .await
        .unwrap();

        let ctx = SessionContext::new();
        let state = ctx.state();

        // Test scan with limit
        let scan_plan = provider.scan(&state, None, &[], Some(5)).await.unwrap();

        // Verify that the scan plan wraps an IcebergDataSource.
        let iceberg_scan = scan_plan
            .downcast_ref::<DataSourceExec>()
            .expect("Expected DataSourceExec")
            .data_source()
            .downcast_ref::<IcebergDataSource>()
            .expect("Expected IcebergDataSource");

        // Verify the limit is set
        assert_eq!(
            iceberg_scan.fetch(),
            Some(5),
            "Limit should be set to 5 in the scan plan"
        );
    }

    async fn get_test_catalog_and_table()
    -> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir) {
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

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                    .into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String))
                    .into(),
            ])
            .build()
            .unwrap();

        let table_creation = TableCreation::builder()
            .name("test_table".to_string())
            .location(format!("{warehouse_path}/test_table"))
            .schema(schema)
            .properties(HashMap::new())
            .build();

        catalog
            .create_table(&namespace, table_creation)
            .await
            .unwrap();

        (
            Arc::new(catalog),
            namespace,
            "test_table".to_string(),
            temp_dir,
        )
    }
}
