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

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::catalog::Session;
use datafusion::common::{exec_datafusion_err, not_impl_err};
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::table::Table;

use crate::error::to_datafusion_error;
use crate::read::IcebergDataSource;

/// Static table provider for read-only snapshot access.
///
/// This provider holds a cached table instance and does not refresh metadata or support
/// write operations. Use this for consistent analytical queries, time-travel scenarios,
/// or when you want to avoid catalog overhead.
///
/// For catalog-backed tables with write support and automatic refresh, use
/// [`IcebergTableProvider`](super::IcebergCatalogTableProvider) instead.
#[derive(Debug, Clone)]
pub struct IcebergStaticTableProvider {
    /// The static table instance (never refreshed)
    table: Table,
    /// Optional snapshot ID for this static view
    snapshot_id: Option<i64>,
    /// A reference-counted arrow `Schema`
    schema: ArrowSchemaRef,
}

impl IcebergStaticTableProvider {
    /// Creates a static provider from a table instance.
    ///
    /// Uses the table's current snapshot for all queries. Does not support write operations.
    pub async fn try_new_from_table(table: Table) -> Result<Self> {
        let schema = Arc::new(
            schema_to_arrow_schema(table.metadata().current_schema())
                .map_err(to_datafusion_error)?,
        );
        Ok(IcebergStaticTableProvider {
            table,
            snapshot_id: None,
            schema,
        })
    }

    /// Creates a static provider for a specific table snapshot.
    ///
    /// Queries the specified snapshot for all operations. Useful for time-travel queries.
    /// Does not support write operations.
    pub async fn try_new_from_table_snapshot(
        table: Table,
        snapshot_id: i64,
    ) -> Result<Self> {
        let snapshot = table
            .metadata()
            .snapshot_by_id(snapshot_id)
            .ok_or_else(|| {
                exec_datafusion_err!(
                    "snapshot id {snapshot_id} not found in table {}",
                    table.identifier().name()
                )
            })?;
        let table_schema = snapshot
            .schema(table.metadata())
            .map_err(to_datafusion_error)?;
        let schema =
            Arc::new(schema_to_arrow_schema(&table_schema).map_err(to_datafusion_error)?);
        Ok(IcebergStaticTableProvider {
            table,
            snapshot_id: Some(snapshot_id),
            schema,
        })
    }

    /// The table as loaded when this provider was built.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// The snapshot this provider reads, or `None` for the current snapshot of
    /// [`Self::table`].
    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }
}

#[async_trait]
impl TableProvider for IcebergStaticTableProvider {
    fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Use cached table (no refresh)
        Ok(DataSourceExec::from_data_source(IcebergDataSource::new(
            self.table.clone(),
            self.snapshot_id,
            self.schema.clone(),
            projection,
            filters,
            limit,
        )?))
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
        _state: &dyn Session,
        _input: Arc<dyn ExecutionPlan>,
        _insert_op: InsertOp,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        not_impl_err!(
            "Write operations are not supported on IcebergStaticTableProvider. \
             Use IcebergTableProvider with a catalog for write support."
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::common::Column;
    use datafusion::prelude::SessionContext;
    use iceberg::TableIdent;
    use iceberg::io::FileIO;
    use iceberg::table::{StaticTable, Table};

    use super::*;

    #[tokio::test]
    async fn test_static_provider_from_table() {
        let table = get_test_table_from_metadata_file().await;
        let table_provider =
            IcebergStaticTableProvider::try_new_from_table(table.clone())
                .await
                .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();
        let df = ctx.sql("SELECT * FROM mytable").await.unwrap();
        let df_schema = df.schema();
        let df_columns = df_schema.fields();
        assert_eq!(df_columns.len(), 3);
        let x_column = df_columns.first().unwrap();
        let column_data = format!(
            "{:?}:{:?}",
            x_column.name(),
            x_column.data_type().to_string()
        );
        assert_eq!(column_data, "\"x\":\"Int64\"");
        let has_column = df_schema.has_column(&Column::from_name("z"));
        assert!(has_column);
    }

    #[tokio::test]
    async fn test_static_provider_from_snapshot() {
        let table = get_test_table_from_metadata_file().await;
        let snapshot_id = table.metadata().snapshots().next().unwrap().snapshot_id();
        let table_provider = IcebergStaticTableProvider::try_new_from_table_snapshot(
            table.clone(),
            snapshot_id,
        )
        .await
        .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();
        let df = ctx.sql("SELECT * FROM mytable").await.unwrap();
        let df_schema = df.schema();
        let df_columns = df_schema.fields();
        assert_eq!(df_columns.len(), 3);
        let x_column = df_columns.first().unwrap();
        let column_data = format!(
            "{:?}:{:?}",
            x_column.name(),
            x_column.data_type().to_string()
        );
        assert_eq!(column_data, "\"x\":\"Int64\"");
        let has_column = df_schema.has_column(&Column::from_name("z"));
        assert!(has_column);
    }

    #[tokio::test]
    async fn test_static_provider_rejects_writes() {
        let table = get_test_table_from_metadata_file().await;
        let table_provider =
            IcebergStaticTableProvider::try_new_from_table(table.clone())
                .await
                .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();

        // Attempt to insert into the static provider should fail
        let result = ctx.sql("INSERT INTO mytable VALUES (1, 2, 3)").await;

        // The error should occur during planning or execution
        // We expect an error indicating write operations are not supported
        assert!(
            result.is_err() || {
                let df = result.unwrap();
                df.collect().await.is_err()
            }
        );
    }

    #[tokio::test]
    async fn test_static_provider_scan() {
        let table = get_test_table_from_metadata_file().await;
        let table_provider =
            IcebergStaticTableProvider::try_new_from_table(table.clone())
                .await
                .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();

        // Test that scan operations work correctly
        let df = ctx.sql("SELECT count(*) FROM mytable").await.unwrap();
        let physical_plan = df.create_physical_plan().await;
        assert!(physical_plan.is_ok());
    }

    #[tokio::test]
    async fn test_limit_pushdown_static_provider() {
        use datafusion::datasource::TableProvider;

        let table = get_test_table_from_metadata_file().await;
        let table_provider =
            IcebergStaticTableProvider::try_new_from_table(table.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        let state = ctx.state();

        // Test scan with limit
        let scan_plan = table_provider
            .scan(&state, None, &[], Some(10))
            .await
            .unwrap();

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
            Some(10),
            "Limit should be set to 10 in the scan plan"
        );
    }

    #[tokio::test]
    async fn test_no_limit_pushdown() {
        use datafusion::datasource::TableProvider;

        let table = get_test_table_from_metadata_file().await;
        let table_provider =
            IcebergStaticTableProvider::try_new_from_table(table.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        let state = ctx.state();

        // Test scan without limit
        let scan_plan = table_provider.scan(&state, None, &[], None).await.unwrap();

        // Verify that the scan plan wraps an IcebergDataSource.
        let iceberg_scan = scan_plan
            .downcast_ref::<DataSourceExec>()
            .expect("Expected DataSourceExec")
            .data_source()
            .downcast_ref::<IcebergDataSource>()
            .expect("Expected IcebergDataSource");

        // Verify the limit is None
        assert_eq!(
            iceberg_scan.fetch(),
            None,
            "Limit should be None when not specified"
        );
    }

    async fn get_test_table_from_metadata_file() -> Table {
        let metadata_file_name = "TableMetadataV2Valid.json";
        let metadata_file_path = format!(
            "{}/tests/test_data/{}",
            env!("CARGO_MANIFEST_DIR"),
            metadata_file_name
        );
        let file_io = FileIO::new_with_fs();
        let static_identifier =
            TableIdent::from_strs(["static_ns", "static_table"]).unwrap();
        let static_table = StaticTable::from_metadata_file(
            &metadata_file_path,
            static_identifier,
            file_io,
        )
        .await
        .unwrap();
        static_table.into_table()
    }
}
