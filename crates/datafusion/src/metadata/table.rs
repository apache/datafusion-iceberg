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
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::inspect::MetadataTableType;
use iceberg::table::Table;

use super::scan::IcebergMetadataDataSource;
use crate::to_datafusion_error;

/// Represents a [`TableProvider`] for the Iceberg [`iceberg::Catalog`],
/// managing access to a [`iceberg::inspect::MetadataTable`].
#[derive(Debug, Clone)]
pub struct IcebergMetadataTableProvider {
    table: Table,
    r#type: MetadataTableType,
    schema: ArrowSchemaRef,
}

#[async_trait]
impl TableProvider for IcebergMetadataTableProvider {
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
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(DataSourceExec::from_data_source(
            IcebergMetadataDataSource::try_new(self.clone(), projection)?,
        ))
    }
}

impl IcebergMetadataTableProvider {
    /// Creates a provider for the `metadata_type` metadata table of an
    /// already loaded table.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata table's schema has no Arrow equivalent.
    pub fn try_new(table: Table, metadata_type: MetadataTableType) -> Result<Self> {
        let metadata_table = table.inspect();
        let schema = match metadata_type {
            MetadataTableType::Snapshots => metadata_table.snapshots().schema(),
            MetadataTableType::Manifests => metadata_table.manifests().schema(),
            MetadataTableType::History => metadata_table.history().schema(),
        };
        let schema =
            Arc::new(schema_to_arrow_schema(&schema).map_err(to_datafusion_error)?);
        Ok(Self {
            table,
            r#type: metadata_type,
            schema,
        })
    }

    /// The table whose metadata this provider reads.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Which metadata table of [`Self::table`] this provider reads.
    pub fn metadata_type(&self) -> &MetadataTableType {
        &self.r#type
    }
}
