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
use std::sync::Arc;

use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::Schema;
use iceberg::{Catalog, CatalogBuilder, Result, TableIdent};
use tempfile::TempDir;

use crate::TableBuilder;

/// An in-memory Iceberg catalog over a temporary directory, holding the
/// tables a test builds. It isn't tied to any dataset.
///
/// It owns every file its tables write. Dropping it deletes them, so keep it
/// alive while anything reads those tables. Create tables with
/// [`FixtureCatalog::table`]. Each test should build its own; nothing is
/// shared between them.
pub struct FixtureCatalog {
    catalog: Arc<MemoryCatalog>,
    _directory: TempDir,
}

impl FixtureCatalog {
    /// Creates an empty catalog in a new temporary directory.
    pub async fn new() -> Result<Self> {
        let directory = TempDir::new()?;
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "fixture",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    directory.path().display().to_string(),
                )]),
            )
            .await?;
        Ok(Self {
            catalog: Arc::new(catalog),
            _directory: directory,
        })
    }

    /// The underlying Iceberg catalog, e.g. to pass to
    /// `IcebergCatalogProvider::try_new`.
    pub fn catalog(&self) -> Arc<dyn Catalog> {
        self.catalog.clone()
    }

    /// Starts building table `ident` with `schema`. Its namespace is created
    /// if it doesn't exist.
    pub fn table(&self, ident: TableIdent, schema: Schema) -> TableBuilder {
        TableBuilder::new(self.catalog.clone(), ident, schema)
    }
}
