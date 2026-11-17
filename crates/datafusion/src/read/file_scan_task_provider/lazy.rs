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
use std::fmt::Formatter;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{Result, exec_err};
use datafusion::physical_plan::Partitioning;
use datafusion::prelude::Expr;
use futures::TryStreamExt;
use iceberg::expr::Predicate;
use iceberg::table::Table;

use super::provider::{FileScanTaskProvider, FileScanTaskProviderContext};
use crate::read::convert_filters_to_predicate;
use crate::to_datafusion_error;

/// Discovers files while executing, in a single partition: the files of a
/// snapshot that may match a predicate.
#[derive(Debug, Clone)]
pub struct LazyFileScanTaskProvider {
    table: Table,
    schema: SchemaRef,
    snapshot_id: Option<i64>,
    projection: Vec<String>,
    predicate: Option<Predicate>,
}

impl LazyFileScanTaskProvider {
    /// Configures discovery from a table and the schema advertised by its table
    /// provider. Projection indices refer to this schema, not the table's current
    /// schema. To reconstruct a provider, pass its already-projected schema and
    /// omit the projection.
    pub fn builder(table: Table, schema: SchemaRef) -> LazyFileScanTaskProviderBuilder {
        LazyFileScanTaskProviderBuilder {
            table,
            schema,
            projection: None,
            snapshot_id: None,
            predicate: None,
        }
    }

    /// Table retained for discovery, including its metadata and storage access.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Output schema after projection, matching the columns selected by discovery.
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Names of the columns selected during discovery, in output schema order.
    pub fn projection(&self) -> &[String] {
        &self.projection
    }

    /// Snapshot to scan, or the table's current one if `None`.
    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    /// Filter used to skip files, if any.
    pub fn predicate(&self) -> Option<&Predicate> {
        self.predicate.as_ref()
    }
}

/// Configures a table scan without discovering files or constructing a reader.
///
/// The builder applies projection once, retaining both the resulting Arrow schema
/// and its column names in the provider. Reader settings and fetch belong to the
/// data source, not to file discovery.
///
/// ```
/// # use datafusion::{arrow::datatypes::SchemaRef, common::Result, prelude::Expr};
/// # use iceberg::table::Table;
/// # use iceberg_datafusion::read::LazyFileScanTaskProvider;
/// fn discovery(
///     table: Table,
///     schema: SchemaRef,
///     projection: Option<&Vec<usize>>,
///     filters: &[Expr],
/// ) -> Result<LazyFileScanTaskProvider> {
///     LazyFileScanTaskProvider::builder(table, schema)
///         .with_projection(projection)
///         .with_filters(filters)
///         .build()
/// }
/// ```
#[derive(Debug, Clone)]
pub struct LazyFileScanTaskProviderBuilder {
    table: Table,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    snapshot_id: Option<i64>,
    predicate: Option<Predicate>,
}

impl LazyFileScanTaskProviderBuilder {
    /// Selects columns from the supplied schema in this order, or all its columns
    /// for `None`. Indices are validated when [`Self::build`] is called.
    pub fn with_projection(mut self, projection: Option<&Vec<usize>>) -> Self {
        self.projection = projection.cloned();
        self
    }

    /// Selects a snapshot, or the retained table's current snapshot for `None`.
    /// The supplied schema is not replaced with the snapshot's schema.
    pub fn with_snapshot_id(mut self, snapshot_id: Option<i64>) -> Self {
        self.snapshot_id = snapshot_id;
        self
    }

    /// Sets the predicate used to prune files and filter their rows.
    pub fn with_predicate(mut self, predicate: Option<Predicate>) -> Self {
        self.predicate = predicate;
        self
    }

    /// Pushes down supported parts of DataFusion filters. The caller must still
    /// apply the original filters, as unsupported parts are omitted.
    pub fn with_filters(self, filters: &[Expr]) -> Self {
        self.with_predicate(convert_filters_to_predicate(filters))
    }

    /// Validates projection and builds a lazy provider without file discovery.
    /// Its stream discovers files only when polled, or can be collected before
    /// execution by [`super::eager::EagerFileScanTaskProvider::plan`].
    ///
    /// # Errors
    ///
    /// Returns an error if a projection index is outside the supplied schema.
    pub fn build(self) -> Result<LazyFileScanTaskProvider> {
        let schema = match self.projection {
            Some(projection) => Arc::new(self.schema.project(&projection)?),
            None => self.schema,
        };
        let projection = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        Ok(LazyFileScanTaskProvider {
            table: self.table,
            schema,
            projection,
            snapshot_id: self.snapshot_id,
            predicate: self.predicate,
        })
    }
}

impl FileScanTaskProvider for LazyFileScanTaskProvider {
    fn partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn open(
        self: Arc<Self>,
        partition: usize,
        _ctx: FileScanTaskProviderContext,
    ) -> Result<iceberg::scan::FileScanTaskStream> {
        if partition != 0 {
            return exec_err!("Invalid Iceberg scan partition {partition}");
        }
        let scan = match self.snapshot_id {
            Some(snapshot_id) => self.table.scan().snapshot_id(snapshot_id),
            None => self.table.scan(),
        };
        let mut scan = scan.select(&self.projection);
        if let Some(predicate) = &self.predicate {
            scan = scan.with_filter(predicate.clone());
        }
        let scan = scan.build().map_err(to_datafusion_error)?;
        // Files are discovered when the stream is first polled, not here.
        Ok(Box::pin(
            futures::stream::once(async move { scan.plan_files().await }).try_flatten(),
        ))
    }

    fn fmt_as(&self, f: &mut Formatter) -> std::fmt::Result {
        if let Some(predicate) = &self.predicate {
            write!(f, ", predicate={predicate}")?;
        }
        Ok(())
    }
}
