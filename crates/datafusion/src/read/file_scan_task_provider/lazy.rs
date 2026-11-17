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
    snapshot_id: Option<i64>,
    projection: Vec<String>,
    predicate: Option<Predicate>,
}

impl LazyFileScanTaskProvider {
    /// Retains the table and projected column names without discovering files.
    /// The names must match the data source's output schema.
    pub fn new(table: Table, projection: Vec<String>) -> Self {
        Self {
            table,
            projection,
            snapshot_id: None,
            predicate: None,
        }
    }

    /// Selects a snapshot, or the retained table's current snapshot for `None`.
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

    /// Table retained for discovery, including its metadata and storage access.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Names of the columns selected during discovery.
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
