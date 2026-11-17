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

use datafusion::common::Result;
use datafusion::physical_plan::Partitioning;
use futures::TryStreamExt;
use iceberg::expr::Predicate;

use super::provider::{FileScanTaskProvider, FileScanTaskProviderContext};
use crate::to_datafusion_error;

/// Discovers files while executing, in a single partition: the files of a
/// snapshot that may match a predicate.
#[derive(Debug, Clone)]
pub struct LazyFileScanTaskProvider {
    pub(crate) snapshot_id: Option<i64>,
    pub(crate) projection: Vec<String>,
    pub(crate) predicate: Option<Predicate>,
}

impl LazyFileScanTaskProvider {
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
        _partition: usize,
        ctx: FileScanTaskProviderContext,
    ) -> Result<iceberg::scan::FileScanTaskStream> {
        let scan = match self.snapshot_id {
            Some(snapshot_id) => ctx.table.scan().snapshot_id(snapshot_id),
            None => ctx.table.scan(),
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
