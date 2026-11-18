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

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion::common::{Result, internal_err};
use datafusion::execution::TaskContext;
use futures::TryStreamExt;
use iceberg::expr::Predicate;
use iceberg::scan::FileScanTaskStream;
use iceberg::table::Table;

use crate::to_datafusion_error;

/// Supplies selected Iceberg file tasks to an [`IcebergDataSource`](super::IcebergDataSource).
///
/// Implementations may discover tasks locally or receive them from another
/// process. Tasks carry their projection, predicate, schema and delete references;
/// implementations must preserve these, and the data source's output schema must
/// match the tasks' projection. Tasks must be readable with the source's FileIO.
///
/// The partition count must be positive and stable. Different partitions may be
/// opened concurrently. Clones preserve the assignment but reset source-local
/// execution state, such as claimed partitions. External state in `TaskContext`
/// may still be scoped to one execution. Returned streams must retain resources
/// they need after the task source is dropped, and release them when dropped.
/// This interface does not require serialization, networking or eager collection.
pub trait FileScanTaskSource: Clone + Debug + Send + Sync + 'static {
    /// Number of local output partitions, not a worker or distributed task count.
    fn partition_count(&self) -> usize;

    /// Opens a partition's stream during execution, without collecting it.
    ///
    /// Discovery can start here or when the stream is polled. Calls on one
    /// instance may coordinate discovery; a repeated claim of the same partition
    /// may be rejected. Implementations must not block waiting for tasks here.
    fn tasks(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<FileScanTaskStream>;

    /// Appends task-source details to the data source's explain output.
    fn fmt_as(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

/// Default single-partition discovery for an Iceberg table scan.
#[derive(Debug, Clone)]
pub struct TableScanTaskSource {
    pub(super) table: Table,
    pub(super) snapshot_id: Option<i64>,
    pub(super) projection: Vec<String>,
    pub(super) predicate: Option<Predicate>,
}

impl TableScanTaskSource {
    /// Table supplying metadata and file discovery.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Selected snapshot, or the table's current snapshot for `None`.
    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    /// Names of the selected output columns.
    pub fn projection(&self) -> &[String] {
        &self.projection
    }

    /// Predicate applied during discovery, if any.
    pub fn predicate(&self) -> Option<&Predicate> {
        self.predicate.as_ref()
    }
}

impl FileScanTaskSource for TableScanTaskSource {
    fn partition_count(&self) -> usize {
        1
    }

    fn tasks(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<FileScanTaskStream> {
        if partition != 0 {
            return internal_err!("Iceberg scan partition {partition} does not exist");
        }
        let mut builder = self.table.scan().select(self.projection.clone());
        if let Some(snapshot_id) = self.snapshot_id {
            builder = builder.snapshot_id(snapshot_id);
        }
        if let Some(predicate) = &self.predicate {
            builder = builder.with_filter(predicate.clone());
        }
        let scan = builder.build().map_err(to_datafusion_error)?;
        Ok(Box::pin(
            futures::stream::once(async move { scan.plan_files().await }).try_flatten(),
        ))
    }

    fn fmt_as(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, ", projection=[{}]", self.projection.join(", "))?;
        if let Some(predicate) = &self.predicate {
            write!(f, ", predicate={predicate}")?;
        }
        Ok(())
    }
}
