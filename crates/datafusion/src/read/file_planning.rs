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

//! Query-local file work selected before execution, independently of task assignment.

use std::mem::{size_of, size_of_val};

use datafusion::common::Result;
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use futures::TryStreamExt;
use iceberg::scan::{FileScanTask, TableScan};

use crate::to_datafusion_error;

/// File work selected by one Iceberg scan before assignment to output partitions.
///
/// The tasks retain their bound predicates, projection, and delete references. Keeping
/// this value alive also keeps the query memory reservation for the retained tasks.
/// This is not a reader and contains no coordinator-local reader state.
#[derive(Debug)]
pub(crate) struct PlannedFileScan {
    tasks: Vec<FileScanTask>,
    _reservation: MemoryReservation,
}

impl PlannedFileScan {
    /// Tasks in Iceberg's planning order. Assign each once per execution.
    pub(crate) fn tasks(&self) -> &[FileScanTask] {
        &self.tasks
    }

    pub(crate) async fn collect(scan: TableScan, context: &TaskContext) -> Result<Self> {
        let reservation =
            MemoryConsumer::new("Iceberg file planning").register(context.memory_pool());
        let mut planned = Self {
            tasks: Vec::new(),
            _reservation: reservation,
        };
        let mut stream = scan.plan_files().await.map_err(to_datafusion_error)?;
        while let Some(task) = stream.try_next().await.map_err(to_datafusion_error)? {
            // Allow for spare Vec capacity and allocator overhead; schemas and specs
            // are shared Arcs, but paths and delete references are owned by each task.
            planned
                ._reservation
                .try_grow(retained_size(&task).saturating_mul(2))?;
            planned.tasks.push(task);
        }
        Ok(planned)
    }
}

fn retained_size(task: &FileScanTask) -> usize {
    let mut bytes = size_of::<FileScanTask>()
        + task.data_file_path().len()
        + size_of_val(task.project_field_ids())
        + size_of_val(task.deletes())
        + task.key_metadata().map_or(0, <[u8]>::len);
    for delete in task.deletes() {
        bytes = bytes
            .saturating_add(delete.file_path().len())
            .saturating_add(delete.referenced_data_file().map_or(0, str::len))
            .saturating_add(delete.equality_ids().map_or(0, size_of_val))
            .saturating_add(delete.key_metadata().map_or(0, <[u8]>::len));
    }
    bytes
}
