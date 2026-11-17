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

use datafusion::common::{Result, exec_err};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::Partitioning;
use futures::{StreamExt, TryStreamExt};
use iceberg::scan::FileScanTask;
use std::fmt::Formatter;
use std::sync::Arc;

use super::lazy::LazyFileScanTaskProvider;
use super::provider::{FileScanTaskProvider, FileScanTaskProviderContext};
use crate::to_datafusion_error;

/// Files discovered before execution, in groups that each become one output
/// partition. Group order and empty groups are retained.
#[derive(Debug)]
pub struct EagerFileScanTaskProvider {
    file_scan_tasks: Vec<Vec<FileScanTask>>,
}

impl EagerFileScanTaskProvider {
    /// Reads `file_scan_tasks`, one output partition per group.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no groups. Empty groups are allowed.
    pub fn try_new(file_scan_tasks: Vec<Vec<FileScanTask>>) -> Result<Self> {
        if file_scan_tasks.is_empty() {
            return exec_err!(
                "An Iceberg task provider requires at least one file group"
            );
        }
        Ok(Self { file_scan_tasks })
    }

    /// Discovers all tasks now, split round-robin into up to the context's target
    /// partitions. The returned provider retains only the tasks, not the table.
    /// Dropping this future drops the discovery stream; no background collector
    /// is spawned. An empty scan retains one empty output partition.
    pub async fn plan(
        discovery: Arc<LazyFileScanTaskProvider>,
        context: Arc<TaskContext>,
    ) -> Result<Self> {
        let target = context.session_config().target_partitions();
        if target == 0 {
            return exec_err!(
                "Iceberg eager planning requires at least one target partition"
            );
        }
        let mut tasks = discovery.open(
            0,
            FileScanTaskProviderContext {
                task_context: &context,
            },
        )?;
        let mut groups = Vec::new();
        let mut partition = 0;
        while let Some(task) = tasks.try_next().await.map_err(to_datafusion_error)? {
            if partition == groups.len() {
                groups.push(Vec::new());
            }
            groups[partition].push(task);
            partition = (partition + 1) % target;
        }
        if groups.is_empty() {
            groups.push(Vec::new());
        }
        Self::try_new(groups)
    }

    /// All file tasks, in partition order and then group order.
    pub fn tasks(&self) -> impl Iterator<Item = &FileScanTask> {
        self.file_scan_tasks.iter().flatten()
    }

    /// The file tasks of each partition.
    pub fn groups(&self) -> &[Vec<FileScanTask>] {
        &self.file_scan_tasks
    }
}

impl FileScanTaskProvider for EagerFileScanTaskProvider {
    fn partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.groups().len())
    }

    fn open(
        self: Arc<Self>,
        partition: usize,
        _context: FileScanTaskProviderContext,
    ) -> Result<iceberg::scan::FileScanTaskStream> {
        let Some(count) = self.groups().get(partition).map(Vec::len) else {
            return exec_err!("Invalid Iceberg scan partition {partition}");
        };
        // Clone each task as it's read; holding `self` keeps the groups alive
        // after the data source is dropped.
        Ok(futures::stream::iter(0..count)
            .map(move |offset| Ok(self.groups()[partition][offset].clone()))
            .boxed())
    }

    fn fmt_as(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, ", planned_files={}", self.tasks().count())
    }
}
