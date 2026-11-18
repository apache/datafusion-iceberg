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
use datafusion::physical_plan::Partitioning;
use futures::StreamExt;
use iceberg::scan::FileScanTask;
use std::collections::BTreeSet;
use std::fmt::Formatter;
use std::sync::Arc;

use super::provider::{FileScanTaskProvider, FileScanTaskProviderContext};

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
        write!(
            f,
            ", provider=eager, planned_files={}",
            self.tasks().count()
        )?;
        // Assigned tasks may have different predicates. Report the distinct task
        // predicates rather than assuming a single filter applies to every file.
        let predicates = self
            .tasks()
            .filter_map(FileScanTask::predicate)
            .map(ToString::to_string)
            .collect::<BTreeSet<_>>();
        if !predicates.is_empty() {
            write!(
                f,
                ", predicates=[{}]",
                predicates.into_iter().collect::<Vec<_>>().join(", ")
            )?;
        }
        Ok(())
    }
}
