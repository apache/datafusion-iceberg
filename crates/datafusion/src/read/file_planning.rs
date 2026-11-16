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

use std::iter::Flatten;
use std::mem::{size_of, size_of_val};
use std::slice::Iter;

use datafusion::common::{Result, exec_err};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use futures::TryStreamExt;
use iceberg::scan::{FileScanTask, TableScan};

use crate::to_datafusion_error;

/// File groups and their descriptor reservation share one lifetime. Each group is
/// one output partition; supplied group order and empty partitions are retained.
#[derive(Debug)]
pub(crate) struct PlannedFileScan {
    groups: Vec<Vec<FileScanTask>>,
    _reservation: MemoryReservation,
}

impl PlannedFileScan {
    pub(crate) fn tasks(&self) -> impl ExactSizeIterator<Item = &FileScanTask> {
        PlannedFiles {
            inner: self.groups.iter().flatten(),
            remaining: self.groups.iter().map(Vec::len).sum(),
        }
    }

    pub(crate) fn groups(&self) -> &[Vec<FileScanTask>] {
        &self.groups
    }

    pub(super) fn new(context: &TaskContext) -> Self {
        Self {
            groups: Vec::new(),
            _reservation: MemoryConsumer::new("Iceberg file planning")
                .register(context.memory_pool()),
        }
    }

    pub(crate) fn from_groups(
        groups: Vec<Vec<FileScanTask>>,
        context: &TaskContext,
    ) -> Result<Self> {
        Self::reserve_groups(
            groups,
            MemoryConsumer::new("Iceberg file planning").register(context.memory_pool()),
        )
    }

    /// Own a replacement assignment, charging it independently to the same pool.
    pub(crate) fn with_groups(&self, groups: Vec<Vec<FileScanTask>>) -> Result<Self> {
        Self::reserve_groups(groups, self._reservation.new_empty())
    }

    fn reserve_groups(
        mut groups: Vec<Vec<FileScanTask>>,
        reservation: MemoryReservation,
    ) -> Result<Self> {
        if groups.is_empty() {
            return exec_err!("Iceberg scan requires at least one file group");
        }
        // Do not retain arbitrary spare capacity from caller-owned buffers.
        // Previously, copying into the flat assignment discarded that capacity.
        groups.shrink_to_fit();
        for group in &mut groups {
            group.shrink_to_fit();
            for task in group {
                reservation.try_grow(retained_size(task).saturating_mul(2))?;
            }
        }
        Ok(Self {
            groups,
            _reservation: reservation,
        })
    }

    pub(crate) async fn collect(
        scan: TableScan,
        context: &TaskContext,
        target_partitions: usize,
    ) -> Result<Self> {
        Self::new(context)
            .collect_into(scan, target_partitions)
            .await
    }

    pub(super) async fn collect_into(
        mut self,
        scan: TableScan,
        target_partitions: usize,
    ) -> Result<Self> {
        let mut stream = scan.plan_files().await.map_err(to_datafusion_error)?;
        let target_partitions = target_partitions.max(1);
        let mut partition = 0;
        while let Some(task) = stream.try_next().await.map_err(to_datafusion_error)? {
            // Reserve before retaining each descriptor, not after full discovery.
            self._reservation
                .try_grow(retained_size(&task).saturating_mul(2))?;
            // Grow only as files arrive, even for a very large partition target.
            if partition == self.groups.len() {
                self.groups.push(Vec::with_capacity(1));
            }
            self.groups[partition].push(task);
            partition = (partition + 1) % target_partitions;
        }
        // Empty discovery still has one partition. Explicit groups never pass
        // through here, so all-empty supplied partitions retain their shape.
        if self.groups.is_empty() {
            self.groups.push(Vec::new());
        }
        Ok(self)
    }
}

// Flatten alone is not ExactSizeIterator. Track the remaining length in the
// iterator, rather than caching a second mutable file count on PlannedFileScan.
struct PlannedFiles<'a> {
    inner: Flatten<Iter<'a, Vec<FileScanTask>>>,
    remaining: usize,
}

impl<'a> Iterator for PlannedFiles<'a> {
    type Item = &'a FileScanTask;

    fn next(&mut self) -> Option<Self::Item> {
        let task = self.inner.next()?;
        self.remaining -= 1;
        Some(task)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for PlannedFiles<'_> {}

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
