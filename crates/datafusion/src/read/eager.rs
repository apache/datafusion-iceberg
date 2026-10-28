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

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Statistics, exec_err};
use datafusion::datasource::source::DataSource;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::DisplayFormatType;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::StreamExt;
use iceberg::arrow::ArrowReader;
use iceberg::scan::FileScanTask;

use super::file_planning::PlannedFileScan;
use super::reader::ScanReader;

/// An Iceberg scan whose complete file assignment is available before execution.
/// Reads already-selected file groups without retaining a table or discovery request.
/// Consumers can inspect and regroup these tasks, or reconstruct a source on another
/// process using a locally configured reader.
#[derive(Clone)]
pub struct EagerIcebergDataSource {
    schema: SchemaRef,
    fetch: Option<usize>,
    reader: ScanReader,
    files: Arc<PlannedFileScan>,
    partitions: Arc<[Vec<usize>]>,
}
impl std::fmt::Debug for EagerIcebergDataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EagerIcebergDataSource")
            .field("schema", &self.schema)
            .field("fetch", &self.fetch)
            .field("partitions", &self.partitions)
            .finish_non_exhaustive()
    }
}
impl EagerIcebergDataSource {
    pub(crate) fn from_planned_files(
        schema: SchemaRef,
        files: PlannedFileScan,
        groups: Vec<Vec<usize>>,
        reader: ArrowReader,
        fetch: Option<usize>,
    ) -> Self {
        Self {
            schema,
            fetch,
            files: Arc::new(files),
            partitions: groups.into(),
            reader: ScanReader::new(reader),
        }
    }

    /// Construct a scan from already selected files grouped by local output partition.
    /// Preserves group order without discovering, pruning, or assigning files.
    /// The caller supplies a process-local reader configured for these files. Reader
    /// clones share its delete cache across partitions; regrouping preserves its
    /// settings, including file concurrency. The schema must match the tasks' projection.
    /// Descriptors are charged to `context`'s memory pool; statistics are unknown.
    ///
    /// # Errors
    /// Returns an error if there are no partition groups or memory reservation fails.
    /// Empty groups are valid, including for an empty scan.
    pub fn from_file_groups(
        schema: SchemaRef,
        groups: Vec<Vec<FileScanTask>>,
        reader: ArrowReader,
        fetch: Option<usize>,
        context: &TaskContext,
    ) -> Result<Self> {
        if groups.is_empty() {
            return exec_err!("Iceberg scan requires at least one file group");
        }
        let mut tasks = Vec::new();
        let mut partitions = Vec::with_capacity(groups.len());
        for group in groups {
            let start = tasks.len();
            tasks.extend(group);
            partitions.push((start..tasks.len()).collect());
        }
        Ok(Self {
            schema,
            fetch,
            reader: ScanReader::new(reader),
            files: Arc::new(PlannedFileScan::from_tasks(tasks, context)?),
            partitions: partitions.into(),
        })
    }

    /// Output schema, including the scan's projection.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Borrow this scan's files in retained-file order, independent of partition groups.
    /// Positions in this iterator are the inputs to [`Self::with_file_groups`].
    /// This sorts file indices, without cloning descriptors or performing I/O.
    /// File iteration order does not promise row ordering.
    pub fn planned_files(&self) -> impl ExactSizeIterator<Item = &FileScanTask> {
        self.file_indices()
            .into_iter()
            .map(|index| &self.files.tasks()[index])
    }

    fn file_indices(&self) -> Vec<usize> {
        let mut indices: Vec<_> = self.partitions.iter().flatten().copied().collect();
        indices.sort_unstable();
        indices
    }

    /// Borrow the files that execution will read for one local partition.
    ///
    /// # Errors
    /// Returns an error for an out-of-bounds partition.
    pub fn planned_partition_tasks(
        &self,
        partition: usize,
    ) -> Result<impl ExactSizeIterator<Item = &FileScanTask>> {
        let Some(indices) = self.partitions.get(partition) else {
            return exec_err!("Invalid Iceberg scan partition {partition}");
        };
        Ok(indices.iter().map(|&index| &self.files.tasks()[index]))
    }

    /// Derive a scan using positions in this source's [`Self::planned_files`] iterator.
    /// Each group is one local output partition; supplied order is preserved.
    /// Files may be omitted, but each position may appear at most once.
    ///
    /// The result shares descriptors and their memory reservation with this source.
    /// It has fresh metrics and unknown statistics because its file set may differ.
    /// No file discovery, assignment policy, or file I/O is performed.
    ///
    /// # Errors
    /// Returns an error for zero groups, out-of-bounds positions, or repeated positions.
    pub fn with_file_groups(&self, mut groups: Vec<Vec<usize>>) -> Result<Self> {
        if groups.is_empty() {
            return exec_err!("Iceberg scan requires at least one file group");
        }
        let indices = self.file_indices();
        let mut used = vec![false; indices.len()];
        for position in groups.iter_mut().flatten() {
            let Some(&index) = indices.get(*position) else {
                return exec_err!("Invalid Iceberg file position {}", *position);
            };
            if std::mem::replace(&mut used[*position], true) {
                return exec_err!("Repeated Iceberg file position {}", *position);
            }
            *position = index;
        }
        let mut source = self.clone();
        source.partitions = groups.into();
        source.reader.metrics = Default::default();
        Ok(source)
    }
}
impl DataSource for EagerIcebergDataSource {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let count = self.planned_partition_tasks(partition)?.len();
        let source = self.clone();
        let tasks = futures::stream::iter(0..count)
            .map(move |offset| {
                Ok(source.files.tasks()[source.partitions[partition][offset]].clone())
            })
            .boxed();
        self.reader
            .open(tasks, Arc::clone(&self.schema), self.fetch, partition)
    }
    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(
            f,
            "format=iceberg, projection=[{}], planned_files={}",
            self.schema
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>()
                .join(", "),
            self.partitions.iter().map(Vec::len).sum::<usize>()
        )?;
        if let Some(fetch) = self.fetch {
            write!(f, ", fetch={fetch}")?;
        }
        Ok(())
    }
    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.partitions.len())
    }
    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(Arc::clone(&self.schema))
    }
    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        Ok(Arc::new(Statistics::new_unknown(&self.schema)))
    }

    fn with_fetch(&self, fetch: Option<usize>) -> Option<Arc<dyn DataSource>> {
        let mut source = self.clone();
        source.fetch = fetch;
        Some(Arc::new(source))
    }
    fn fetch(&self) -> Option<usize> {
        self.fetch
    }
    fn try_swapping_with_projection(
        &self,
        _projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn DataSource>>> {
        Ok(None)
    }
    fn metrics(&self) -> ExecutionPlanMetricsSet {
        self.reader.metrics.clone()
    }
}
