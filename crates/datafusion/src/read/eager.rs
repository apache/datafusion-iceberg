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
use iceberg::io::FileIO;
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
    file_io: FileIO,
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
        file_io: FileIO,
        reader: ArrowReader,
        fetch: Option<usize>,
    ) -> Self {
        Self {
            schema,
            fetch,
            file_io,
            files: Arc::new(files),
            partitions: groups.into(),
            reader: ScanReader::new(reader),
        }
    }

    pub(crate) fn from_file_groups(
        schema: SchemaRef,
        groups: Vec<Vec<FileScanTask>>,
        file_io: FileIO,
        reader: ArrowReader,
        fetch: Option<usize>,
        context: &TaskContext,
    ) -> Result<Self> {
        let (tasks, partitions) = Self::flatten_groups(groups)?;
        Ok(Self::from_planned_files(
            schema,
            PlannedFileScan::from_tasks(tasks, context)?,
            partitions,
            file_io,
            reader,
            fetch,
        ))
    }

    fn flatten_groups(
        groups: Vec<Vec<FileScanTask>>,
    ) -> Result<(Vec<FileScanTask>, Vec<Vec<usize>>)> {
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
        Ok((tasks, partitions))
    }

    /// Output schema, including the scan's projection.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// File access associated with this source's reader, for reconstruction.
    pub fn file_io(&self) -> &FileIO {
        &self.file_io
    }

    /// Borrow all retained file tasks. Iteration does not promise row ordering.
    /// Use [`Self::planned_partition_files`] to inspect partition membership.
    pub fn planned_files(&self) -> impl ExactSizeIterator<Item = &FileScanTask> {
        self.files.tasks().iter()
    }

    /// Borrow the files that execution will read for one local partition.
    ///
    /// # Errors
    /// Returns an error for an out-of-bounds partition.
    pub fn planned_partition_files(
        &self,
        partition: usize,
    ) -> Result<impl ExactSizeIterator<Item = &FileScanTask>> {
        let Some(indices) = self.partitions.get(partition) else {
            return exec_err!("Invalid Iceberg scan partition {partition}");
        };
        Ok(indices.iter().map(|&index| &self.files.tasks()[index]))
    }

    /// Replace the planned files, preserving the supplied local partitions and order.
    /// Empty partitions are valid. Repeated tasks are not deduplicated;
    /// the caller is responsible for avoiding unwanted duplicate work and ensuring
    /// tasks match this source's projected schema and file access.
    ///
    /// Preserves fetch and reader settings, including the shared delete loader.
    /// Returns fresh metrics and unknown statistics without discovery or file I/O.
    /// The owned descriptors have an independent reservation in the same query pool;
    /// cloning tasks from this source therefore requires additional query memory.
    ///
    /// # Errors
    /// Returns an error for zero groups or insufficient query memory.
    pub fn with_planned_files(&self, groups: Vec<Vec<FileScanTask>>) -> Result<Self> {
        let (tasks, partitions) = Self::flatten_groups(groups)?;
        let files = self.files.with_tasks(tasks)?;
        let mut source = self.clone();
        source.files = Arc::new(files);
        source.partitions = partitions.into();
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
        let count = self.planned_partition_files(partition)?.len();
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
