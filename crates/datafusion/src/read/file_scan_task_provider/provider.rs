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
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion::common::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::Partitioning;
use iceberg::table::Table;

/// Supplies the file tasks an [`IcebergDataSource`](crate::read::IcebergDataSource)
/// reads, split into output partitions.
///
/// The data source owns reading, limits and metrics; implementations only
/// decide which files each partition reads and when they are discovered.
/// Implementations are `Any`, so consumers such as codecs can downcast a
/// source's [`file_scan_task_provider`](crate::read::IcebergDataSource::file_scan_task_provider)
/// to a concrete type:
///
/// ```
/// # use std::any::Any;
/// # use iceberg_datafusion::read::{IcebergDataSource, EagerFileScanTaskProvider};
/// fn planned_files(source: &IcebergDataSource) -> Option<&EagerFileScanTaskProvider> {
///     (source.file_scan_task_provider() as &dyn Any).downcast_ref::<EagerFileScanTaskProvider>()
/// }
/// ```
pub trait FileScanTaskProvider: Any + Debug + Send + Sync {
    /// How the data source's output is partitioned, with at least one
    /// partition. Any expressions refer to columns of the data source's
    /// projected schema.
    fn partitioning(&self) -> Partitioning;

    /// Streams the file tasks of `partition`, which is below the
    /// [`Self::partitioning`] partition count.
    fn open(
        self: Arc<Self>,
        partition: usize,
        context: FileScanTaskProviderContext,
    ) -> Result<iceberg::scan::FileScanTaskStream>;

    /// Appends details to the data source's `EXPLAIN` line, each starting
    /// with `", "`. Writes nothing by default.
    fn fmt_as(&self, _f: &mut Formatter) -> std::fmt::Result {
        Ok(())
    }
}

pub struct FileScanTaskProviderContext<'a> {
    pub task_context: &'a Arc<TaskContext>,
    pub table: &'a Table,
}
