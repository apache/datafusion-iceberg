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

use datafusion::arrow::datatypes::Schema;
use datafusion::common::{Result, Statistics};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::Partitioning;
use futures::stream::BoxStream;
use iceberg::scan::FileScanTask;

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
    ) -> Result<BoxStream<'static, Result<FileScanTask>>>;

    /// Statistics before the source's fetch, without discovering or reading files.
    /// `None` requests the whole assignment; `Some(i)` requests partition `i`,
    /// which must be below the partition count. Whole-assignment estimates must
    /// not be repeated as estimates of individual partitions.
    ///
    /// `schema` is the source's projected output schema. Return one column statistic
    /// per field, in that order, leaving unknown values as `Precision::Absent`.
    /// Byte sizes describe output data, not compressed files. Use inexact estimates
    /// when predicates or deletes may remove rows. The source fills missing byte
    /// estimates where possible and applies its fetch limit.
    ///
    /// The default provides no estimates. Errors from providers are propagated by
    /// the source.
    fn partition_statistics(
        &self,
        _partition: Option<usize>,
        schema: &Schema,
    ) -> Result<Statistics> {
        Ok(Statistics::new_unknown(schema))
    }

    /// Appends details to the data source's `EXPLAIN` line, each starting
    /// with `", "`. Writes nothing by default.
    fn fmt_as(&self, _f: &mut Formatter) -> std::fmt::Result {
        Ok(())
    }
}

/// Execution context for a task stream. Discovery-specific inputs, such as a
/// table, belong to the provider rather than the data source.
pub struct FileScanTaskProviderContext<'a> {
    pub task_context: &'a Arc<TaskContext>,
}
