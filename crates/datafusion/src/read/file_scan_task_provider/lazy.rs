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
use std::fmt::Formatter;
use std::sync::Arc;

use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::common::runtime::SpawnedTask;
use datafusion::common::stats::Precision;
use datafusion::common::{
    DataFusionError, Result, Statistics, exec_datafusion_err, exec_err,
};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::Partitioning;
use datafusion::prelude::Expr;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt, stream};
use iceberg::expr::Predicate;
use iceberg::scan::{FileScanTask, FileScanTaskStream, TableScan};
use iceberg::spec::SnapshotRef;
use iceberg::table::Table;

use super::eager::EagerFileScanTaskProvider;
use super::provider::{FileScanTaskProvider, FileScanTaskProviderContext};
use crate::IcebergDataFusionConfig;
use crate::read::convert_filters_to_predicate;
use crate::to_datafusion_error;

/// Snapshot summary field from the spec's "Optional Snapshot Summary Fields".
const TOTAL_RECORDS: &str = "total-records";

/// Plans file tasks during execution. A single partition reads the planning
/// stream directly. Parallel scans use DataFusion sibling state: each reader
/// takes unclaimed files from the same Flume queue, even if it opens late.
/// Use `reset_plan_states` before executing the same parallel plan again.
#[derive(Debug, Clone)]
pub struct LazyFileScanTaskProvider {
    table: Table,
    scan: Arc<TableScan>,
    schema: SchemaRef,
    snapshot_id: Option<i64>,
    projection: Vec<String>,
    predicate: Option<Predicate>,
    target_partitions: usize,
}

/// Sibling state for one execution, initialized when its first reader is polled.
enum StreamState {
    NotStarted,
    Errored(Arc<DataFusionError>),
    Ready(PartitionedFanout),
}

struct PartitionedFanout {
    producer: Arc<SpawnedTask<()>>,
    feeds: Vec<flume::Receiver<Result<FileScanTask>>>,
}

impl PartitionedFanout {
    fn new(partitions: usize, mut tasks: FileScanTaskStream) -> Self {
        let (tx, rx) = flume::unbounded();
        let producer = SpawnedTask::spawn(async move {
            while let Some(task) = tasks.next().await {
                if tx.send(task.map_err(to_datafusion_error)).is_err() {
                    break;
                }
            }
        });
        Self {
            producer: Arc::new(producer),
            feeds: vec![rx; partitions],
        }
    }

    fn open(&mut self) -> Result<BoxStream<'static, Result<FileScanTask>>> {
        let feed = self.feeds.pop().ok_or_else(|| {
            exec_datafusion_err!("All file task streams have already been opened")
        })?;
        // Keep the producer alive even if the plan drops before its readers.
        let producer = Arc::clone(&self.producer);
        Ok(feed
            .into_stream()
            .inspect(move |_| {
                let _ = &producer;
            })
            .boxed())
    }
}

impl LazyFileScanTaskProvider {
    /// Collects the planned tasks into round-robin groups, using the
    /// context's target partition count. The result no longer retains the table.
    /// Empty discovery produces one empty group. Dropping this future drops the
    /// discovery stream; this method does not spawn a background collector.
    pub async fn into_eager(
        self,
        context: Arc<TaskContext>,
    ) -> Result<EagerFileScanTaskProvider> {
        let target = context.session_config().target_partitions();
        if target == 0 {
            return exec_err!(
                "Iceberg eager planning requires at least one target partition"
            );
        }
        let mut tasks = self.scan.plan_files().await.map_err(to_datafusion_error)?;
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
        EagerFileScanTaskProvider::try_new(groups)
    }

    /// Table retained for discovery, including its metadata and storage access.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Output schema after projection, matching the columns selected by discovery.
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Names of the columns selected during discovery, in output schema order.
    pub fn projection(&self) -> &[String] {
        &self.projection
    }

    /// Explicit snapshot selection, or `None` if the current snapshot was requested.
    /// Use [`Self::snapshot`] to inspect the snapshot resolved during planning.
    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    /// Snapshot resolved during planning, shared by discovery and statistics.
    /// Returns `None` only if the table has no current snapshot.
    pub fn snapshot(&self) -> Option<&SnapshotRef> {
        self.scan.snapshot()
    }

    /// Filter used to skip files, if any.
    pub fn predicate(&self) -> Option<&Predicate> {
        self.predicate.as_ref()
    }
}

/// Builds a task provider for the columns of a projected Arrow schema.
///
/// Configuration is supplied at construction. `iceberg.execution.eager_file_planning`
/// selects eager collection; otherwise discovery remains deferred until execution.
/// Pass the same projected schema to [`crate::read::IcebergDataSource::builder`].
///
/// ```
/// # use std::sync::Arc;
/// # use datafusion::{arrow::datatypes::SchemaRef, common::Result, execution::TaskContext, prelude::Expr};
/// # use iceberg::table::Table;
/// # use iceberg_datafusion::read::{FileScanTaskProvider, FileScanTaskProviderBuilder};
/// async fn plan_tasks(
///     context: Arc<TaskContext>,
///     table: Table,
///     projected_schema: SchemaRef,
///     filters: &[Expr],
/// ) -> Result<Arc<dyn FileScanTaskProvider>> {
///     FileScanTaskProviderBuilder::new(context, table, projected_schema)
///         .with_filters(filters)
///         .build()
///         .await
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FileScanTaskProviderBuilder {
    context: Arc<TaskContext>,
    table: Table,
    projected_schema: SchemaRef,
    snapshot_id: Option<i64>,
    predicate: Option<Predicate>,
}

impl FileScanTaskProviderBuilder {
    /// Selects the columns named in `projected_schema`, in schema order.
    pub fn new(
        context: Arc<TaskContext>,
        table: Table,
        projected_schema: SchemaRef,
    ) -> Self {
        Self {
            context,
            table,
            projected_schema,
            snapshot_id: None,
            predicate: None,
        }
    }

    /// Selects a snapshot, or the retained table's current snapshot for `None`.
    /// The supplied schema is not replaced with the snapshot's schema.
    pub fn with_snapshot_id(mut self, snapshot_id: Option<i64>) -> Self {
        self.snapshot_id = snapshot_id;
        self
    }

    /// Sets the predicate used to prune files and filter their rows.
    pub fn with_predicate(mut self, predicate: Option<Predicate>) -> Self {
        self.predicate = predicate;
        self
    }

    /// Pushes down supported parts of DataFusion filters. The caller must still
    /// apply the original filters, as unsupported parts are omitted.
    pub fn with_filters(self, filters: &[Expr]) -> Self {
        self.with_predicate(convert_filters_to_predicate(filters))
    }

    /// Resolves the snapshot, validates projection and builds the configured provider.
    /// Lazy mode performs no file discovery; eager mode awaits [`LazyFileScanTaskProvider::into_eager`].
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot does not exist, projection or predicate
    /// binding fails, or eager file discovery fails.
    pub async fn build(self) -> Result<Arc<dyn FileScanTaskProvider>> {
        let projection: Vec<String> = self
            .projected_schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let target_partitions = self.context.session_config().target_partitions();
        if target_partitions == 0 {
            return exec_err!(
                "Iceberg file planning requires at least one target partition"
            );
        }
        // TableScan resolves and retains the snapshot without discovering files.
        // Keep that same scan for execution and its snapshot for statistics.
        let mut scan = self.table.scan().select(&projection);
        if let Some(snapshot_id) = self.snapshot_id {
            scan = scan.snapshot_id(snapshot_id);
        }
        if let Some(predicate) = &self.predicate {
            scan = scan.with_filter(predicate.clone());
        }
        let scan = Arc::new(scan.build().map_err(to_datafusion_error)?);
        let discovery = LazyFileScanTaskProvider {
            table: self.table,
            scan,
            schema: self.projected_schema,
            projection,
            snapshot_id: self.snapshot_id,
            predicate: self.predicate,
            target_partitions,
        };
        if IcebergDataFusionConfig::from_config_or_default(self.context.session_config())
            .execution
            .eager_file_planning
        {
            Ok(Arc::new(discovery.into_eager(self.context).await?))
        } else {
            Ok(Arc::new(discovery))
        }
    }
}

impl FileScanTaskProvider for LazyFileScanTaskProvider {
    fn partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.target_partitions)
    }

    fn create_sibling_state(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        if self.target_partitions == 1 {
            return None;
        }
        Some(Arc::new(tokio::sync::Mutex::new(StreamState::NotStarted)))
    }

    fn open(
        self: Arc<Self>,
        partition: usize,
        context: FileScanTaskProviderContext,
    ) -> Result<BoxStream<'static, Result<FileScanTask>>> {
        if partition >= self.target_partitions {
            return exec_err!("Invalid Iceberg scan partition {partition}");
        }
        if self.target_partitions == 1 {
            return Ok(stream::once(async move { self.scan.plan_files().await })
                .try_flatten()
                .map_err(to_datafusion_error)
                .boxed());
        }
        let state = context
            .sibling_state
            .and_then(|state| state.downcast::<tokio::sync::Mutex<StreamState>>().ok())
            .ok_or_else(|| {
                exec_datafusion_err!(
                    "Parallel lazy scans require sibling state from DataSourceExec"
                )
            })?;
        let feed = async move {
            let mut state = state.lock().await;
            match &mut *state {
                StreamState::Ready(fanout) => return fanout.open(),
                StreamState::Errored(error) => {
                    return Err(DataFusionError::Shared(Arc::clone(error)));
                }
                StreamState::NotStarted => {}
            }
            match self.scan.plan_files().await {
                Ok(tasks) => {
                    let mut fanout =
                        PartitionedFanout::new(self.target_partitions, tasks);
                    let feed = fanout.open()?;
                    *state = StreamState::Ready(fanout);
                    Ok(feed)
                }
                Err(error) => {
                    let error = Arc::new(to_datafusion_error(error));
                    *state = StreamState::Errored(Arc::clone(&error));
                    Err(DataFusionError::Shared(error))
                }
            }
        };
        Ok(stream::once(feed).try_flatten().boxed())
    }

    /// Estimates the rows the scan returns before its predicate and fetch,
    /// from the snapshot summary.
    ///
    /// `total-records` counts rows before deletes, so the estimate is an upper
    /// bound. Deletes aren't subtracted, because the spec lets
    /// `total-position-deletes` count more rows than the deletes remove. A
    /// position delete file can stay live after the data files it references
    /// are removed. A deletion vector repeats the deletes of position delete
    /// files that can stay live beside it. An underestimate can make join
    /// selection collect a large table as the build side. Summary fields are
    /// optional, so a missing or invalid `total-records` leaves the estimate
    /// unknown rather than failing the query.
    fn partition_statistics(
        &self,
        partition: Option<usize>,
        schema: &Schema,
    ) -> Result<Statistics> {
        if partition.is_some_and(|partition| partition >= self.target_partitions) {
            return exec_err!("Invalid Iceberg scan partition {partition:?}");
        }
        let statistics = Statistics::new_unknown(schema);
        let Some(snapshot) = self.snapshot() else {
            return Ok(statistics.with_num_rows(Precision::Exact(0)));
        };
        // Before discovery, no individual reader has an assigned share of the files.
        if partition.is_some() && self.target_partitions > 1 {
            return Ok(statistics);
        }
        let num_rows = match snapshot
            .summary()
            .additional_properties
            .get(TOTAL_RECORDS)
            .and_then(|records| records.parse::<usize>().ok())
        {
            Some(records) => Precision::Inexact(records),
            None => Precision::Absent,
        };
        Ok(statistics.with_num_rows(num_rows))
    }

    fn fmt_as(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, ", provider=lazy")?;
        if let Some(predicate) = &self.predicate {
            write!(f, ", predicate={predicate}")?;
        }
        Ok(())
    }
}
