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
use std::fmt::Formatter;
use std::sync::Arc;

use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::common::stats::Precision;
use datafusion::common::{Result, Statistics, exec_err};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::Partitioning;
use datafusion::prelude::Expr;
use futures::TryStreamExt;
use iceberg::expr::Predicate;
use iceberg::scan::TableScan;
use iceberg::spec::SnapshotRef;
use iceberg::table::Table;

use super::eager::EagerFileScanTaskProvider;
use super::provider::{FileScanTaskProvider, FileScanTaskProviderContext};
use crate::IcebergDataFusionConfig;
use crate::read::convert_filters_to_predicate;
use crate::to_datafusion_error;

/// Snapshot summary field from the spec's "Optional Snapshot Summary Fields".
const TOTAL_RECORDS: &str = "total-records";

/// Discovers files while executing, in a single partition: the files of a
/// snapshot that may match a predicate.
#[derive(Debug, Clone)]
pub struct LazyFileScanTaskProvider {
    table: Table,
    scan: Arc<TableScan>,
    schema: SchemaRef,
    snapshot_id: Option<i64>,
    projection: Vec<String>,
    predicate: Option<Predicate>,
}

impl LazyFileScanTaskProvider {
    /// Consumes discovery and retains its tasks in round-robin groups, using the
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
        let mut tasks = Arc::new(self).open(
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

/// Builds a task provider and its projected schema from a table scan request.
///
/// Configuration is supplied at construction. `iceberg.execution.eager_file_planning`
/// selects eager collection; otherwise discovery remains deferred until execution.
/// The projected schema is returned alongside the provider so callers never need
/// to recover it by downcasting an eager or lazy implementation. Reader settings
/// and fetch belong to the data-source builder.
///
/// ```
/// # use std::sync::Arc;
/// # use datafusion::{arrow::datatypes::SchemaRef, common::Result, execution::TaskContext, prelude::Expr};
/// # use iceberg::table::Table;
/// # use iceberg_datafusion::read::{FileScanTaskProvider, FileScanTaskProviderBuilder};
/// async fn plan_tasks(
///     context: Arc<TaskContext>,
///     table: Table,
///     schema: SchemaRef,
///     projection: Option<&Vec<usize>>,
///     filters: &[Expr],
/// ) -> Result<(SchemaRef, Arc<dyn FileScanTaskProvider>)> {
///     FileScanTaskProviderBuilder::new(context, table, schema)
///         .with_projection(projection)
///         .with_filters(filters)
///         .build()
///         .await
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FileScanTaskProviderBuilder {
    context: Arc<TaskContext>,
    table: Table,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    snapshot_id: Option<i64>,
    predicate: Option<Predicate>,
}

impl FileScanTaskProviderBuilder {
    /// Starts with the planning context and the schema advertised by the table
    /// provider. Projection indices refer to this schema, not the table's current
    /// schema. Pass an already-projected schema and omit projection to retain it.
    pub fn new(context: Arc<TaskContext>, table: Table, schema: SchemaRef) -> Self {
        Self {
            context,
            table,
            schema,
            projection: None,
            snapshot_id: None,
            predicate: None,
        }
    }

    /// Selects columns from the supplied schema in this order, or all its columns
    /// for `None`. Indices are validated when [`Self::build`] is called.
    pub fn with_projection(mut self, projection: Option<&Vec<usize>>) -> Self {
        self.projection = projection.cloned();
        self
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
    /// The returned schema matches the columns selected by both implementations.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot does not exist, projection or predicate
    /// binding fails, or eager file discovery fails.
    pub async fn build(self) -> Result<(SchemaRef, Arc<dyn FileScanTaskProvider>)> {
        let schema = match self.projection {
            Some(projection) => Arc::new(self.schema.project(&projection)?),
            None => self.schema,
        };
        let projection: Vec<String> = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
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
            schema: Arc::clone(&schema),
            projection,
            snapshot_id: self.snapshot_id,
            predicate: self.predicate,
        };
        let provider: Arc<dyn FileScanTaskProvider> =
            if IcebergDataFusionConfig::from_config_or_default(
                self.context.session_config(),
            )
            .execution
            .eager_file_planning
            {
                Arc::new(discovery.into_eager(self.context).await?)
            } else {
                Arc::new(discovery)
            };
        Ok((schema, provider))
    }
}

impl FileScanTaskProvider for LazyFileScanTaskProvider {
    fn partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn open(
        self: Arc<Self>,
        partition: usize,
        _ctx: FileScanTaskProviderContext,
    ) -> Result<iceberg::scan::FileScanTaskStream> {
        if partition != 0 {
            return exec_err!("Invalid Iceberg scan partition {partition}");
        }
        let scan = Arc::clone(&self.scan);
        // Files are discovered when the stream is first polled, not here.
        Ok(Box::pin(
            futures::stream::once(async move { scan.plan_files().await }).try_flatten(),
        ))
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
        if partition.is_some_and(|partition| partition != 0) {
            return exec_err!("Invalid Iceberg scan partition {partition:?}");
        }
        let statistics = Statistics::new_unknown(schema);
        let Some(snapshot) = self.snapshot() else {
            return Ok(statistics.with_num_rows(Precision::Exact(0)));
        };
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
