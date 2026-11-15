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

use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Statistics, exec_err, internal_err};
use datafusion::datasource::source::DataSource;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::{DisplayFormatType, Partitioning};
use datafusion::prelude::Expr;
use futures::TryStreamExt;
use iceberg::arrow::ArrowReader;
use iceberg::expr::Predicate;
use iceberg::io::FileIO;
use iceberg::scan::{FileScanTask, FileScanTaskStream, TableScan};
use iceberg::table::Table;

use super::eager::EagerIcebergDataSource;
use super::file_planning::PlannedFileScan;
use super::predicate::convert_filters_to_predicate;
use super::reader::ScanReader;
use crate::to_datafusion_error;

/// Discovers file tasks lazily while executing an Iceberg table scan.
#[derive(Clone)]
pub struct IcebergDataSource {
    /// A table in the catalog.
    table: Table,
    /// Snapshot of the table to scan.
    snapshot_id: Option<i64>,
    /// Schema returned by this data source after projection.
    schema: ArrowSchemaRef,
    /// Projection column names in the output schema.
    projection: Vec<String>,
    /// Filter to apply to the table scan
    predicate: Option<Predicate>,
    /// Optional limit on the number of rows to return
    fetch: Option<usize>,
    reader: ScanReader,
}

impl std::fmt::Debug for IcebergDataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergDataSource")
            .field("table", &self.table)
            .field("snapshot_id", &self.snapshot_id)
            .field("schema", &self.schema)
            .field("projection", &self.projection)
            .field("predicate", &self.predicate)
            .field("fetch", &self.fetch)
            .finish_non_exhaustive()
    }
}

impl IcebergDataSource {
    /// Starts building a source that scans `table` and returns the columns of
    /// `schema`, all of them unless [`IcebergDataSourceBuilder::with_projection`]
    /// selects some. Columns are read from Iceberg by name.
    pub fn builder(table: Table, schema: ArrowSchemaRef) -> IcebergDataSourceBuilder {
        IcebergDataSourceBuilder {
            input: ScanInput::Table(table),
            schema,
            projection: None,
            snapshot_id: None,
            predicate: None,
            fetch: None,
            reader: None,
        }
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    pub fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    pub fn projection(&self) -> &[String] {
        &self.projection
    }

    pub fn predicate(&self) -> Option<&Predicate> {
        self.predicate.as_ref()
    }

    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }
}

/// Builds an [`IcebergDataSource`]. Created by [`IcebergDataSource::builder`].
///
/// A table provider passes the table's schema and selects columns with
/// [`Self::with_projection`]. Call [`Self::build`] for lazy discovery or
/// [`Self::build_eager`] to return a source of already-assigned files.
///
/// To rebuild a source, such as in another process,
/// pass its [`IcebergDataSource::schema`], which is already projected, and no
/// projection. For already-planned files, reconstruct an eager source with
/// [`Self::from_planned_files`] instead of discovering files again.
#[derive(Clone)]
pub struct IcebergDataSourceBuilder {
    input: ScanInput,
    schema: ArrowSchemaRef,
    projection: Option<Vec<usize>>,
    snapshot_id: Option<i64>,
    predicate: Option<Predicate>,
    fetch: Option<usize>,
    reader: Option<ArrowReader>,
}

#[derive(Clone)]
enum ScanInput {
    Table(Table),
    Planned {
        groups: Vec<Vec<FileScanTask>>,
        file_io: FileIO,
    },
}

impl std::fmt::Debug for IcebergDataSourceBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergDataSourceBuilder")
            .field("schema", &self.schema)
            .field("projection", &self.projection)
            .field("snapshot_id", &self.snapshot_id)
            .field("predicate", &self.predicate)
            .field("fetch", &self.fetch)
            .finish_non_exhaustive()
    }
}

impl IcebergDataSourceBuilder {
    /// Build from already-planned files without a table, catalog or discovery I/O.
    /// Each group is one local output partition; empty groups and supplied order
    /// are preserved. `schema` must already be projected to match the tasks.
    /// `reader` must use the supplied FileIO and process-local runtime/settings.
    /// Reader clones share their delete loader; their settings are not overwritten.
    ///
    /// Finish with [`Self::build_planned`], which reserves descriptor memory in the
    /// supplied query context. Discovery-only projection, snapshot and predicate
    /// settings are rejected rather than silently ignored. Statistics are unknown.
    /// Use [`EagerIcebergDataSource::with_planned_files`] to replace the assignment
    /// of an existing source instead.
    ///
    /// ```
    /// use datafusion::arrow::datatypes::SchemaRef;
    /// use datafusion::common::Result;
    /// use datafusion::execution::TaskContext;
    /// use iceberg::arrow::ArrowReader;
    /// use iceberg::io::FileIO;
    /// use iceberg::scan::FileScanTask;
    /// use iceberg_datafusion::read::{EagerIcebergDataSource, IcebergDataSourceBuilder};
    ///
    /// fn restore(
    ///     schema: SchemaRef,
    ///     groups: Vec<Vec<FileScanTask>>,
    ///     file_io: FileIO,
    ///     local_reader: ArrowReader,
    ///     fetch: Option<usize>,
    ///     context: &TaskContext,
    /// ) -> Result<EagerIcebergDataSource> {
    ///     IcebergDataSourceBuilder::from_planned_files(schema, groups, file_io, local_reader)
    ///         .with_fetch(fetch)
    ///         .build_planned(context)
    /// }
    /// ```
    pub fn from_planned_files(
        schema: ArrowSchemaRef,
        groups: Vec<Vec<FileScanTask>>,
        file_io: FileIO,
        reader: ArrowReader,
    ) -> Self {
        Self {
            input: ScanInput::Planned { groups, file_io },
            schema,
            projection: None,
            snapshot_id: None,
            predicate: None,
            fetch: None,
            reader: Some(reader),
        }
    }

    /// Finish already-planned input synchronously, including in a worker codec.
    /// Does not discover files or perform I/O. Supplied groups are preserved exactly.
    ///
    /// # Errors
    /// Returns an error for table input, discovery-only settings, zero groups or
    /// insufficient query memory. Empty individual groups are valid.
    pub fn build_planned(self, context: &TaskContext) -> Result<EagerIcebergDataSource> {
        let ScanInput::Planned { groups, file_io } = self.input else {
            return exec_err!("build_planned requires already-planned Iceberg files");
        };
        if self.projection.is_some()
            || self.snapshot_id.is_some()
            || self.predicate.is_some()
        {
            return exec_err!(
                "Planned Iceberg files cannot use discovery projection, snapshot or predicate settings"
            );
        }
        EagerIcebergDataSource::from_file_groups(
            self.schema,
            groups,
            file_io,
            self.reader.expect("planned input always supplies a reader"),
            self.fetch,
            context,
        )
    }

    /// Use an explicitly configured reader, preserving its settings in both modes.
    /// Its file access must match the table or planned input's FileIO.
    pub fn with_reader(mut self, reader: ArrowReader) -> Self {
        self.reader = Some(reader);
        self
    }

    /// Returns only the columns of the schema at these indices, or all of them
    /// if `None` (the default). Indices are into the schema passed to
    /// [`IcebergDataSource::builder`].
    pub fn with_projection(mut self, projection: Option<&Vec<usize>>) -> Self {
        self.projection = projection.cloned();
        self
    }

    /// Scans this snapshot, or the table's current one if `None` (the default).
    pub fn with_snapshot_id(mut self, snapshot_id: Option<i64>) -> Self {
        self.snapshot_id = snapshot_id;
        self
    }

    /// Filters the scan with `predicate`, or not at all if `None` (the default).
    pub fn with_predicate(mut self, predicate: Option<Predicate>) -> Self {
        self.predicate = predicate;
        self
    }

    /// Filters the scan with the parts of `filters` that convert to an Iceberg
    /// predicate. The rest are dropped, so the scan may return rows that
    /// `filters` reject.
    pub fn with_filters(self, filters: &[Expr]) -> Self {
        self.with_predicate(convert_filters_to_predicate(filters))
    }

    /// Stops after `fetch` rows, or reads every row if `None` (the default).
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        self
    }

    /// Discover files and assign them round-robin to local output partitions.
    /// Already-planned input delegates to [`Self::build_planned`] without discovery.
    /// For table input, partition count is bounded by selected files (at least one).
    /// Table discovery uses the caller's I/O-capable runtime and the query memory pool.
    /// Dropping this future releases the retained tasks and reservation.
    /// Without a reader override, multiple local partitions use one in-flight
    /// file per partition; reassignment preserves the reader's settings.
    ///
    /// # Errors
    /// Returns an error for invalid projection, failed discovery, or insufficient
    /// query memory for the retained descriptors.
    pub async fn build_eager(
        self,
        context: Arc<TaskContext>,
    ) -> Result<EagerIcebergDataSource> {
        if matches!(self.input, ScanInput::Planned { .. }) {
            return self.build_planned(&context);
        }
        let schema = self.projected_schema()?;
        let ScanInput::Table(table) = self.input else {
            unreachable!()
        };
        let projection: Vec<_> = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let scan = build_table_scan(
            &table,
            self.snapshot_id,
            Some(&projection),
            self.predicate.as_ref(),
        )?;
        let files = PlannedFileScan::collect(scan, &context).await?;
        let partitions = files
            .tasks()
            .len()
            .max(1)
            .min(context.session_config().target_partitions().max(1));
        let reader = self.reader.unwrap_or_else(|| {
            let reader = table.reader_builder();
            let reader = if partitions > 1 {
                reader.with_data_file_concurrency_limit(1)
            } else {
                reader
            };
            reader.build()
        });
        let mut groups = vec![Vec::new(); partitions];
        for index in 0..files.tasks().len() {
            groups[index % partitions].push(index);
        }
        Ok(EagerIcebergDataSource::from_planned_files(
            schema,
            files,
            groups,
            table.file_io().clone(),
            reader,
            self.fetch,
        ))
    }

    /// Build a lazy source and its shared reader without performing file-discovery I/O.
    ///
    /// # Errors
    /// Returns an error if the projection is invalid or the input is already planned.
    pub fn build(self) -> Result<IcebergDataSource> {
        let schema = self.projected_schema()?;
        let projection = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let ScanInput::Table(table) = self.input else {
            return exec_err!(
                "Planned Iceberg files require build_planned or build_eager"
            );
        };
        let reader = ScanReader::new(
            self.reader
                .unwrap_or_else(|| table.reader_builder().build()),
        );
        Ok(IcebergDataSource {
            table,
            snapshot_id: self.snapshot_id,
            schema,
            projection,
            predicate: self.predicate,
            fetch: self.fetch,
            reader,
        })
    }

    fn projected_schema(&self) -> Result<ArrowSchemaRef> {
        match &self.projection {
            None => Ok(self.schema.clone()),
            Some(projection) => Ok(Arc::new(self.schema.project(projection)?)),
        }
    }
}

impl DataSource for IcebergDataSource {
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
        if partition != 0 {
            return internal_err!("Iceberg scan partition {partition} does not exist");
        }
        let scan = build_table_scan(
            &self.table,
            self.snapshot_id,
            Some(&self.projection),
            self.predicate.as_ref(),
        )?;
        let tasks: FileScanTaskStream = Box::pin(
            futures::stream::once(async move { scan.plan_files().await }).try_flatten(),
        );
        self.reader
            .open(tasks, self.schema.clone(), self.fetch, partition)
    }

    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "format=iceberg",)?;
        write!(f, ", projection=[{}]", self.projection.join(", "))?;
        if let Some(predicate) = &self.predicate {
            write!(f, ", predicate={predicate}")?;
        }
        if let Some(fetch) = self.fetch {
            write!(f, ", fetch={fetch}")?;
        }
        Ok(())
    }

    fn output_partitioning(&self) -> Partitioning {
        // TODO: Partitioned lazy discovery: https://github.com/apache/datafusion-iceberg/issues/84
        Partitioning::UnknownPartitioning(1)
    }

    fn metrics(&self) -> ExecutionPlanMetricsSet {
        self.reader.metrics.clone()
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(self.schema.clone())
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
}

fn build_table_scan(
    table: &Table,
    snapshot_id: Option<i64>,
    column_names: Option<&[String]>,
    predicate: Option<&Predicate>,
) -> Result<TableScan> {
    let builder = match snapshot_id {
        Some(snapshot_id) => table.scan().snapshot_id(snapshot_id),
        None => table.scan(),
    };
    let mut builder = match column_names {
        Some(columns) => builder.select(columns),
        None => builder.select_all(),
    };
    if let Some(predicate) = predicate {
        builder = builder.with_filter(predicate.clone());
    }
    builder.build().map_err(to_datafusion_error)
}
