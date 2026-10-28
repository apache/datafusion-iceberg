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
use datafusion::common::{Statistics, internal_err};
use datafusion::datasource::source::DataSource;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::{DisplayFormatType, Partitioning};
use datafusion::prelude::Expr;
use futures::TryStreamExt;
use iceberg::expr::Predicate;
use iceberg::scan::{FileScanTaskStream, TableScan};
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
            table,
            schema,
            projection: None,
            snapshot_id: None,
            predicate: None,
            fetch: None,
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
/// [`EagerIcebergDataSource::from_file_groups`] instead of discovering files again.
#[derive(Debug, Clone)]
pub struct IcebergDataSourceBuilder {
    table: Table,
    schema: ArrowSchemaRef,
    projection: Option<Vec<usize>>,
    snapshot_id: Option<i64>,
    predicate: Option<Predicate>,
    fetch: Option<usize>,
}

impl IcebergDataSourceBuilder {
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
    /// The partition count is bounded by the selected file count (at least one).
    /// Planning uses the caller's I/O-capable runtime and the query memory pool.
    /// Dropping this future releases the retained tasks and reservation.
    /// Multiple local partitions use one in-flight file per partition; regrouping
    /// the returned source preserves that reader setting.
    ///
    /// # Errors
    /// Returns an error for invalid projection, failed discovery, or insufficient
    /// query memory for the retained descriptors.
    pub async fn build_eager(
        self,
        context: Arc<TaskContext>,
    ) -> Result<EagerIcebergDataSource> {
        let schema = self.projected_schema()?;
        let projection: Vec<_> = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let scan = build_table_scan(
            &self.table,
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
        let reader = self.table.reader_builder();
        let reader = if partitions > 1 {
            reader.with_data_file_concurrency_limit(1)
        } else {
            reader
        };
        let mut groups = vec![Vec::new(); partitions];
        for index in 0..files.tasks().len() {
            groups[index % partitions].push(index);
        }
        Ok(EagerIcebergDataSource::from_planned_files(
            schema,
            files,
            groups,
            reader.build(),
            self.fetch,
        ))
    }

    /// Build a lazy source and its shared reader without performing file-discovery I/O.
    ///
    /// # Errors
    /// Returns an error if the projection is invalid.
    pub fn build(self) -> Result<IcebergDataSource> {
        let schema = self.projected_schema()?;
        let projection = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let reader = ScanReader::new(self.table.reader_builder().build());
        Ok(IcebergDataSource {
            table: self.table,
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
        // Partitioned lazy discovery: https://github.com/apache/datafusion-iceberg/issues/84
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
