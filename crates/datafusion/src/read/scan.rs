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
use futures::TryStreamExt;
use iceberg::arrow::ArrowReader;
use iceberg::expr::Predicate;
use iceberg::scan::{FileScanTaskStream, TableScan};
use iceberg::table::Table;

use super::reader::ScanReader;
use crate::to_datafusion_error;

/// Discovers file tasks lazily while executing an Iceberg table scan.
/// Construct with [`super::IcebergDataSourceBuilder`].
#[derive(Clone)]
pub struct IcebergDataSource {
    pub(super) table: Table,
    pub(super) snapshot_id: Option<i64>,
    pub(super) schema: ArrowSchemaRef,
    pub(super) projection: Vec<String>,
    pub(super) predicate: Option<Predicate>,
    pub(super) fetch: Option<usize>,
    pub(super) reader: ArrowReader,
    pub(super) metrics: ExecutionPlanMetricsSet,
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
        let mut reader = ScanReader::new(self.reader.clone());
        reader.metrics = self.metrics.clone();
        reader.open(tasks, self.schema.clone(), self.fetch, partition)
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
        self.metrics.clone()
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

pub(super) fn build_table_scan(
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
