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

use std::pin::Pin;
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::common::Statistics;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::datasource::source::DataSource;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayFormatType, Partitioning};
use datafusion::prelude::Expr;
use futures::{Stream, TryStreamExt};
use iceberg::expr::Predicate;
use iceberg::table::Table;

use super::predicate::convert_filters_to_predicate;
use crate::to_datafusion_error;

/// Manages the scanning process of an Iceberg [`Table`], encapsulating the
/// necessary details and computed properties required for execution planning.
#[derive(Debug, Clone)]
pub struct IcebergDataSource {
    /// A table in the catalog.
    table: Table,
    /// Snapshot of the table to scan.
    snapshot_id: Option<i64>,
    /// Schema returned by this data source after projection.
    schema: ArrowSchemaRef,
    /// Projection column names in the output schema.
    projection: Vec<String>,
    /// Filters to apply to the table scan
    predicates: Option<Predicate>,
    /// Optional limit on the number of rows to return
    fetch: Option<usize>,
}

impl IcebergDataSource {
    /// Creates a new [`IcebergDataSource`] object.
    pub(crate) fn new(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Self> {
        let output_schema = match projection {
            None => schema,
            Some(projection) => Arc::new(schema.project(projection)?),
        };
        Ok(Self::new_with_predicate(
            table,
            snapshot_id,
            output_schema,
            convert_filters_to_predicate(filters),
            limit,
        ))
    }

    /// Rebuilds a source from the parts needed to scan an Iceberg table.
    ///
    /// The supplied schema determines both the source's output schema and the
    /// columns read from Iceberg. Callers can recreate a source from its
    /// accessors after transporting those values to another process.
    pub fn new_with_predicate(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        predicates: Option<Predicate>,
        fetch: Option<usize>,
    ) -> Self {
        let projection = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        Self {
            table,
            snapshot_id,
            schema,
            projection,
            predicates,
            fetch,
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

    pub fn predicates(&self) -> Option<&Predicate> {
        self.predicates.as_ref()
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
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let fut = get_batch_stream(
            self.table.clone(),
            self.snapshot_id,
            Some(self.projection.clone()),
            self.predicates.clone(),
        );
        let stream = futures::stream::once(fut).try_flatten();

        // Apply the fetch if specified.
        let limited_stream: Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>> =
            if let Some(limit) = self.fetch {
                let mut remaining = limit;
                Box::pin(stream.try_filter_map(move |batch| {
                    futures::future::ready(if remaining == 0 {
                        Ok(None)
                    } else if batch.num_rows() <= remaining {
                        remaining -= batch.num_rows();
                        Ok(Some(batch))
                    } else {
                        let limited_batch = batch.slice(0, remaining);
                        remaining = 0;
                        Ok(Some(limited_batch))
                    })
                }))
            } else {
                Box::pin(stream)
            };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            limited_stream,
        )))
    }

    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "format=iceberg",)?;
        write!(f, ", projection=[{}]", self.projection.join(", "))?;
        if let Some(predicate) = &self.predicates {
            write!(f, ", predicate={predicate}")?;
        }
        if let Some(fetch) = self.fetch {
            write!(f, ", fetch={fetch}")?;
        }
        Ok(())
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
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

/// Asynchronously retrieves a stream of [`RecordBatch`] instances
/// from a given table.
///
/// This function initializes a [`TableScan`], builds it,
/// and then converts it into a stream of Arrow [`RecordBatch`]es.
async fn get_batch_stream(
    table: Table,
    snapshot_id: Option<i64>,
    column_names: Option<Vec<String>>,
    predicates: Option<Predicate>,
) -> Result<Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>> {
    let scan_builder = match snapshot_id {
        Some(snapshot_id) => table.scan().snapshot_id(snapshot_id),
        None => table.scan(),
    };

    let mut scan_builder = match column_names {
        Some(column_names) => scan_builder.select(column_names),
        None => scan_builder.select_all(),
    };
    if let Some(pred) = predicates {
        scan_builder = scan_builder.with_filter(pred);
    }
    let table_scan = scan_builder.build().map_err(to_datafusion_error)?;

    let stream = table_scan
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?
        .map_err(to_datafusion_error);
    Ok(Box::pin(stream))
}
