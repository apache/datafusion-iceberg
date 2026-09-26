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
use std::vec;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties};
use datafusion::prelude::Expr;
use futures::{Stream, TryStreamExt};
use iceberg::expr::Predicate;
use iceberg::table::Table;

use super::expr_to_predicate::convert_filters_to_predicate;
use crate::to_datafusion_error;

/// Manages the scanning process of an Iceberg [`Table`], encapsulating the
/// necessary details and computed properties required for execution planning.
#[derive(Debug)]
pub struct IcebergTableScan {
    /// A table in the catalog.
    table: Table,
    /// Snapshot of the table to scan.
    snapshot_id: Option<i64>,
    /// Stores certain, often expensive to compute,
    /// plan properties used in query optimization.
    plan_properties: Arc<PlanProperties>,
    /// Projection column names, None means all columns
    projection: Option<Vec<String>>,
    /// Filters to apply to the table scan
    predicates: Option<Predicate>,
    /// Optional limit on the number of rows to return
    limit: Option<usize>,
}

impl IcebergTableScan {
    /// Creates a new [`IcebergTableScan`] object.
    pub(crate) fn new(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Self> {
        Self::new_with_predicate(
            table,
            snapshot_id,
            schema,
            projection.map(Vec::as_slice),
            convert_filters_to_predicate(filters),
            limit,
        )
    }

    /// Creates a scan of `table` from an already-converted Iceberg
    /// [`Predicate`] rather than DataFusion filters, for rebuilding a scan from
    /// its parts, such as after sending them to another process. A predicate
    /// cannot be converted back to the filters it came from.
    ///
    /// The arguments mean what the matching accessors return:
    ///
    /// - `snapshot_id`: the snapshot to read, or `None` for the table's current
    ///   snapshot.
    /// - `schema`: the Arrow schema of the table the scan reads, as its
    ///   provider reports it.
    /// - `projection`: indices into `schema` of the columns to read, or `None`
    ///   for all. The columns are read from the table by name.
    /// - `predicate`: pushed down to Iceberg to skip data files and rows. The
    ///   table providers report their filters as
    ///   [`Inexact`](datafusion::logical_expr::TableProviderFilterPushDown::Inexact),
    ///   so DataFusion still applies them above the scan.
    ///
    /// # Errors
    ///
    /// Returns an error if `projection` holds an index outside `schema`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// use datafusion::catalog::TableProvider;
    /// use datafusion::physical_plan::ExecutionPlan;
    /// use datafusion::prelude::{SessionContext, col, lit};
    /// use datafusion_iceberg::IcebergStaticTableProvider;
    /// use datafusion_iceberg::physical_plan::IcebergTableScan;
    /// use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    /// use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    /// use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
    ///
    /// # tokio::runtime::Runtime::new()?.block_on(async {
    /// # let warehouse = tempfile::tempdir()?;
    /// # let props = HashMap::from([(
    /// #     MEMORY_CATALOG_WAREHOUSE.to_string(),
    /// #     warehouse.path().display().to_string(),
    /// # )]);
    /// # let catalog = MemoryCatalogBuilder::default().load("memory", props).await?;
    /// # let namespace = NamespaceIdent::new("ns".to_string());
    /// # catalog.create_namespace(&namespace, HashMap::new()).await?;
    /// # let schema = Schema::builder()
    /// #     .with_fields(vec![
    /// #         NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
    /// #         NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
    /// #             .into(),
    /// #     ])
    /// #     .build()?;
    /// # let creation = TableCreation::builder().name("t".to_string()).schema(schema).build();
    /// # let table = catalog.create_table(&namespace, creation).await?;
    /// let provider = IcebergStaticTableProvider::try_new_from_table(table).await?;
    /// let ctx = SessionContext::new();
    /// let filters = [col("id").gt(lit(1))];
    /// let plan = provider
    ///     .scan(&ctx.state(), Some(&vec![1]), &filters, None)
    ///     .await?;
    /// let scan = plan.downcast_ref::<IcebergTableScan>().unwrap();
    ///
    /// // Rebuild an equivalent scan from the original's parts.
    /// let schema = provider.schema();
    /// let projection = scan
    ///     .projection()
    ///     .map(|names| {
    ///         names
    ///             .iter()
    ///             .map(|name| schema.index_of(name))
    ///             .collect::<Result<Vec<_>, _>>()
    ///     })
    ///     .transpose()?;
    /// let rebuilt = IcebergTableScan::new_with_predicate(
    ///     scan.table().clone(),
    ///     scan.snapshot_id(),
    ///     schema,
    ///     projection.as_deref(),
    ///     scan.predicates().cloned(),
    ///     scan.limit(),
    /// )?;
    /// assert_eq!(rebuilt.schema(), scan.schema());
    /// assert_eq!(rebuilt.predicates(), scan.predicates());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// # })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_with_predicate(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&[usize]>,
        predicate: Option<Predicate>,
        limit: Option<usize>,
    ) -> Result<Self> {
        let output_schema = match projection {
            None => schema,
            Some(projection) => Arc::new(schema.project(projection)?),
        };
        // The columns to read, by name; `None` reads them all.
        let projection = projection.map(|_| {
            output_schema
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect()
        });
        let plan_properties = Self::compute_properties(output_schema);

        Ok(Self {
            table,
            snapshot_id,
            plan_properties,
            projection,
            predicates: predicate,
            limit,
        })
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    pub fn projection(&self) -> Option<&[String]> {
        self.projection.as_deref()
    }

    pub fn predicates(&self) -> Option<&Predicate> {
        self.predicates.as_ref()
    }

    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Computes [`PlanProperties`] used in query optimization.
    fn compute_properties(schema: ArrowSchemaRef) -> Arc<PlanProperties> {
        // TODO:
        // This is more or less a placeholder, to be replaced
        // once we support output-partitioning
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }
}

impl ExecutionPlan for IcebergTableScan {
    fn name(&self) -> &str {
        "IcebergTableScan"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan + 'static>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let fut = get_batch_stream(
            self.table.clone(),
            self.snapshot_id,
            self.projection.clone(),
            self.predicates.clone(),
        );
        let stream = futures::stream::once(fut).try_flatten();

        // Apply limit if specified
        let limited_stream: Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>> =
            if let Some(limit) = self.limit {
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
            self.schema(),
            limited_stream,
        )))
    }
}

impl DisplayAs for IcebergTableScan {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(
            f,
            "IcebergTableScan projection:[{}] predicate:[{}]",
            self.projection
                .clone()
                .map_or(String::new(), |v| v.join(",")),
            self.predicates
                .clone()
                .map_or(String::from(""), |p| format!("{p}"))
        )?;
        if let Some(limit) = self.limit {
            write!(f, " limit:[{limit}]")?;
        }
        Ok(())
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
