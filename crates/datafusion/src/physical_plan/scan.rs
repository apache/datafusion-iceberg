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
use datafusion::arrow::compute::SortOptions;
use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use datafusion::catalog::Session;
use datafusion::common::plan_err;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::Result;
use datafusion::execution::memory_pool::MemoryConsumer;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, PhysicalExpr, PhysicalSortExpr,
};
use datafusion::physical_plan::common::spawn_buffered;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet,
};
use datafusion::physical_plan::sorts::streaming_merge::StreamingMergeBuilder;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties};
use datafusion::prelude::Expr;
use futures::{Stream, StreamExt, TryStreamExt};
use iceberg::expr::Predicate;
use iceberg::scan::{FileScanTask, TableScan};
use iceberg::spec::{
    NullOrder, PrimitiveType, Schema, SortDirection, SortOrder, Transform, Type,
};
use iceberg::table::Table;

use super::expr_to_predicate::convert_filters_to_predicate;
use crate::{IcebergDataFusionConfig, to_datafusion_error};

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
    /// The columns to read, by name: the fields of the output schema
    projection: Vec<String>,
    /// Filters to apply to the table scan
    predicates: Option<Predicate>,
    /// Optional limit on the number of rows to return
    limit: Option<usize>,
    /// The data files to merge in the order the scan reports, listed while the
    /// scan was planned. `None` lists the files when the scan runs and reports
    /// no order.
    sorted: Option<SortedTasks>,
    metrics: ExecutionPlanMetricsSet,
}

/// Data files that all record the same sort order, and the part of that order
/// the scan reports, over the columns of its output.
#[derive(Debug)]
struct SortedTasks {
    tasks: Vec<FileScanTask>,
    ordering: LexOrdering,
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

    /// Creates a scan of `table` from an already-converted Iceberg
    /// [`Predicate`] rather than DataFusion filters, for rebuilding a scan from
    /// its parts, such as after sending them to another process. A predicate
    /// cannot be converted back to the filters it came from. A scan with
    /// [`Self::sorted_tasks`] is only rebuilt once they are also passed to
    /// [`Self::with_sorted_tasks`].
    ///
    /// Each argument takes what the matching accessor returns (`schema` what
    /// [`ExecutionPlan::schema`] does, and `predicates` what
    /// [`Self::predicates`] does):
    ///
    /// - `snapshot_id`: the snapshot to read, or `None` for the table's current
    ///   snapshot.
    /// - `schema`: the Arrow schema the scan outputs. The scan reads the
    ///   columns of the same names from the snapshot it scans, and no others.
    ///   A name that snapshot's schema lacks fails the scan when it runs.
    /// - `predicates`: pushed down to Iceberg to skip data files and rows. The
    ///   table providers report their filters as
    ///   [`Inexact`](datafusion::logical_expr::TableProviderFilterPushDown::Inexact),
    ///   so DataFusion still applies them above the scan.
    /// - `limit`: the most rows the scan returns, or `None` for all of them.
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
    /// // Rebuild an equivalent scan from the original's accessors alone.
    /// let rebuilt = IcebergTableScan::new_with_predicate(
    ///     scan.table().clone(),
    ///     scan.snapshot_id(),
    ///     scan.schema(),
    ///     scan.predicates().cloned(),
    ///     scan.limit(),
    /// );
    /// assert_eq!(rebuilt.schema(), scan.schema());
    /// assert_eq!(rebuilt.projection(), scan.projection());
    /// assert_eq!(rebuilt.predicates(), scan.predicates());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// # })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_with_predicate(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        predicates: Option<Predicate>,
        limit: Option<usize>,
    ) -> Self {
        // Reading the columns by name, rather than all of them, keeps the
        // batches matching `schema` even when the table has columns it lacks.
        let projection = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let plan_properties = Self::compute_properties(schema, None);

        Self {
            table,
            snapshot_id,
            plan_properties,
            projection,
            predicates,
            limit,
            sorted: None,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Lists the scan's data files and reports the sort order they all record,
    /// if the session enables `iceberg.planning.preserve_data_ordering` and
    /// they are few enough to merge. Otherwise returns the scan unchanged.
    pub(crate) async fn plan_sort_order(self, state: &dyn Session) -> Result<Self> {
        let config = state
            .config()
            .options()
            .extensions
            .get::<IcebergDataFusionConfig>()
            .map(|config| config.planning.clone())
            .unwrap_or_default();
        if !config.preserve_data_ordering {
            return Ok(self);
        }

        let metadata = self.table.metadata();
        let snapshot = match self.snapshot_id {
            Some(snapshot_id) => metadata.snapshot_by_id(snapshot_id),
            None => metadata.current_snapshot(),
        };
        let Some(snapshot) = snapshot else {
            return Ok(self);
        };
        let schema = snapshot.schema(metadata).map_err(to_datafusion_error)?;
        let output = self.schema();
        if !metadata
            .sort_orders_iter()
            .any(|order| lex_ordering(order, &schema, &output).is_some())
        {
            return Ok(self);
        }

        let tasks: Vec<FileScanTask> = build_table_scan(
            &self.table,
            self.snapshot_id,
            self.projection.clone(),
            self.predicates.clone(),
        )?
        .plan_files()
        .await
        .map_err(to_datafusion_error)?
        .take(config.max_merge_files.saturating_add(1))
        .try_collect()
        .await
        .map_err(to_datafusion_error)?;
        if tasks.len() > config.max_merge_files {
            return Ok(self);
        }
        Ok(match shared_ordering(&tasks, &output) {
            Some(ordering) => self.with_sorted(SortedTasks { tasks, ordering }),
            None => self,
        })
    }

    /// Makes the scan read `tasks`, as returned by [`Self::sorted_tasks`], and
    /// merge them in the sort order they all record, which it reports to
    /// DataFusion. A scan rebuilt from the accessors of one that has sorted
    /// tasks must be given them, as the plan above it may rely on that order.
    ///
    /// Fails if `tasks` do not all record a sort order the scan can report.
    pub fn with_sorted_tasks(self, tasks: Vec<FileScanTask>) -> Result<Self> {
        let Some(ordering) = shared_ordering(&tasks, &self.schema()) else {
            return plan_err!(
                "IcebergTableScan cannot report an order for the given data files: \
                 they do not all record the same sort order over its columns"
            );
        };
        Ok(self.with_sorted(SortedTasks { tasks, ordering }))
    }

    fn with_sorted(mut self, sorted: SortedTasks) -> Self {
        self.plan_properties =
            Self::compute_properties(self.schema(), Some(&sorted.ordering));
        self.sorted = Some(sorted);
        self
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    /// The names of the columns the scan reads, which are the fields of its
    /// schema.
    pub fn projection(&self) -> &[String] {
        &self.projection
    }

    pub fn predicates(&self) -> Option<&Predicate> {
        self.predicates.as_ref()
    }

    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// The data files the scan merges in the order it reports, listed while it
    /// was planned, or `None` if it reports no order and lists its files when
    /// it runs.
    pub fn sorted_tasks(&self) -> Option<&[FileScanTask]> {
        self.sorted.as_ref().map(|sorted| sorted.tasks.as_slice())
    }

    /// Computes [`PlanProperties`] used in query optimization.
    fn compute_properties(
        schema: ArrowSchemaRef,
        ordering: Option<&LexOrdering>,
    ) -> Arc<PlanProperties> {
        let eq_properties = match ordering {
            Some(ordering) => {
                EquivalenceProperties::new_with_orderings(schema, [ordering.clone()])
            }
            None => EquivalenceProperties::new(schema),
        };
        // TODO:
        // This is more or less a placeholder, to be replaced
        // once we support output-partitioning
        Arc::new(PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }

    /// Reads each sorted data file as its own stream and merges the streams in
    /// the order the scan reports.
    fn merge_sorted(
        &self,
        sorted: &SortedTasks,
        partition: usize,
        context: &TaskContext,
    ) -> Result<SendableRecordBatchStream> {
        // The builder defaults to the reader settings `TableScan::to_arrow`
        // uses, and a limit of one data file keeps a stream in its file's order.
        let reader = self
            .table
            .reader_builder()
            .with_data_file_concurrency_limit(1)
            .build();
        let mut streams = sorted
            .tasks
            .iter()
            .map(|task| {
                let tasks = futures::stream::iter([Ok(task.clone())]).boxed();
                let batches = reader
                    .clone()
                    .read(tasks)
                    .map_err(to_datafusion_error)?
                    .stream()
                    .map_err(to_datafusion_error);
                Ok(
                    Box::pin(RecordBatchStreamAdapter::new(self.schema(), batches))
                        as SendableRecordBatchStream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if streams.len() == 1 {
            return Ok(streams.remove(0));
        }

        let reservation = MemoryConsumer::new(format!("IcebergTableScan[{partition}]"))
            .register(&context.runtime_env().memory_pool);
        StreamingMergeBuilder::new()
            .with_streams(
                streams
                    .into_iter()
                    .map(|stream| spawn_buffered(stream, 1))
                    .collect(),
            )
            .with_schema(self.schema())
            .with_expressions(&sorted.ordering)
            .with_metrics(BaselineMetrics::new(&self.metrics, partition))
            .with_batch_size(context.session_config().batch_size())
            .with_fetch(self.limit)
            .with_reservation(reservation)
            .build()
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

    fn metrics(&self) -> Option<MetricsSet> {
        self.sorted.as_ref().map(|_| self.metrics.clone_inner())
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let stream: Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>> =
            match &self.sorted {
                Some(sorted) => self.merge_sorted(sorted, partition, &context)?,
                None => {
                    let fut = get_batch_stream(
                        self.table.clone(),
                        self.snapshot_id,
                        self.projection.clone(),
                        self.predicates.clone(),
                    );
                    Box::pin(futures::stream::once(fut).try_flatten())
                }
            };

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
            self.projection.join(","),
            self.predicates
                .clone()
                .map_or(String::from(""), |p| format!("{p}"))
        )?;
        if let Some(sorted) = &self.sorted {
            write!(
                f,
                " output_ordering:[{}] files:[{}]",
                sorted.ordering,
                sorted.tasks.len()
            )?;
        }
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
    column_names: Vec<String>,
    predicates: Option<Predicate>,
) -> Result<Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>> {
    let table_scan = build_table_scan(&table, snapshot_id, column_names, predicates)?;

    let stream = table_scan
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?
        .map_err(to_datafusion_error);
    Ok(Box::pin(stream))
}

/// Builds the [`TableScan`] of the columns `column_names` of a snapshot of
/// `table`, or of its current snapshot, filtered by `predicates`.
fn build_table_scan(
    table: &Table,
    snapshot_id: Option<i64>,
    column_names: Vec<String>,
    predicates: Option<Predicate>,
) -> Result<TableScan> {
    let scan_builder = match snapshot_id {
        Some(snapshot_id) => table.scan().snapshot_id(snapshot_id),
        None => table.scan(),
    };

    let mut scan_builder = scan_builder.select(column_names);
    if let Some(pred) = predicates {
        scan_builder = scan_builder.with_filter(pred);
    }
    scan_builder.build().map_err(to_datafusion_error)
}

/// The order of the columns of `output` that every one of `tasks` is sorted
/// in, or `None` if they record different sort orders, or one records none.
fn shared_ordering(tasks: &[FileScanTask], output: &ArrowSchema) -> Option<LexOrdering> {
    let first = tasks.first()?;
    let order = first.sort_order()?;
    if !tasks.iter().all(|task| {
        task.sort_order_id() == first.sort_order_id() && task.sort_order().is_some()
    }) {
        return None;
    }
    lex_ordering(order, first.schema(), output)
}

/// The longest leading part of `order` that DataFusion can merge and report as
/// an order of the columns of `output`, or `None` if that is empty. `schema`
/// is the Iceberg schema the scan reads.
fn lex_ordering(
    order: &SortOrder,
    schema: &Schema,
    output: &ArrowSchema,
) -> Option<LexOrdering> {
    let sort_exprs = order.fields.iter().map_while(|field| {
        if field.transform != Transform::Identity {
            return None;
        }
        let source = schema.as_struct().field_by_id(field.source_id)?;
        if !orders_like_writers(&source.field_type) {
            return None;
        }
        let index = output.index_of(&source.name).ok()?;
        Some(PhysicalSortExpr::new(
            Arc::new(Column::new(&source.name, index)),
            SortOptions {
                descending: field.direction == SortDirection::Descending,
                nulls_first: field.null_order == NullOrder::First,
            },
        ))
    });
    LexOrdering::new(sort_exprs)
}

/// Whether DataFusion orders values of `field_type` as the engines that write
/// sorted Iceberg files do. It orders -0.0 before 0.0 and negative NaN before
/// every number, where Spark treats the zeros as equal and NaN as the largest
/// value, and the Iceberg spec leaves the order of UUIDs undefined
/// (apache/iceberg#14216).
fn orders_like_writers(field_type: &Type) -> bool {
    matches!(
        field_type,
        Type::Primitive(
            PrimitiveType::Boolean
                | PrimitiveType::Int
                | PrimitiveType::Long
                | PrimitiveType::Decimal { .. }
                | PrimitiveType::Date
                | PrimitiveType::Time
                | PrimitiveType::Timestamp
                | PrimitiveType::Timestamptz
                | PrimitiveType::TimestampNs
                | PrimitiveType::TimestamptzNs
                | PrimitiveType::String
                | PrimitiveType::Fixed(_)
                | PrimitiveType::Binary
        )
    )
}

#[cfg(test)]
mod tests {
    use iceberg::arrow::schema_to_arrow_schema;
    use iceberg::spec::{NestedField, SortField, StructType};

    use super::*;

    fn schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                    .into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
                    .into(),
                NestedField::optional(3, "ts", Type::Primitive(PrimitiveType::Timestamp))
                    .into(),
                NestedField::optional(
                    4,
                    "amount",
                    Type::Primitive(PrimitiveType::Decimal {
                        precision: 10,
                        scale: 2,
                    }),
                )
                .into(),
                NestedField::optional(5, "price", Type::Primitive(PrimitiveType::Double))
                    .into(),
                NestedField::optional(6, "ratio", Type::Primitive(PrimitiveType::Float))
                    .into(),
                NestedField::optional(7, "uid", Type::Primitive(PrimitiveType::Uuid))
                    .into(),
                NestedField::optional(
                    8,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(
                            9,
                            "inner",
                            Type::Primitive(PrimitiveType::Int),
                        )
                        .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap()
    }

    fn field(
        source_id: i32,
        transform: Transform,
        direction: SortDirection,
        null_order: NullOrder,
    ) -> SortField {
        SortField::builder()
            .source_id(source_id)
            .transform(transform)
            .direction(direction)
            .null_order(null_order)
            .build()
    }

    fn asc(source_id: i32) -> SortField {
        field(
            source_id,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        )
    }

    /// The order reported for `fields` over the columns `projection` of
    /// [`schema`], rendered as DataFusion displays it.
    fn ordering(fields: Vec<SortField>, projection: &[&str]) -> Option<String> {
        let schema = schema();
        let mut builder = SortOrder::builder();
        builder.with_order_id(1);
        for field in fields {
            builder.with_sort_field(field);
        }
        let order = builder.build(&schema).unwrap();
        let arrow = schema_to_arrow_schema(&schema).unwrap();
        let indices: Vec<usize> = projection
            .iter()
            .map(|name| arrow.index_of(name).unwrap())
            .collect();
        let output = arrow.project(&indices).unwrap();
        lex_ordering(&order, &schema, &output).map(|ordering| ordering.to_string())
    }

    const ALL: &[&str] = &["id", "name", "ts", "amount", "price", "ratio", "uid", "s"];

    #[test]
    fn test_identity_fields_keep_direction_and_null_order() {
        let fields = vec![
            field(
                1,
                Transform::Identity,
                SortDirection::Ascending,
                NullOrder::First,
            ),
            field(
                2,
                Transform::Identity,
                SortDirection::Ascending,
                NullOrder::Last,
            ),
            field(
                3,
                Transform::Identity,
                SortDirection::Descending,
                NullOrder::First,
            ),
            field(
                4,
                Transform::Identity,
                SortDirection::Descending,
                NullOrder::Last,
            ),
        ];
        assert_eq!(
            ordering(fields, ALL).as_deref(),
            Some("id@0 ASC, name@1 ASC NULLS LAST, ts@2 DESC, amount@3 DESC NULLS LAST")
        );
    }

    #[test]
    fn test_ordering_stops_at_non_identity_transform() {
        for transform in [Transform::Bucket(4), Transform::Truncate(2), Transform::Day] {
            let source_id = if transform == Transform::Day { 3 } else { 2 };
            let fields = vec![
                asc(1),
                field(
                    source_id,
                    transform,
                    SortDirection::Ascending,
                    NullOrder::First,
                ),
                asc(4),
            ];
            assert_eq!(
                ordering(fields, ALL).as_deref(),
                Some("id@0 ASC"),
                "{transform}"
            );
        }
        let fields = vec![field(
            1,
            Transform::Bucket(4),
            SortDirection::Ascending,
            NullOrder::First,
        )];
        assert_eq!(ordering(fields, ALL), None);
    }

    #[test]
    fn test_ordering_stops_at_floating_point_and_uuid() {
        for source_id in [5, 6, 7] {
            assert_eq!(
                ordering(vec![asc(1), asc(source_id), asc(2)], ALL).as_deref(),
                Some("id@0 ASC")
            );
            assert_eq!(
                ordering(vec![asc(1), asc(source_id)], ALL).as_deref(),
                Some("id@0 ASC")
            );
            assert_eq!(ordering(vec![asc(source_id), asc(1)], ALL), None);
        }
    }

    #[test]
    fn test_ordering_stops_at_nested_field() {
        assert_eq!(
            ordering(vec![asc(1), asc(9), asc(2)], ALL).as_deref(),
            Some("id@0 ASC")
        );
        assert_eq!(ordering(vec![asc(9)], ALL), None);
    }

    #[test]
    fn test_ordering_stops_at_unprojected_field() {
        assert_eq!(
            ordering(vec![asc(1), asc(2), asc(3)], &["ts", "id"]).as_deref(),
            Some("id@1 ASC")
        );
        assert_eq!(ordering(vec![asc(2), asc(1)], &["id"]), None);
    }

    #[test]
    fn test_ordering_stops_at_field_missing_from_schema() {
        let mut builder = SortOrder::builder();
        builder.with_order_id(1);
        builder.with_sort_field(asc(1));
        builder.with_sort_field(asc(42));
        builder.with_sort_field(asc(2));
        let order = builder.build_unbound().unwrap();
        let schema = schema();
        let output = schema_to_arrow_schema(&schema).unwrap();
        assert_eq!(
            lex_ordering(&order, &schema, &output).map(|o| o.to_string()),
            Some("id@0 ASC".to_string())
        );
    }
}
