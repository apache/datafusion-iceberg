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

//! Partition value projection for Iceberg tables.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, RecordBatch, StructArray, make_array};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::datatypes::{DataType, Fields, Schema as ArrowSchema};
use datafusion::common::cast::as_struct_array;
use datafusion::common::{internal_err, plan_err};
use datafusion::error::Result;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr_common::physical_expr::fmt_sql;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::{ColumnarValue, ExecutionPlan};
use iceberg::arrow::{
    PROJECTED_PARTITION_VALUE_COLUMN, schema_to_arrow_schema, strip_metadata_from_schema,
    type_to_arrow_type,
};
use iceberg::spec::{
    PartitionSpec, Schema as IcebergSchema, SchemaRef as IcebergSchemaRef, StructType,
    Type,
};
use iceberg::table::Table;
use iceberg::transform::{BoxedTransformFunction, create_transform_function};
use iceberg::{Error, ErrorKind};

use crate::to_datafusion_error;

/// Extends an ExecutionPlan with partition value calculations for Iceberg tables.
///
/// This function takes an input ExecutionPlan and extends it with an additional column
/// containing calculated partition values based on the table's partition specification.
/// For unpartitioned tables, returns the original plan unchanged.
///
/// # Arguments
/// * `input` - The input ExecutionPlan to extend
/// * `table` - The Iceberg table with partition specification
///
/// # Returns
/// * `Ok(Arc<dyn ExecutionPlan>)` - Extended plan with partition values column
/// * `Err` - If partition spec is not found or transformation fails
pub fn project_with_partition(
    input: Arc<dyn ExecutionPlan>,
    table: &Table,
) -> Result<Arc<dyn ExecutionPlan>> {
    let metadata = table.metadata();
    let partition_spec = metadata.default_partition_spec();
    let table_schema = metadata.current_schema();

    if partition_spec.is_unpartitioned() {
        return Ok(input);
    }

    let input_schema = input.schema();

    // Validate that input_schema matches the Iceberg table schema
    // Strip metadata from both schemas before comparison to ignore metadata differences
    let expected_arrow_schema =
        schema_to_arrow_schema(table_schema.as_ref()).map_err(to_datafusion_error)?;
    let input_schema_cleaned =
        strip_metadata_from_schema(&input_schema).map_err(to_datafusion_error)?;
    let expected_schema_cleaned = strip_metadata_from_schema(&expected_arrow_schema)
        .map_err(to_datafusion_error)?;

    if input_schema_cleaned != expected_schema_cleaned {
        return plan_err!(
            "Input schema does not match Iceberg table schema.\n\
             Expected schema: {expected_schema_cleaned}\n\
             Input schema: {input_schema_cleaned}"
        );
    }

    let mut projection_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
        Vec::with_capacity(input_schema.fields().len() + 1);

    for (index, field) in input_schema.fields().iter().enumerate() {
        let column_expr = Arc::new(Column::new(field.name(), index));
        projection_exprs.push((column_expr, field.name().clone()));
    }

    let partition_expr = Arc::new(PartitionExpr::try_new_over_table(
        partition_spec.clone(),
        table_schema.clone(),
    )?);
    projection_exprs.push((partition_expr, PROJECTED_PARTITION_VALUE_COLUMN.to_string()));

    let projection = ProjectionExec::try_new(projection_exprs, input)?;
    Ok(Arc::new(projection))
}

/// PhysicalExpr implementation for partition value calculation
///
/// Each partition field is computed from one child, the top-level column holding its
/// source, so [`Self::children`] are exactly the columns the expression reads.
/// Optimizer rules that move the expression onto a different input, such as merging
/// it into the projection below, rewrite those children like any other column
/// reference.
///
/// The transform functions cannot be serialized, so the spec and schema they were
/// built from are retained. A codec rebuilds the expression by passing them to
/// [`Self::try_new`] together with the decoded children.
#[derive(Debug, Clone)]
pub struct PartitionExpr {
    partition_spec: Arc<PartitionSpec>,
    table_schema: IcebergSchemaRef,
    transforms: Arc<[FieldTransform]>,
    partition_fields: Fields,
    sources: Vec<Arc<dyn PhysicalExpr>>,
}

/// How one partition field is computed from its child, derived from the spec and
/// schema.
#[derive(Debug)]
struct FieldTransform {
    /// Struct field positions from the child down to a nested source; empty for a
    /// top-level source.
    nested_path: Vec<usize>,
    transform: BoxedTransformFunction,
}

impl PartitionExpr {
    /// Builds the expression from the spec and schema that define it, reading each
    /// partition field's source from the matching expression in `sources`.
    ///
    /// `sources` holds one expression per partition field: the top-level input
    /// column that holds that field's source column. A codec passes the decoded
    /// children of the expression it serialized, which already point at the right
    /// input columns. The transforms and partition type are derived here rather than
    /// passed in, so they cannot drift from the retained spec and schema.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is unpartitioned, cannot be bound to the schema,
    /// or does not have exactly one source per partition field.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use datafusion::physical_expr::PhysicalExpr;
    /// use datafusion::physical_expr::expressions::Column;
    /// use iceberg_datafusion::write::PartitionExpr;
    /// use iceberg::spec::{NestedField, PartitionSpec, PrimitiveType, Schema, Transform, Type};
    ///
    /// let schema = Arc::new(
    ///     Schema::builder()
    ///         .with_fields(vec![
    ///             NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
    ///         ])
    ///         .build()?,
    /// );
    /// let spec = Arc::new(
    ///     PartitionSpec::builder(schema.clone())
    ///         .add_partition_field("id", "id_bucket", Transform::Bucket(16))?
    ///         .build()?,
    /// );
    /// let expr = PartitionExpr::try_new(spec, schema, vec![Arc::new(Column::new("id", 0))])?;
    ///
    /// // A worker rebuilds an equal expression from the retained spec and schema and
    /// // the decoded children.
    /// let rebuilt = PartitionExpr::try_new(
    ///     Arc::new(expr.partition_spec().as_ref().clone()),
    ///     Arc::new(expr.table_schema().as_ref().clone()),
    ///     expr.children().into_iter().cloned().collect(),
    /// )?;
    /// assert_eq!(expr, rebuilt);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_new(
        partition_spec: Arc<PartitionSpec>,
        table_schema: IcebergSchemaRef,
        sources: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Self> {
        if partition_spec.is_unpartitioned() {
            return Err(to_datafusion_error(Error::new(
                ErrorKind::DataInvalid,
                "Cannot compute partition values for an unpartitioned spec",
            )));
        }
        check_source_count(partition_spec.fields().len(), sources.len())?;
        let partition_type = partition_spec
            .partition_type(&table_schema)
            .map_err(to_datafusion_error)?;
        let DataType::Struct(partition_fields) =
            type_to_arrow_type(&Type::Struct(partition_type))
                .map_err(to_datafusion_error)?
        else {
            return internal_err!(
                "Iceberg partition type must convert to an Arrow struct"
            );
        };
        let transforms = partition_spec
            .fields()
            .iter()
            .map(|pf| {
                Ok(FieldTransform {
                    nested_path: source_path(&table_schema, pf.source_id)?[1..].to_vec(),
                    transform: create_transform_function(&pf.transform)
                        .map_err(to_datafusion_error)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            partition_spec,
            table_schema,
            transforms: transforms.into(),
            partition_fields,
            sources,
        })
    }

    /// Builds the expression over an input laid out like the table, which
    /// [`project_with_partition`] checks before using it.
    pub(crate) fn try_new_over_table(
        partition_spec: Arc<PartitionSpec>,
        table_schema: IcebergSchemaRef,
    ) -> Result<Self> {
        let sources = partition_spec
            .fields()
            .iter()
            .map(|pf| {
                let index = source_path(&table_schema, pf.source_id)?[0];
                let column = &table_schema.as_struct().fields()[index];
                Ok(Arc::new(Column::new(&column.name, index)) as Arc<dyn PhysicalExpr>)
            })
            .collect::<Result<Vec<_>>>()?;
        Self::try_new(partition_spec, table_schema, sources)
    }

    /// The partition spec this expression computes values for. With
    /// [`Self::table_schema`] and the children, all [`Self::try_new`] needs to
    /// rebuild an equal one.
    pub fn partition_spec(&self) -> &Arc<PartitionSpec> {
        &self.partition_spec
    }

    /// The table schema the spec is bound to. Needed to rebuild: the spec refers to
    /// columns by `source_id`, and only the schema resolves those.
    pub fn table_schema(&self) -> &IcebergSchemaRef {
        &self.table_schema
    }

    fn fmt_with(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        fmt_source: impl Fn(&dyn PhysicalExpr) -> String,
    ) -> std::fmt::Result {
        let fields: Vec<String> = self
            .partition_spec
            .fields()
            .iter()
            .zip(&self.sources)
            .map(|(pf, source)| {
                format!(
                    "{}: {}({})",
                    pf.name,
                    pf.transform,
                    fmt_source(source.as_ref())
                )
            })
            .collect();
        write!(f, "iceberg_partition_values({})", fields.join(", "))
    }
}

/// The position of each field from the top level of `schema` down to `source_id`.
fn source_path(schema: &IcebergSchema, source_id: i32) -> Result<Vec<usize>> {
    let mut path = Vec::new();
    if !push_field_path(schema.as_struct(), source_id, &mut path) {
        return Err(to_datafusion_error(Error::new(
            ErrorKind::DataInvalid,
            format!("Partition source field {source_id} not found in table schema"),
        )));
    }
    Ok(path)
}

/// Appends the positions from `fields` down to `field_id` to `path`, returning
/// whether the field was found. `path` is left as it was if not.
fn push_field_path(fields: &StructType, field_id: i32, path: &mut Vec<usize>) -> bool {
    fields.fields().iter().enumerate().any(|(index, field)| {
        path.push(index);
        let found = field.id == field_id
            || matches!(
                field.field_type.as_ref(),
                Type::Struct(nested) if push_field_path(nested, field_id, path)
            );
        if !found {
            path.pop();
        }
        found
    })
}

fn check_source_count(fields: usize, sources: usize) -> Result<()> {
    if fields != sources {
        return plan_err!(
            "PartitionExpr needs exactly one source per partition field: expected \
             {fields}, got {sources}"
        );
    }
    Ok(())
}

/// Child `index` of a struct array, null wherever the struct is null. Arrow leaves
/// a child's values under a null parent unspecified, so they must not be read.
fn struct_field(array: &ArrayRef, index: usize) -> Result<ArrayRef> {
    let parent = as_struct_array(array)?;
    let child = parent.column(index);
    let Some(parent_nulls) = parent.nulls() else {
        return Ok(Arc::clone(child));
    };
    let nulls = NullBuffer::union(Some(parent_nulls), child.logical_nulls().as_ref());
    Ok(make_array(
        child.to_data().into_builder().nulls(nulls).build()?,
    ))
}

// Equal when the spec, schema and sources are equal. The transforms and partition
// type are derived from the first two, so they take no part. Both comparisons include
// `spec_id` and `schema_id`, which `Hash` relies on.
impl PartialEq for PartitionExpr {
    fn eq(&self, other: &Self) -> bool {
        self.partition_spec == other.partition_spec
            && self.table_schema == other.table_schema
            && self.sources == other.sources
    }
}

impl Eq for PartitionExpr {}

impl PhysicalExpr for PartitionExpr {
    fn data_type(&self, _input_schema: &ArrowSchema) -> Result<DataType> {
        Ok(DataType::Struct(self.partition_fields.clone()))
    }

    fn nullable(&self, _input_schema: &ArrowSchema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let values = self
            .sources
            .iter()
            .zip(self.transforms.iter())
            .map(|(source, field)| {
                let mut source = source.evaluate(batch)?.into_array(batch.num_rows())?;
                for &index in &field.nested_path {
                    source = struct_field(&source, index)?;
                }
                field
                    .transform
                    .transform(source)
                    .map_err(to_datafusion_error)
            })
            .collect::<Result<Vec<_>>>()?;
        let array = StructArray::try_new(self.partition_fields.clone(), values, None)?;
        Ok(ColumnarValue::Array(Arc::new(array)))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.sources.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        check_source_count(self.sources.len(), children.len())?;
        Ok(Arc::new(Self {
            sources: children,
            ..self.as_ref().clone()
        }))
    }

    fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with(f, |source| fmt_sql(source).to_string())
    }
}

impl std::fmt::Display for PartitionExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with(f, |source| source.to_string())
    }
}

impl std::hash::Hash for PartitionExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Neither PartitionSpec nor Schema implements Hash, so hash their ids:
        // equal expressions agree on both, and collisions fall through to eq.
        self.partition_spec.spec_id().hash(state);
        self.table_schema.schema_id().hash(state);
        self.sources.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use datafusion::arrow::array::{ArrayRef, Int32Array, StringArray, StructArray};
    use datafusion::arrow::datatypes::{DataType, Field, Fields};
    use datafusion::arrow::util::pretty::pretty_format_columns;
    use datafusion::parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use datafusion::physical_plan::empty::EmptyExec;
    use iceberg::io::FileIO;
    use iceberg::spec::{
        FormatVersion, NestedField, PrimitiveType, Schema, StructType, Transform, Type,
    };
    use iceberg::{Runtime, TableIdent};

    use super::*;
    use crate::from_datafusion_error;

    fn hash_of(expr: &PartitionExpr) -> u64 {
        let mut hasher = DefaultHasher::new();
        expr.hash(&mut hasher);
        hasher.finish()
    }

    /// Evaluates `expr` over `batch` and renders the partition values as a table.
    fn partition_values(expr: &dyn PhysicalExpr, batch: &RecordBatch) -> String {
        let values = expr
            .evaluate(batch)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap();
        pretty_format_columns(PROJECTED_PARTITION_VALUE_COLUMN, &[values])
            .unwrap()
            .to_string()
    }

    /// Partitions [`schema_with_part`] by `identity(id)`, leaving `part` unused.
    fn partitioned_on_id() -> PartitionExpr {
        let table_schema = schema_with_part(PrimitiveType::Int);
        let partition_spec = Arc::new(
            PartitionSpec::builder(table_schema.clone())
                .add_partition_field("id", "id_partition", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );
        PartitionExpr::try_new_over_table(partition_spec, table_schema).unwrap()
    }

    // `part` typed per caller, so a spec built against one schema can be paired with
    // a schema that no longer matches it, as happens when a codec rebuilds.
    fn schema_with_part(part_type: PrimitiveType) -> IcebergSchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                        .into(),
                    NestedField::required(2, "part", Type::Primitive(part_type)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn test_partition_expr_with_projection() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                    .into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String))
                    .into(),
            ])
            .build()
            .unwrap();

        let partition_spec = Arc::new(
            PartitionSpec::builder(Arc::new(table_schema.clone()))
                .add_partition_field("id", "id_partition", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let input = Arc::new(EmptyExec::new(arrow_schema.clone()));

        let mut projection_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
            Vec::with_capacity(arrow_schema.fields().len() + 1);
        for (i, field) in arrow_schema.fields().iter().enumerate() {
            let column_expr = Arc::new(Column::new(field.name(), i));
            projection_exprs.push((column_expr, field.name().clone()));
        }

        let partition_expr = Arc::new(
            PartitionExpr::try_new_over_table(
                partition_spec,
                Arc::new(table_schema.clone()),
            )
            .unwrap(),
        );
        projection_exprs
            .push((partition_expr, PROJECTED_PARTITION_VALUE_COLUMN.to_string()));

        let projection = ProjectionExec::try_new(projection_exprs, input).unwrap();
        let result = Arc::new(projection);

        assert_eq!(result.schema().fields().len(), 3);
        assert_eq!(result.schema().field(0).name(), "id");
        assert_eq!(result.schema().field(1).name(), "name");
        assert_eq!(result.schema().field(2).name(), "_partition");
    }

    #[test]
    fn test_partition_expr_evaluate() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                    .into(),
                NestedField::required(2, "data", Type::Primitive(PrimitiveType::String))
                    .into(),
            ])
            .build()
            .unwrap();

        let partition_spec = PartitionSpec::builder(Arc::new(table_schema.clone()))
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![10, 20, 30])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let expr = PartitionExpr::try_new_over_table(
            Arc::new(partition_spec),
            Arc::new(table_schema),
        )
        .unwrap();

        // Partition fields are optional and carry their partition field id.
        let partition_type = DataType::Struct(Fields::from(vec![
            Field::new("id_partition", DataType::Int32, true).with_metadata(
                HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "1000".to_string(),
                )]),
            ),
        ]));
        assert_eq!(expr.data_type(&arrow_schema).unwrap(), partition_type);
        assert!(!expr.nullable(&arrow_schema).unwrap());
        insta::assert_snapshot!(partition_values(&expr, &batch), @r"
        +--------------------+
        | _partition         |
        +--------------------+
        | {id_partition: 10} |
        | {id_partition: 20} |
        | {id_partition: 30} |
        +--------------------+
        ");
    }

    #[test]
    fn test_partition_expr_children_are_its_source_columns() {
        let expr = Arc::new(partitioned_on_id());
        // `part` is not a source, so it is not a child.
        assert_eq!(
            expr.to_string(),
            "iceberg_partition_values(id_partition: identity(id@0))"
        );

        // An input laid out unlike the table, as when the optimizer merges the
        // partition projection into one that reorders columns. Rewriting the child
        // is all it takes to read `id` from its new position.
        let remapped = expr
            .with_new_children(vec![Arc::new(Column::new("id", 2))])
            .unwrap();
        assert_eq!(
            remapped.to_string(),
            "iceberg_partition_values(id_partition: identity(id@2))"
        );
        let batch = RecordBatch::try_from_iter(vec![
            ("x", Arc::new(Int32Array::from(vec![7, 8])) as ArrayRef),
            ("part", Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef),
            ("id", Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef),
        ])
        .unwrap();
        insta::assert_snapshot!(partition_values(remapped.as_ref(), &batch), @r"
        +--------------------+
        | _partition         |
        +--------------------+
        | {id_partition: 10} |
        | {id_partition: 20} |
        +--------------------+
        ");
    }

    #[test]
    fn test_with_new_children_requires_one_child_per_partition_field() {
        let expr = Arc::new(partitioned_on_id());

        let err = expr
            .with_new_children(vec![
                Arc::new(Column::new("id", 0)),
                Arc::new(Column::new("part", 1)),
            ])
            .unwrap_err();
        insta::assert_snapshot!(err.strip_backtrace(), @"Error during planning: PartitionExpr needs exactly one source per partition field: expected 1, got 2");
    }

    #[test]
    fn test_partition_expr_rebuilds_from_its_retained_parts() {
        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                        .into(),
                ])
                .build()
                .unwrap(),
        );
        let partition_spec = Arc::new(
            PartitionSpec::builder(table_schema.clone())
                .add_partition_field("id", "id_partition", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            arrow_schema,
            vec![Arc::new(Int32Array::from(vec![10, 20, 30]))],
        )
        .unwrap();

        let expr =
            PartitionExpr::try_new_over_table(partition_spec, table_schema).unwrap();

        // Rebuild from deep copies, as a codec would after decoding; cloning the
        // Arcs instead would let a pointer-based impl pass.
        let rebuilt = PartitionExpr::try_new(
            Arc::new(expr.partition_spec().as_ref().clone()),
            Arc::new(expr.table_schema().as_ref().clone()),
            vec![Arc::new(Column::new("id", 0))],
        )
        .unwrap();

        let eval = |e: &PartitionExpr| match e.evaluate(&batch).unwrap() {
            ColumnarValue::Array(array) => array,
            _ => panic!("Expected array result"),
        };
        assert_eq!(&eval(&rebuilt), &eval(&expr));

        // Same values is not enough: it must also compare and hash as the same
        // expression, or plan-level equality and dedup treat the two as unrelated.
        assert_eq!(rebuilt, expr);
        assert_eq!(hash_of(&rebuilt), hash_of(&expr));
    }

    #[test]
    fn test_try_new_error_paths() {
        let int_schema = schema_with_part(PrimitiveType::Int);
        let on_part = |transform| {
            Arc::new(
                PartitionSpec::builder(int_schema.clone())
                    .add_partition_field("part", "p", transform)
                    .unwrap()
                    .build()
                    .unwrap(),
            )
        };
        let without_part = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        let part = || vec![Arc::new(Column::new("part", 1)) as Arc<dyn PhysicalExpr>];

        // try_new takes the spec, schema and sources separately, so a codec can pass
        // ones that do not fit. project_with_partition reaches none of these: it
        // returns early when unpartitioned and derives all three from the table.
        let cases = [
            (
                Arc::new(PartitionSpec::builder(int_schema.clone()).build().unwrap()),
                int_schema.clone(),
                vec![],
            ),
            (on_part(Transform::Identity), int_schema.clone(), vec![]),
            (on_part(Transform::Identity), without_part, part()),
            (
                on_part(Transform::Bucket(4)),
                schema_with_part(PrimitiveType::Boolean),
                part(),
            ),
        ];

        // Each error with the iceberg kind that survives `from_datafusion_error`.
        let errors: Vec<String> = cases
            .into_iter()
            .map(|(spec, schema, sources)| {
                let err = PartitionExpr::try_new(spec, schema, sources).unwrap_err();
                let message = err.strip_backtrace();
                format!("{}: {message}", from_datafusion_error(err).kind())
            })
            .collect();
        insta::assert_snapshot!(errors.join("\n\n"), @r"
        DataInvalid: IcebergError(DataInvalid)
        caused by
        Execution error: DataInvalid => Cannot compute partition values for an unpartitioned spec

        Unexpected: Error during planning: PartitionExpr needs exactly one source per partition field: expected 1, got 0

        DataInvalid: IcebergError(DataInvalid)
        caused by
        Execution error: DataInvalid => Partition source field 2 not found in table schema

        DataInvalid: IcebergError(DataInvalid)
        caused by
        Execution error: DataInvalid => boolean is not a valid input type of bucket transform
        ");
    }

    #[test]
    fn test_partition_expr_equality_is_value_based() {
        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                        .into(),
                    NestedField::required(2, "part", Type::Primitive(PrimitiveType::Int))
                        .into(),
                ])
                .build()
                .unwrap(),
        );
        let spec = |spec_id: i32, source: &str| {
            Arc::new(
                PartitionSpec::builder(table_schema.clone())
                    .with_spec_id(spec_id)
                    .add_partition_field(source, "p", Transform::Identity)
                    .unwrap()
                    .build()
                    .unwrap(),
            )
        };
        let expr =
            |spec| PartitionExpr::try_new_over_table(spec, table_schema.clone()).unwrap();

        // The same inputs describe the same expression.
        assert_eq!(expr(spec(1, "id")), expr(spec(1, "id")));

        // Genuinely different specs stay apart.
        assert_ne!(expr(spec(1, "id")), expr(spec(2, "part")));

        // Same spec_id, different partition column: comparing only the ids that
        // Hash uses would wrongly call these equal.
        assert_ne!(expr(spec(7, "id")), expr(spec(7, "part")));
    }

    #[test]
    fn test_partition_expr_differs_on_sources() {
        // The same spec and schema reading `id` from a different input column, as
        // after the optimizer moves the expression onto a reordered input. Equating
        // the two would let DataFusion reuse one in place of the other.
        let expr = partitioned_on_id();
        let moved = PartitionExpr::try_new(
            expr.partition_spec().clone(),
            expr.table_schema().clone(),
            vec![Arc::new(Column::new("id", 2))],
        )
        .unwrap();

        assert_ne!(expr, moved);
    }

    #[test]
    fn test_partition_expr_differs_on_schema_with_shared_id() {
        // Two tables can both be on schema 0 with `part` typed differently, which
        // changes the partition type. The ids Hash uses are identical here, so only
        // the struct comparison in eq separates these. Narrowing eq to those ids
        // would wrongly call them equal.
        let int_schema = schema_with_part(PrimitiveType::Int);
        let long_schema = schema_with_part(PrimitiveType::Long);
        let shared = Arc::new(
            PartitionSpec::builder(int_schema.clone())
                .add_partition_field("part", "p", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        let on_int =
            PartitionExpr::try_new_over_table(shared.clone(), int_schema).unwrap();
        let on_long = PartitionExpr::try_new_over_table(shared, long_schema).unwrap();

        assert_eq!(
            on_int.table_schema().schema_id(),
            on_long.table_schema().schema_id()
        );
        assert_ne!(on_int, on_long);
    }

    #[test]
    fn test_nested_partition() {
        let address_type = StructType::new(vec![
            NestedField::required(3, "street", Type::Primitive(PrimitiveType::String))
                .into(),
            NestedField::required(4, "city", Type::Primitive(PrimitiveType::String))
                .into(),
        ]);
        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                        .into(),
                    NestedField::optional(2, "address", Type::Struct(address_type))
                        .into(),
                ])
                .build()
                .unwrap(),
        );
        let partition_spec = Arc::new(
            PartitionSpec::builder(table_schema.clone())
                .add_partition_field(
                    "address.city",
                    "city_partition",
                    Transform::Identity,
                )
                .unwrap()
                .build()
                .unwrap(),
        );
        let expr =
            PartitionExpr::try_new_over_table(partition_spec, table_schema).unwrap();
        // The child is the top-level column that holds the nested source.
        assert_eq!(
            expr.to_string(),
            "iceberg_partition_values(city_partition: identity(address@1))"
        );

        // The third address is null, but its `city` slot still holds a value,
        // which must not become a partition value.
        let address = StructArray::new(
            Fields::from(vec![
                Field::new("street", DataType::Utf8, false),
                Field::new("city", DataType::Utf8, false),
            ]),
            vec![
                Arc::new(StringArray::from(vec!["123 Main St", "456 Oak Ave", ""])),
                Arc::new(StringArray::from(vec!["New York", "Los Angeles", "stale"])),
            ],
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let batch = RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef),
            ("address", Arc::new(address) as ArrayRef),
        ])
        .unwrap();
        insta::assert_snapshot!(partition_values(&expr, &batch), @r"
        +-------------------------------+
        | _partition                    |
        +-------------------------------+
        | {city_partition: New York}    |
        | {city_partition: Los Angeles} |
        | {city_partition: }            |
        +-------------------------------+
        ");
    }

    #[tokio::test]
    async fn test_schema_validation_matching_schemas() {
        let table_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                        .into(),
                    NestedField::required(
                        2,
                        "name",
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = PartitionSpec::builder(table_schema.clone())
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let sort_order = iceberg::spec::SortOrder::builder()
            .build(&table_schema)
            .unwrap();

        let table_metadata_builder = iceberg::spec::TableMetadataBuilder::new(
            (*table_schema).clone(),
            partition_spec,
            sort_order,
            "/test/table".to_string(),
            FormatVersion::V2,
            HashMap::new(),
        )
        .unwrap();

        let table_metadata = table_metadata_builder.build().unwrap();

        // Create Arrow schema matching the table schema
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let input = Arc::new(EmptyExec::new(arrow_schema));

        let table = Table::builder()
            .metadata(table_metadata.metadata)
            .identifier(TableIdent::from_strs(["test", "table"]).unwrap())
            .file_io(FileIO::new_with_fs())
            .metadata_location("/test/metadata.json")
            .runtime(Runtime::current())
            .build()
            .unwrap();

        let result = project_with_partition(input, &table);
        assert!(result.is_ok(), "Schema validation should pass");
    }

    #[tokio::test]
    async fn test_schema_validation_mismatched_schemas() {
        let table_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                        .into(),
                    NestedField::required(
                        2,
                        "name",
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = PartitionSpec::builder(table_schema.clone())
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let sort_order = iceberg::spec::SortOrder::builder()
            .build(&table_schema)
            .unwrap();

        let table_metadata_builder = iceberg::spec::TableMetadataBuilder::new(
            (*table_schema).clone(),
            partition_spec,
            sort_order,
            "/test/table".to_string(),
            FormatVersion::V2,
            HashMap::new(),
        )
        .unwrap();

        let table_metadata = table_metadata_builder.build().unwrap();

        // Create Arrow schema with different field name (mismatched)
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("different_name", DataType::Utf8, false), // Wrong field name
        ]));

        let input = Arc::new(EmptyExec::new(arrow_schema));

        let table = Table::builder()
            .metadata(table_metadata.metadata)
            .identifier(TableIdent::from_strs(["test", "table"]).unwrap())
            .file_io(FileIO::new_with_fs())
            .metadata_location("/test/metadata.json")
            .runtime(Runtime::current())
            .build()
            .unwrap();

        let result = project_with_partition(input, &table);
        assert!(
            result.is_err(),
            "Schema validation should fail for mismatched schemas"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Input schema does not match Iceberg table schema")
        );
    }

    #[tokio::test]
    async fn test_schema_validation_with_metadata_differences() {
        let table_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                        .into(),
                    NestedField::required(
                        2,
                        "name",
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = PartitionSpec::builder(table_schema.clone())
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let sort_order = iceberg::spec::SortOrder::builder()
            .build(&table_schema)
            .unwrap();

        let table_metadata_builder = iceberg::spec::TableMetadataBuilder::new(
            (*table_schema).clone(),
            partition_spec,
            sort_order,
            "/test/table".to_string(),
            FormatVersion::V2,
            HashMap::new(),
        )
        .unwrap();

        let table_metadata = table_metadata_builder.build().unwrap();

        // Create Arrow schema with metadata (should be ignored in comparison)
        let mut metadata = HashMap::new();
        metadata.insert("extra".to_string(), "metadata".to_string());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(metadata.clone()),
            Field::new("name", DataType::Utf8, false).with_metadata(metadata),
        ]));

        let input = Arc::new(EmptyExec::new(arrow_schema));

        let table = Table::builder()
            .metadata(table_metadata.metadata)
            .identifier(TableIdent::from_strs(["test", "table"]).unwrap())
            .file_io(FileIO::new_with_fs())
            .metadata_location("/test/metadata.json")
            .runtime(Runtime::current())
            .build()
            .unwrap();

        let result = project_with_partition(input, &table);
        assert!(
            result.is_ok(),
            "Schema validation should pass even with metadata differences"
        );
    }
}
