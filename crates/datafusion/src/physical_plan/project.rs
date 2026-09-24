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

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::{DataType, Schema as ArrowSchema};
use datafusion::common::plan_err;
use datafusion::error::Result;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::{ColumnarValue, ExecutionPlan};
use iceberg::arrow::{
    PROJECTED_PARTITION_VALUE_COLUMN, PartitionValueCalculator, schema_to_arrow_schema,
    strip_metadata_from_schema,
};
use iceberg::spec::{PartitionSpec, SchemaRef as IcebergSchemaRef};
use iceberg::table::Table;

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

    let partition_expr = Arc::new(PartitionExpr::try_new(
        partition_spec.clone(),
        table_schema.clone(),
    )?);
    projection_exprs.push((partition_expr, PROJECTED_PARTITION_VALUE_COLUMN.to_string()));

    let projection = ProjectionExec::try_new(projection_exprs, input)?;
    Ok(Arc::new(projection))
}

/// PhysicalExpr implementation for partition value calculation
///
/// The [`PartitionValueCalculator`] cannot be serialized, so the spec and schema it
/// was built from are retained for [`Self::try_new`] to rebuild from.
#[derive(Debug, Clone)]
pub struct PartitionExpr {
    calculator: Arc<PartitionValueCalculator>,
    partition_spec: Arc<PartitionSpec>,
    table_schema: IcebergSchemaRef,
}

impl PartitionExpr {
    /// Builds the expression from the spec and schema that define it.
    ///
    /// The calculator is built here rather than passed in, so it cannot drift from
    /// the retained inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is unpartitioned or cannot be bound to the schema.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use datafusion_iceberg::physical_plan::PartitionExpr;
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
    /// let expr = PartitionExpr::try_new(spec, schema)?;
    ///
    /// // A worker rebuilds an equal expression from the two retained inputs.
    /// let rebuilt = PartitionExpr::try_new(
    ///     Arc::new(expr.partition_spec().as_ref().clone()),
    ///     Arc::new(expr.table_schema().as_ref().clone()),
    /// )?;
    /// assert_eq!(expr, rebuilt);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_new(
        partition_spec: Arc<PartitionSpec>,
        table_schema: IcebergSchemaRef,
    ) -> Result<Self> {
        let calculator = PartitionValueCalculator::try_new(
            partition_spec.as_ref(),
            table_schema.as_ref(),
        )
        .map_err(to_datafusion_error)?;
        Ok(Self {
            calculator: Arc::new(calculator),
            partition_spec,
            table_schema,
        })
    }

    /// The partition spec this expression computes values for. With
    /// [`Self::table_schema`], all [`Self::try_new`] needs to rebuild an equal one.
    pub fn partition_spec(&self) -> &Arc<PartitionSpec> {
        &self.partition_spec
    }

    /// The table schema the spec is bound to. Needed to rebuild: the spec refers to
    /// columns by `source_id`, and only the schema resolves those.
    pub fn table_schema(&self) -> &IcebergSchemaRef {
        &self.table_schema
    }
}

// Equal when the spec and schema are equal. The calculator is derived from them, so
// it takes no part. Both comparisons include `spec_id` and `schema_id`, which `Hash`
// relies on.
impl PartialEq for PartitionExpr {
    fn eq(&self, other: &Self) -> bool {
        self.partition_spec == other.partition_spec
            && self.table_schema == other.table_schema
    }
}

impl Eq for PartitionExpr {}

impl PhysicalExpr for PartitionExpr {
    fn data_type(&self, _input_schema: &ArrowSchema) -> Result<DataType> {
        Ok(self.calculator.partition_arrow_type().clone())
    }

    fn nullable(&self, _input_schema: &ArrowSchema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let array = self
            .calculator
            .calculate(batch)
            .map_err(to_datafusion_error)?;
        Ok(ColumnarValue::Array(array))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(self)
    }

    fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let field_names: Vec<String> = self
            .partition_spec
            .fields()
            .iter()
            .map(|pf| format!("{}({})", pf.transform, pf.name))
            .collect();
        write!(f, "iceberg_partition_values[{}]", field_names.join(", "))
    }
}

impl std::fmt::Display for PartitionExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let field_names: Vec<&str> = self
            .partition_spec
            .fields()
            .iter()
            .map(|pf| pf.name.as_str())
            .collect();
        write!(f, "iceberg_partition_values({})", field_names.join(", "))
    }
}

impl std::hash::Hash for PartitionExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Neither PartitionSpec nor Schema implements Hash, so hash their ids:
        // equal expressions agree on both, and collisions fall through to eq.
        self.partition_spec.spec_id().hash(state);
        self.table_schema.schema_id().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use datafusion::arrow::array::{ArrayRef, Int32Array, StructArray};
    use datafusion::arrow::datatypes::{DataType, Field, Fields};
    use datafusion::physical_plan::empty::EmptyExec;
    use iceberg::spec::{
        NestedField, PrimitiveType, Schema, StructType, Transform, Type,
    };
    use iceberg::test_utils::test_runtime;

    use super::*;

    fn hash_of(expr: &PartitionExpr) -> u64 {
        let mut hasher = DefaultHasher::new();
        expr.hash(&mut hasher);
        hasher.finish()
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
    fn test_partition_calculator_basic() {
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

        let partition_spec = PartitionSpec::builder(Arc::new(table_schema.clone()))
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let calculator =
            PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();

        // Verify partition type
        assert_eq!(calculator.partition_type().fields().len(), 1);
        assert_eq!(calculator.partition_type().fields()[0].name, "id_partition");
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
            PartitionExpr::try_new(partition_spec, Arc::new(table_schema.clone()))
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
                Arc::new(datafusion::arrow::array::StringArray::from(vec![
                    "a", "b", "c",
                ])),
            ],
        )
        .unwrap();

        let partition_spec = Arc::new(partition_spec);
        let calculator =
            PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();
        let partition_type = calculator.partition_arrow_type().clone();
        let expr = PartitionExpr::try_new(partition_spec, Arc::new(table_schema.clone()))
            .unwrap();

        assert_eq!(expr.data_type(&arrow_schema).unwrap(), partition_type);
        assert!(!expr.nullable(&arrow_schema).unwrap());

        let result = expr.evaluate(&batch).unwrap();
        match result {
            ColumnarValue::Array(array) => {
                let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
                let id_partition = struct_array
                    .column_by_name("id_partition")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                assert_eq!(id_partition.value(0), 10);
                assert_eq!(id_partition.value(1), 20);
                assert_eq!(id_partition.value(2), 30);
            }
            _ => panic!("Expected array result"),
        }
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

        let expr = PartitionExpr::try_new(partition_spec, table_schema).unwrap();

        // Rebuild from deep copies, as a codec would; cloning the Arcs instead
        // would let a pointer-based impl pass.
        let rebuilt = PartitionExpr::try_new(
            Arc::new(expr.partition_spec().as_ref().clone()),
            Arc::new(expr.table_schema().as_ref().clone()),
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

        // try_new takes the spec and schema separately, so a codec can pair two that
        // do not fit. project_with_partition reaches none of these: it returns early
        // when unpartitioned and reads both from the same table metadata.
        let cases = [
            (
                Arc::new(PartitionSpec::builder(int_schema.clone()).build().unwrap()),
                int_schema.clone(),
                "unpartitioned",
            ),
            (
                on_part(Transform::Identity),
                without_part,
                "Field not found",
            ),
            (
                on_part(Transform::Bucket(4)),
                schema_with_part(PrimitiveType::Boolean),
                "not a valid input type of bucket transform",
            ),
        ];

        for (spec, schema, expected) in cases {
            let err = PartitionExpr::try_new(spec, schema)
                .unwrap_err()
                .to_string();
            assert!(err.contains(expected), "expected {expected:?}, got {err}");
        }
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
        let expr = |spec| PartitionExpr::try_new(spec, table_schema.clone()).unwrap();

        // The same inputs describe the same expression.
        assert_eq!(expr(spec(1, "id")), expr(spec(1, "id")));

        // Genuinely different specs stay apart.
        assert_ne!(expr(spec(1, "id")), expr(spec(2, "part")));

        // Same spec_id, different partition column: comparing only the ids that
        // Hash uses would wrongly call these equal.
        assert_ne!(expr(spec(7, "id")), expr(spec(7, "part")));
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

        let on_int = PartitionExpr::try_new(shared.clone(), int_schema).unwrap();
        let on_long = PartitionExpr::try_new(shared, long_schema).unwrap();

        assert_eq!(
            on_int.table_schema().schema_id(),
            on_long.table_schema().schema_id()
        );
        assert_ne!(on_int, on_long);
    }

    #[test]
    fn test_nested_partition() {
        let address_struct = StructType::new(vec![
            NestedField::required(3, "street", Type::Primitive(PrimitiveType::String))
                .into(),
            NestedField::required(4, "city", Type::Primitive(PrimitiveType::String))
                .into(),
        ]);

        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                    .into(),
                NestedField::required(2, "address", Type::Struct(address_struct)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = PartitionSpec::builder(Arc::new(table_schema.clone()))
            .add_partition_field("address.city", "city_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let struct_fields = Fields::from(vec![
            Field::new("street", DataType::Utf8, false),
            Field::new("city", DataType::Utf8, false),
        ]);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("address", DataType::Struct(struct_fields), false),
        ]));

        let street_array = Arc::new(datafusion::arrow::array::StringArray::from(vec![
            "123 Main St",
            "456 Oak Ave",
        ]));
        let city_array = Arc::new(datafusion::arrow::array::StringArray::from(vec![
            "New York",
            "Los Angeles",
        ]));

        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("street", DataType::Utf8, false)),
                street_array as ArrayRef,
            ),
            (
                Arc::new(Field::new("city", DataType::Utf8, false)),
                city_array as ArrayRef,
            ),
        ]);

        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(struct_array),
            ],
        )
        .unwrap();

        let calculator =
            PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();
        let array = calculator.calculate(&batch).unwrap();

        let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
        let city_partition = struct_array
            .column_by_name("city_partition")
            .unwrap()
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StringArray>()
            .unwrap();

        assert_eq!(city_partition.value(0), "New York");
        assert_eq!(city_partition.value(1), "Los Angeles");
    }

    #[test]
    fn test_schema_validation_matching_schemas() {
        use iceberg::TableIdent;
        use iceberg::io::FileIO;
        use iceberg::spec::{FormatVersion, NestedField, PrimitiveType, Schema, Type};

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
            std::collections::HashMap::new(),
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
            .runtime(test_runtime())
            .build()
            .unwrap();

        let result = project_with_partition(input, &table);
        assert!(result.is_ok(), "Schema validation should pass");
    }

    #[test]
    fn test_schema_validation_mismatched_schemas() {
        use iceberg::TableIdent;
        use iceberg::io::FileIO;
        use iceberg::spec::{FormatVersion, NestedField, PrimitiveType, Schema, Type};

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
            std::collections::HashMap::new(),
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
            .runtime(test_runtime())
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

    #[test]
    fn test_schema_validation_with_metadata_differences() {
        use std::collections::HashMap;

        use iceberg::TableIdent;
        use iceberg::io::FileIO;
        use iceberg::spec::{FormatVersion, NestedField, PrimitiveType, Schema, Type};

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
            .runtime(test_runtime())
            .build()
            .unwrap();

        let result = project_with_partition(input, &table);
        assert!(
            result.is_ok(),
            "Schema validation should pass even with metadata differences"
        );
    }
}
