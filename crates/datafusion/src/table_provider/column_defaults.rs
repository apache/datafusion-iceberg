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

//! Conversions between DataFusion column defaults and Iceberg `write-default` values.
//!
//! A SQL `DEFAULT` is stored as the field's `write-default` when the table is created,
//! and each `write-default` is handed back to DataFusion, which fills it into columns
//! an INSERT omits.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, RecordBatchOptions};
use datafusion::arrow::datatypes::Schema as ArrowSchema;
use datafusion::common::tree_node::TreeNode;
use datafusion::common::{DFSchema, ScalarValue, not_impl_err, plan_err};
use datafusion::datasource::TableProvider;
use datafusion::error::Result;
use datafusion::logical_expr::execution_props::ExecutionProps;
use datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion::logical_expr::{Expr, ExprSchemable, Volatility, lit};
use datafusion::physical_expr::create_physical_expr;
use iceberg::arrow::{UTC_TIME_ZONE, arrow_primitive_to_literal, type_to_arrow_type};
use iceberg::spec::{Literal, PrimitiveLiteral, PrimitiveType, Schema, Type};

use crate::to_datafusion_error;

/// The `write-default` of each top-level field of `schema` that has one, as a literal
/// keyed by field name.
///
/// Defaults that have no DataFusion literal, such as those of non-primitive fields, are
/// left out, so DataFusion fills NULL into those columns instead.
pub(crate) fn column_defaults(schema: &Schema) -> HashMap<String, Expr> {
    schema
        .as_struct()
        .fields()
        .iter()
        .filter_map(
            |field| match (field.field_type.as_ref(), &field.write_default) {
                (Type::Primitive(ty), Some(Literal::Primitive(value))) => {
                    Some((field.name.clone(), lit(primitive_to_scalar(ty, value)?)))
                }
                _ => None,
            },
        )
        .collect()
}

/// `schema` with each top-level field's `write-default` set to the `DEFAULT` that
/// `table` declares for the column of the same name.
///
/// A `DEFAULT` must be a constant, because the table stores its value rather than the
/// expression. `DEFAULT NULL` is the same as no default.
pub(crate) fn with_write_defaults(
    schema: Schema,
    table: &dyn TableProvider,
) -> Result<Schema> {
    let fields = schema
        .as_struct()
        .fields()
        .iter()
        .map(|field| {
            let Some(default) = table.get_column_default(&field.name) else {
                return Ok(Arc::clone(field));
            };
            Ok(
                match default_to_literal(&field.name, &field.field_type, default)? {
                    Some(value) => {
                        Arc::new(field.as_ref().clone().with_write_default(value))
                    }
                    None => Arc::clone(field),
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Schema::builder()
        .with_schema_id(schema.schema_id())
        .with_identifier_field_ids(schema.identifier_field_ids())
        .with_fields(fields)
        .build()
        .map_err(to_datafusion_error)
}

/// Evaluates the `DEFAULT` of column `name` to a value of type `ty`, or `None` for NULL.
fn default_to_literal(name: &str, ty: &Type, default: &Expr) -> Result<Option<Literal>> {
    if !ty.is_primitive() {
        return not_impl_err!("DEFAULT is not supported for column {name} of type {ty}");
    }

    // Functions that aren't immutable, such as `now()` or `random()`, would be evaluated
    // once here instead of on every INSERT.
    let is_constant = !default.exists(|expr| {
        Ok(match expr {
            Expr::Column(_) => true,
            Expr::ScalarFunction(function) => {
                function.func.signature().volatility != Volatility::Immutable
            }
            _ => false,
        })
    })?;
    if !is_constant {
        return plan_err!("DEFAULT for column {name} must be a constant, got {default}");
    }

    let data_type = type_to_arrow_type(ty).map_err(to_datafusion_error)?;
    let default = default.clone().cast_to(&data_type, &DFSchema::empty())?;
    let one_row = RecordBatch::try_new_with_options(
        Arc::new(ArrowSchema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )?;
    let value = create_physical_expr(
        &default,
        &DFSchema::empty(),
        &ExecutionProps::new(),
        &PhysicalPlanningContext::default(),
    )?
    .evaluate(&one_row)?
    .into_array(1)?;

    Ok(arrow_primitive_to_literal(&value, ty)
        .map_err(to_datafusion_error)?
        .pop()
        .flatten())
}

/// `value` as a scalar of the Arrow type that `schema_to_arrow_schema` maps `ty` to, or
/// `None` if `value` isn't a value of `ty`.
fn primitive_to_scalar(
    ty: &PrimitiveType,
    value: &PrimitiveLiteral,
) -> Option<ScalarValue> {
    let utc = || Some(UTC_TIME_ZONE.into());
    let scalar = match (ty, value) {
        (PrimitiveType::Boolean, PrimitiveLiteral::Boolean(v)) => {
            ScalarValue::Boolean(Some(*v))
        }
        (PrimitiveType::Int, PrimitiveLiteral::Int(v)) => ScalarValue::Int32(Some(*v)),
        (PrimitiveType::Long, PrimitiveLiteral::Long(v)) => ScalarValue::Int64(Some(*v)),
        (PrimitiveType::Float, PrimitiveLiteral::Float(v)) => {
            ScalarValue::Float32(Some(v.into_inner()))
        }
        (PrimitiveType::Double, PrimitiveLiteral::Double(v)) => {
            ScalarValue::Float64(Some(v.into_inner()))
        }
        (PrimitiveType::Decimal { precision, scale }, PrimitiveLiteral::Int128(v)) => {
            ScalarValue::Decimal128(
                Some(*v),
                u8::try_from(*precision).ok()?,
                i8::try_from(*scale).ok()?,
            )
        }
        (PrimitiveType::Date, PrimitiveLiteral::Int(v)) => ScalarValue::Date32(Some(*v)),
        (PrimitiveType::Time, PrimitiveLiteral::Long(v)) => {
            ScalarValue::Time64Microsecond(Some(*v))
        }
        (PrimitiveType::Timestamp, PrimitiveLiteral::Long(v)) => {
            ScalarValue::TimestampMicrosecond(Some(*v), None)
        }
        (PrimitiveType::Timestamptz, PrimitiveLiteral::Long(v)) => {
            ScalarValue::TimestampMicrosecond(Some(*v), utc())
        }
        (PrimitiveType::TimestampNs, PrimitiveLiteral::Long(v)) => {
            ScalarValue::TimestampNanosecond(Some(*v), None)
        }
        (PrimitiveType::TimestamptzNs, PrimitiveLiteral::Long(v)) => {
            ScalarValue::TimestampNanosecond(Some(*v), utc())
        }
        (PrimitiveType::String, PrimitiveLiteral::String(v)) => {
            ScalarValue::Utf8(Some(v.clone()))
        }
        (PrimitiveType::Uuid, PrimitiveLiteral::UInt128(v)) => {
            ScalarValue::FixedSizeBinary(16, Some(v.to_be_bytes().to_vec()))
        }
        (PrimitiveType::Fixed(len), PrimitiveLiteral::Binary(v)) => {
            ScalarValue::FixedSizeBinary(i32::try_from(*len).ok()?, Some(v.clone()))
        }
        (PrimitiveType::Binary, PrimitiveLiteral::Binary(v)) => {
            ScalarValue::LargeBinary(Some(v.clone()))
        }
        _ => return None,
    };
    Some(scalar)
}
