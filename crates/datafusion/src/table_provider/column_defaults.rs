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

use datafusion::common::tree_node::TreeNode;
use datafusion::common::{DFSchema, ScalarValue, not_impl_err, plan_err};
use datafusion::datasource::TableProvider;
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, ExprSchemable, Volatility, lit};
use datafusion::optimizer::simplify_expressions::{ExprSimplifier, SimplifyContext};
use iceberg::arrow::{arrow_primitive_to_literal, type_to_arrow_type};
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

/// Folds the `DEFAULT` of column `name` to a value of type `ty`, or `None` for NULL.
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

    // The SQL planner doesn't coerce DEFAULT expressions, so a mixed-type one such as
    // `1 + 0.5` has to be coerced before it can be folded.
    let data_type = type_to_arrow_type(ty).map_err(to_datafusion_error)?;
    let simplifier = ExprSimplifier::new(SimplifyContext::default());
    let folded = simplifier.simplify(
        simplifier
            .coerce(default.clone(), &DFSchema::empty())?
            .cast_to(&data_type, &DFSchema::empty())?,
    )?;
    let Expr::Literal(value, _) = folded else {
        return plan_err!(
            "DEFAULT for column {name} must evaluate to a constant, got {default}"
        );
    };

    Ok(arrow_primitive_to_literal(&value.to_array()?, ty)
        .map_err(to_datafusion_error)?
        .pop()
        .flatten())
}

/// `value` as a scalar of the Arrow type that `type_to_arrow_type` maps `ty` to, or
/// `None` if it can't be converted.
fn primitive_to_scalar(
    ty: &PrimitiveType,
    value: &PrimitiveLiteral,
) -> Option<ScalarValue> {
    // The literal's storage type is enough here: casting to the column's Arrow type
    // reinterprets integers as dates, times and timestamps, and bytes as fixed-size
    // binary.
    let scalar = match (ty, value) {
        (PrimitiveType::Decimal { precision, scale }, PrimitiveLiteral::Int128(v)) => {
            ScalarValue::Decimal128(
                Some(*v),
                u8::try_from(*precision).ok()?,
                i8::try_from(*scale).ok()?,
            )
        }
        (_, PrimitiveLiteral::Boolean(v)) => ScalarValue::Boolean(Some(*v)),
        (_, PrimitiveLiteral::Int(v)) => ScalarValue::Int32(Some(*v)),
        (_, PrimitiveLiteral::Long(v)) => ScalarValue::Int64(Some(*v)),
        (_, PrimitiveLiteral::Float(v)) => ScalarValue::Float32(Some(v.into_inner())),
        (_, PrimitiveLiteral::Double(v)) => ScalarValue::Float64(Some(v.into_inner())),
        (_, PrimitiveLiteral::String(v)) => ScalarValue::Utf8(Some(v.clone())),
        (_, PrimitiveLiteral::Binary(v)) => ScalarValue::Binary(Some(v.clone())),
        (_, PrimitiveLiteral::UInt128(v)) => {
            ScalarValue::Binary(Some(v.to_be_bytes().to_vec()))
        }
        _ => return None,
    };
    let data_type = type_to_arrow_type(&Type::Primitive(ty.clone())).ok()?;
    scalar.cast_to(&data_type).ok()
}
