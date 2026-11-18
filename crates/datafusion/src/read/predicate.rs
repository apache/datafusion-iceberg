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

use datafusion::arrow::datatypes::DataType;
use datafusion::logical_expr::expr::{InList, ScalarFunction};
use datafusion::logical_expr::{BinaryExpr, Expr, Like, Operator};
use datafusion::scalar::ScalarValue;
use iceberg::expr::{BinaryExpression, Predicate, PredicateOperator, Reference};
use iceberg::spec::{Datum, PrimitiveLiteral};

/// A filter converted to an iceberg [`Predicate`].
enum Converted {
    /// Matches exactly the rows the filter matches.
    Exact(Predicate),
    /// Implied by the filter but not equivalent to it, because an unconvertible
    /// AND operand was dropped. Sound to push down, unsound to negate.
    Implied(Predicate),
}

impl Converted {
    fn into_predicate(self) -> Predicate {
        match self {
            Converted::Exact(predicate) | Converted::Implied(predicate) => predicate,
        }
    }
}

/// Converts DataFusion filters ([`Expr`]) to an iceberg [`Predicate`].
/// If none of the filters could be converted, return `None` which adds no predicates to the scan operation.
/// If the conversion was successful, return the converted predicates combined with an AND operator.
pub fn convert_filters_to_predicate(filters: &[Expr]) -> Option<Predicate> {
    filters
        .iter()
        .filter_map(convert_filter_to_predicate)
        .reduce(Predicate::and)
}

fn convert_filter_to_predicate(expr: &Expr) -> Option<Predicate> {
    // A bare column as a whole filter is a boolean column check: `col = true`.
    // TODO: Also convert it as an AND or OR operand, as `NOT col` already is.
    // `col = true` is equivalent to a boolean column, so that is sound.
    match column(expr) {
        Some(column) => Some(column.equal_to(Datum::bool(true))),
        None => predicate(expr).map(Converted::into_predicate),
    }
}

/// Converts a filter expression, or returns `None` if no part of it can be
/// pushed down.
fn predicate(expr: &Expr) -> Option<Converted> {
    let exact = match expr {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::And,
            right,
        }) => return and(predicate(left), predicate(right)),
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Or,
            right,
        }) => return Some(or(predicate(left)?, predicate(right)?)),
        Expr::BinaryExpr(binary) => comparison(binary)?,
        Expr::Not(inner) => match column(inner) {
            // NOT of a bare boolean column: `col = false`
            Some(column) => column.equal_to(Datum::bool(false)),
            None => match predicate(inner)? {
                Converted::Exact(p) => !p,
                Converted::Implied(_) => return None,
            },
        },
        Expr::InList(InList {
            expr,
            list,
            negated,
        }) => {
            let column = column(expr)?;
            let datums = list.iter().map(literal).collect::<Option<Vec<_>>>()?;
            if *negated {
                column.is_not_in(datums)
            } else {
                column.is_in(datums)
            }
        }
        Expr::IsNull(expr) => column(expr)?.is_null(),
        Expr::IsNotNull(expr) => column(expr)?.is_not_null(),
        // Only a simple prefix pattern (`'prefix%'`) is pushed down. Iceberg's
        // StartsWith is case-sensitive, so ILIKE isn't, and neither is an
        // explicit ESCAPE clause.
        // TODO: Support an ESCAPE clause by having `like_prefix` unescape with
        // that character instead of `\`.
        Expr::Like(Like {
            negated,
            expr,
            pattern,
            escape_char: None,
            case_insensitive: false,
        }) => {
            let datum = literal(pattern)?;
            let PrimitiveLiteral::String(pattern) = datum.literal() else {
                return None;
            };
            let prefix = Datum::string(like_prefix(pattern)?);
            let column = column(expr)?;
            if *negated {
                column.not_starts_with(prefix)
            } else {
                column.starts_with(prefix)
            }
        }
        // Scalar functions are identified by name, unlike dedicated `Expr`
        // variants such as `Expr::IsNull`.
        Expr::ScalarFunction(ScalarFunction { func, args }) => {
            match (func.name(), args.as_slice()) {
                ("isnan", [arg]) => resolve_nan_preserving_reference(arg)?.is_nan(),
                _ => return None,
            }
        }
        _ => return None,
    };
    Some(Converted::Exact(exact))
}

/// Drops an unconvertible operand, which leaves a predicate implied by the
/// conjunction.
fn and(left: Option<Converted>, right: Option<Converted>) -> Option<Converted> {
    match (left, right) {
        (Some(Converted::Exact(left)), Some(Converted::Exact(right))) => {
            Some(Converted::Exact(left.and(right)))
        }
        (Some(left), Some(right)) => Some(Converted::Implied(
            left.into_predicate().and(right.into_predicate()),
        )),
        (Some(kept), None) | (None, Some(kept)) => {
            Some(Converted::Implied(kept.into_predicate()))
        }
        (None, None) => None,
    }
}

/// Takes both operands: dropping one would drop the rows only it matches.
fn or(left: Converted, right: Converted) -> Converted {
    match (left, right) {
        (Converted::Exact(left), Converted::Exact(right)) => {
            Converted::Exact(left.or(right))
        }
        (left, right) => {
            Converted::Implied(left.into_predicate().or(right.into_predicate()))
        }
    }
}

/// Converts `column op literal`, or `literal op column` with `op` swapped.
fn comparison(binary: &BinaryExpr) -> Option<Predicate> {
    let (column, op, datum) = match (column(&binary.left), literal(&binary.right)) {
        (Some(column), Some(datum)) => (column, binary.op, datum),
        _ => (
            column(&binary.right)?,
            binary.op.swap()?,
            literal(&binary.left)?,
        ),
    };
    let op = match op {
        Operator::Eq => PredicateOperator::Eq,
        Operator::NotEq => PredicateOperator::NotEq,
        Operator::Lt => PredicateOperator::LessThan,
        Operator::LtEq => PredicateOperator::LessThanOrEq,
        Operator::Gt => PredicateOperator::GreaterThan,
        Operator::GtEq => PredicateOperator::GreaterThanOrEq,
        _ => return None,
    };
    Some(Predicate::Binary(BinaryExpression::new(op, column, datum)))
}

fn column(expr: &Expr) -> Option<Reference> {
    match expr {
        Expr::Column(column) => Some(Reference::new(column.name())),
        _ => None,
    }
}

fn literal(expr: &Expr) -> Option<Datum> {
    match expr {
        Expr::Literal(value, _) => scalar_value_to_datum(value),
        _ => None,
    }
}

/// Returns the literal prefix of a `prefix%` LIKE pattern, or `None` for any other
/// pattern. Without an `ESCAPE` clause, DataFusion and Arrow treat `\` as an escape
/// character: `\c` matches a literal `c`, and a trailing `\` matches itself. The prefix
/// is unescaped the same way, and an escaped trailing `%` is not a wildcard.
fn like_prefix(pattern: &str) -> Option<String> {
    let mut prefix = String::with_capacity(pattern.len());
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => prefix.push(chars.next().unwrap_or('\\')),
            '%' => return chars.as_str().is_empty().then_some(prefix),
            '_' => return None,
            c => prefix.push(c),
        }
    }
    None
}

/// Attempts to resolve a numeric expression argument down to a single column
/// [`Reference`] such that `isnan(arg)` is logically equivalent to
/// `isnan(reference)`.
///
/// Both table providers report filter pushdown as `Inexact`, so DataFusion
/// re-applies the original predicate after scanning. We therefore only need the
/// pushed-down predicate to be implied by the original filter (it may match
/// extra rows, but must never drop a matching one). Every transformation handled
/// here preserves NaN-ness *exactly* — the result is NaN if and only if the
/// wrapped column is NaN — so both `isnan(arg)` and `NOT isnan(arg)` are sound:
///
/// * negation: `-x` is NaN iff `x` is NaN
/// * `abs(x)`: `abs(x)` is NaN iff `x` is NaN
/// * `x + c`, `c + x`, `x - c`, `c - x` for a finite literal `c`
/// * `x * c`, `c * x`, `x / c` for a finite, non-zero literal `c`
///
/// Multiplication/division by zero and `c / x` are intentionally rejected: e.g.
/// `x * 0` is NaN when `x` is `±inf`, so it does not imply `x` is NaN.
fn resolve_nan_preserving_reference(expr: &Expr) -> Option<Reference> {
    match expr {
        Expr::Column(column) => Some(Reference::new(column.name())),
        Expr::Negative(inner) => resolve_nan_preserving_reference(inner),
        Expr::ScalarFunction(ScalarFunction { func, args })
            if func.name() == "abs" && args.len() == 1 =>
        {
            resolve_nan_preserving_reference(&args[0])
        }
        Expr::BinaryExpr(binary) => resolve_nan_preserving_binary(binary),
        _ => None,
    }
}

/// Resolves the column reference from an arithmetic expression that combines a
/// single column with a finite literal while preserving NaN-ness. See
/// [`resolve_nan_preserving_reference`] for the soundness argument.
///
/// Expressions with column references on both sides (e.g. `(x + 1) * (x - 2)`)
/// are not supported. Handling them safely would require both operands to
/// resolve to the *same* column (`x + y` cannot be expressed as a single
/// `col IS NAN`) and the operator combination itself to be NaN-preserving:
/// `(x + 1) * (x - 2)` is NaN iff `x` is NaN, but `(x + 1) - (x - 2)` is NaN
/// for `x = inf` (`inf - inf`) even though `x` is not.
///
/// TODO: support NaN-preserving expressions with column references on both
/// sides, see <https://github.com/apache/iceberg-rust/issues/2154>.
fn resolve_nan_preserving_binary(binary: &BinaryExpr) -> Option<Reference> {
    let (left, right) = (&binary.left, &binary.right);
    match binary.op {
        // `x + c`, `c + x`, `x - c` and `c - x` are NaN iff `x` is NaN, for any
        // finite literal `c`. The column may be on either side.
        Operator::Plus | Operator::Minus => {
            if finite_literal(right).is_some() {
                resolve_nan_preserving_reference(left)
            } else if finite_literal(left).is_some() {
                resolve_nan_preserving_reference(right)
            } else {
                None
            }
        }

        // `x * c` and `c * x` are NaN iff `x` is NaN, but only when `c` is
        // non-zero. Per IEEE-754:
        //   - inf is not NaN
        //   - inf * 0 is NaN
        // so multiplying by zero is rejected. The column may be on either side.
        Operator::Multiply => {
            if matches!(finite_literal(right), Some(c) if c != 0.0) {
                resolve_nan_preserving_reference(left)
            } else if matches!(finite_literal(left), Some(c) if c != 0.0) {
                resolve_nan_preserving_reference(right)
            } else {
                None
            }
        }

        // `x / c` is NaN iff `x` is NaN, for a finite non-zero literal `c`.
        // `c / x` is rejected and the column must be the dividend (left side).
        // Per IEEE-754:
        //   - 0 is not NaN
        //   - 0 / 0 is NaN
        // so `c / x` is not NaN-preserving.
        Operator::Divide => {
            if matches!(finite_literal(right), Some(c) if c != 0.0) {
                resolve_nan_preserving_reference(left)
            } else {
                None
            }
        }

        _ => None,
    }
}

/// Returns the value of `expr` as an `f64` if it is a finite numeric literal
/// (i.e. not a non-literal, non-numeric, or infinite/NaN value). The numeric
/// conversion is delegated to DataFusion's [`ScalarValue::cast_to`]; the value
/// is only used to inspect finiteness and sign (precision loss is irrelevant).
fn finite_literal(expr: &Expr) -> Option<f64> {
    let Expr::Literal(value, _) = expr else {
        return None;
    };
    match value.cast_to(&DataType::Float64).ok()? {
        ScalarValue::Float64(Some(v)) if v.is_finite() => Some(v),
        _ => None,
    }
}

const MILLIS_PER_DAY: i64 = 24 * 60 * 60 * 1000;

/// Convert a scalar value to an iceberg datum.
fn scalar_value_to_datum(value: &ScalarValue) -> Option<Datum> {
    match value {
        ScalarValue::Boolean(Some(v)) => Some(Datum::bool(*v)),
        ScalarValue::Int8(Some(v)) => Some(Datum::int(*v as i32)),
        ScalarValue::Int16(Some(v)) => Some(Datum::int(*v as i32)),
        ScalarValue::Int32(Some(v)) => Some(Datum::int(*v)),
        ScalarValue::Int64(Some(v)) => Some(Datum::long(*v)),
        ScalarValue::Float32(Some(v)) => Some(Datum::double(*v as f64)),
        ScalarValue::Float64(Some(v)) => Some(Datum::double(*v)),
        ScalarValue::Utf8(Some(v)) => Some(Datum::string(v.clone())),
        ScalarValue::LargeUtf8(Some(v)) => Some(Datum::string(v.clone())),
        ScalarValue::Binary(Some(v)) => Some(Datum::binary(v.clone())),
        ScalarValue::LargeBinary(Some(v)) => Some(Datum::binary(v.clone())),
        ScalarValue::Date32(Some(v)) => Some(Datum::date(*v)),
        ScalarValue::Date64(Some(v)) => Some(Datum::date((*v / MILLIS_PER_DAY) as i32)),
        // Timestamp conversions
        // Note: TimestampSecond and TimestampMillisecond are not handled here because
        // DataFusion's type coercion always converts them to match the column type
        // (either TimestampMicrosecond or TimestampNanosecond) before predicate pushdown.
        // See unit tests for how those conversions would work if needed.
        ScalarValue::TimestampMicrosecond(Some(v), None) => {
            Some(Datum::timestamp_micros(*v))
        }
        ScalarValue::TimestampNanosecond(Some(v), None) => {
            Some(Datum::timestamp_nanos(*v))
        }
        // Note: Arrow stores tz-aware timestamps as UTC epoch values; the tz
        // only affects display. The same is true for iceberg's tz-aware datum
        // types.
        ScalarValue::TimestampMicrosecond(Some(v), Some(_)) => {
            Some(Datum::timestamptz_micros(*v))
        }
        ScalarValue::TimestampNanosecond(Some(v), Some(_)) => {
            Some(Datum::timestamptz_nanos(*v))
        }
        // TODO: Convert `Decimal128` literals with `Datum::decimal`, so filters
        // on decimal columns are pushed down.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use datafusion::common::DFSchema;
    use datafusion::logical_expr::utils::split_conjunction;
    use datafusion::prelude::{Expr, SessionContext};
    use iceberg::expr::{Predicate, Reference};
    use iceberg::spec::Datum;
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

    use super::convert_filters_to_predicate;

    fn create_test_schema() -> DFSchema {
        let arrow_schema =
            Schema::new(vec![
                Field::new("foo", DataType::Int32, true).with_metadata(HashMap::from([
                    (PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string()),
                ])),
                Field::new("bar", DataType::Utf8, true).with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "2".to_string(),
                )])),
                Field::new("ts", DataType::Timestamp(TimeUnit::Second, None), true)
                    .with_metadata(HashMap::from([(
                        PARQUET_FIELD_ID_META_KEY.to_string(),
                        "3".to_string(),
                    )])),
                Field::new("qux", DataType::Float64, true).with_metadata(HashMap::from(
                    [(PARQUET_FIELD_ID_META_KEY.to_string(), "4".to_string())],
                )),
                Field::new("b", DataType::Boolean, true).with_metadata(HashMap::from([
                    (PARQUET_FIELD_ID_META_KEY.to_string(), "5".to_string()),
                ])),
            ]);
        DFSchema::try_from_qualified_schema("my_table", &arrow_schema).unwrap()
    }

    fn convert_to_iceberg_predicate(sql: &str) -> Option<Predicate> {
        let df_schema = create_test_schema();
        let expr = SessionContext::new()
            .parse_sql_expr(sql, &df_schema)
            .unwrap();
        let exprs: Vec<Expr> = split_conjunction(&expr).into_iter().cloned().collect();
        convert_filters_to_predicate(&exprs[..])
    }

    #[test]
    fn test_predicate_conversion_with_single_condition() {
        let predicate = convert_to_iceberg_predicate("foo = 1").unwrap();
        assert_eq!(predicate, Reference::new("foo").equal_to(Datum::long(1)));

        let predicate = convert_to_iceberg_predicate("foo != 1").unwrap();
        assert_eq!(
            predicate,
            Reference::new("foo").not_equal_to(Datum::long(1))
        );

        let predicate = convert_to_iceberg_predicate("foo > 1").unwrap();
        assert_eq!(
            predicate,
            Reference::new("foo").greater_than(Datum::long(1))
        );

        let predicate = convert_to_iceberg_predicate("foo >= 1").unwrap();
        assert_eq!(
            predicate,
            Reference::new("foo").greater_than_or_equal_to(Datum::long(1))
        );

        let predicate = convert_to_iceberg_predicate("foo < 1").unwrap();
        assert_eq!(predicate, Reference::new("foo").less_than(Datum::long(1)));

        let predicate = convert_to_iceberg_predicate("foo <= 1").unwrap();
        assert_eq!(
            predicate,
            Reference::new("foo").less_than_or_equal_to(Datum::long(1))
        );

        let predicate = convert_to_iceberg_predicate("foo is null").unwrap();
        assert_eq!(predicate, Reference::new("foo").is_null());

        let predicate = convert_to_iceberg_predicate("foo is not null").unwrap();
        assert_eq!(predicate, Reference::new("foo").is_not_null());

        let predicate = convert_to_iceberg_predicate("foo in (5, 6)").unwrap();
        assert_eq!(
            predicate,
            Reference::new("foo").is_in([Datum::long(5), Datum::long(6)])
        );

        let predicate = convert_to_iceberg_predicate("foo not in (5, 6)").unwrap();
        assert_eq!(
            predicate,
            Reference::new("foo").is_not_in([Datum::long(5), Datum::long(6)])
        );

        let predicate = convert_to_iceberg_predicate("not foo = 1").unwrap();
        assert_eq!(predicate, !Reference::new("foo").equal_to(Datum::long(1)));
    }

    #[test]
    fn test_predicate_conversion_with_single_unsupported_condition() {
        for sql in [
            "foo + 1 = 1",
            "length(bar) = 1",
            "foo in (1, 2, foo)",
            "1 in (foo)",
            "foo + 1 is null",
            "foo = qux",
            "1 = 1",
            "foo is distinct from 1",
        ] {
            assert_eq!(convert_to_iceberg_predicate(sql), None, "sql: {sql}");
        }
    }

    #[test]
    fn test_predicate_conversion_with_single_condition_rev() {
        let foo = || Reference::new("foo");
        for (sql, expected) in [
            ("1 = foo", foo().equal_to(Datum::long(1))),
            ("1 != foo", foo().not_equal_to(Datum::long(1))),
            ("1 < foo", foo().greater_than(Datum::long(1))),
            ("1 <= foo", foo().greater_than_or_equal_to(Datum::long(1))),
            ("1 > foo", foo().less_than(Datum::long(1))),
            ("1 >= foo", foo().less_than_or_equal_to(Datum::long(1))),
        ] {
            assert_eq!(
                convert_to_iceberg_predicate(sql),
                Some(expected),
                "sql: {sql}"
            );
        }
    }

    #[test]
    fn test_predicate_conversion_with_boolean_column() {
        let b = || Reference::new("b");
        let foo = || Reference::new("foo");

        // A bare boolean column is a filter only on its own. A top-level AND is
        // split into separate filters, so `b` here is on its own.
        assert_eq!(
            convert_to_iceberg_predicate("b"),
            Some(b().equal_to(Datum::bool(true)))
        );
        assert_eq!(
            convert_to_iceberg_predicate("not b"),
            Some(b().equal_to(Datum::bool(false)))
        );
        assert_eq!(
            convert_to_iceberg_predicate("b and foo > 1"),
            Some(
                b().equal_to(Datum::bool(true))
                    .and(foo().greater_than(Datum::long(1)))
            )
        );

        // As an operand of OR or of a nested AND, it isn't converted.
        assert_eq!(convert_to_iceberg_predicate("b or foo > 1"), None);
        assert_eq!(
            convert_to_iceberg_predicate("(b and foo > 1) or foo < 0"),
            Some(
                foo()
                    .greater_than(Datum::long(1))
                    .or(foo().less_than(Datum::long(0)))
            )
        );
        assert_eq!(convert_to_iceberg_predicate("not (b and foo > 1)"), None);
    }

    #[test]
    fn test_predicate_conversion_with_and_condition() {
        let sql = "foo > 1 and bar = 'test'";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        let expected_predicate = Predicate::and(
            Reference::new("foo").greater_than(Datum::long(1)),
            Reference::new("bar").equal_to(Datum::string("test")),
        );
        assert_eq!(predicate, expected_predicate);
    }

    #[test]
    fn test_predicate_conversion_with_and_condition_unsupported() {
        let sql = "foo > 1 and length(bar) = 1";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        let expected_predicate = Reference::new("foo").greater_than(Datum::long(1));
        assert_eq!(predicate, expected_predicate);
    }

    #[test]
    fn test_predicate_conversion_with_and_condition_both_unsupported() {
        let sql = "foo in (1, 2, foo) and length(bar) = 1";
        let predicate = convert_to_iceberg_predicate(sql);
        assert_eq!(predicate, None);
    }

    #[test]
    fn test_predicate_conversion_with_or_condition_unsupported() {
        let sql = "foo > 1 or length(bar) = 1";
        let predicate = convert_to_iceberg_predicate(sql);
        assert_eq!(predicate, None);
    }

    #[test]
    fn test_predicate_conversion_with_or_condition_supported() {
        let sql = "foo > 1 or bar = 'test'";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        let expected_predicate = Predicate::or(
            Reference::new("foo").greater_than(Datum::long(1)),
            Reference::new("bar").equal_to(Datum::string("test")),
        );
        assert_eq!(predicate, expected_predicate);
    }

    #[test]
    fn test_predicate_conversion_with_complex_binary_expr() {
        let sql = "(foo > 1 and bar = 'test') or foo < 0 ";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();

        let inner_predicate = Predicate::and(
            Reference::new("foo").greater_than(Datum::long(1)),
            Reference::new("bar").equal_to(Datum::string("test")),
        );
        let expected_predicate = Predicate::or(
            inner_predicate,
            Reference::new("foo").less_than(Datum::long(0)),
        );
        assert_eq!(predicate, expected_predicate);
    }

    #[test]
    fn test_predicate_conversion_with_one_and_expr_supported() {
        let sql = "(foo > 1 and length(bar) = 1 ) or foo < 0 ";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();

        let inner_predicate = Reference::new("foo").greater_than(Datum::long(1));
        let expected_predicate = Predicate::or(
            inner_predicate,
            Reference::new("foo").less_than(Datum::long(0)),
        );
        assert_eq!(predicate, expected_predicate);
    }

    #[test]
    fn test_predicate_conversion_with_complex_binary_expr_unsupported() {
        let sql = "(foo > 1 or length(bar) = 1 ) and foo < 0 ";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        let expected_predicate = Reference::new("foo").less_than(Datum::long(0));
        assert_eq!(predicate, expected_predicate);
    }

    #[test]
    fn test_predicate_conversion_with_not_of_partial_and() {
        for sql in [
            "not (foo > 1 and length(bar) = 1)",
            "not ((foo > 1 and length(bar) = 1) or foo < 0)",
        ] {
            assert_eq!(convert_to_iceberg_predicate(sql), None, "sql: {sql}");
        }

        let predicate =
            convert_to_iceberg_predicate("not (foo > 1 and bar = 'test')").unwrap();
        let expected_predicate = !Predicate::and(
            Reference::new("foo").greater_than(Datum::long(1)),
            Reference::new("bar").equal_to(Datum::string("test")),
        );
        assert_eq!(predicate, expected_predicate);
    }

    #[test]
    fn test_predicate_conversion_with_cast() {
        for sql in [
            "ts >= timestamp '2023-01-05T00:00:00'",
            "ts >= date '2023-01-05T11:00:00'",
            "cast(foo as varchar) < '5'",
            "not isnan(cast(bar as double))",
        ] {
            assert_eq!(convert_to_iceberg_predicate(sql), None, "sql: {sql}");
        }
    }

    #[test]
    fn test_scalar_value_to_datum_timestamp() {
        use datafusion::common::ScalarValue;

        // Test TimestampMicrosecond - maps directly to Datum::timestamp_micros
        let ts_micros = 1672876800000000i64; // 2023-01-05 00:00:00 UTC in microseconds
        let datum = super::scalar_value_to_datum(&ScalarValue::TimestampMicrosecond(
            Some(ts_micros),
            None,
        ));
        assert_eq!(datum, Some(Datum::timestamp_micros(ts_micros)));

        // Test TimestampNanosecond - maps to Datum::timestamp_nanos to preserve precision
        let ts_nanos = 1672876800000000500i64; // 2023-01-05 00:00:00.000000500 UTC in nanoseconds
        let datum = super::scalar_value_to_datum(&ScalarValue::TimestampNanosecond(
            Some(ts_nanos),
            None,
        ));
        assert_eq!(datum, Some(Datum::timestamp_nanos(ts_nanos)));

        let datum = super::scalar_value_to_datum(&ScalarValue::TimestampMicrosecond(
            Some(ts_micros),
            Some("+01:00".into()),
        ));
        assert_eq!(datum, Some(Datum::timestamptz_micros(ts_micros)));

        let datum = super::scalar_value_to_datum(&ScalarValue::TimestampNanosecond(
            Some(ts_nanos),
            Some("UTC".into()),
        ));
        assert_eq!(datum, Some(Datum::timestamptz_nanos(ts_nanos)));

        // Test None timestamp
        let datum =
            super::scalar_value_to_datum(&ScalarValue::TimestampMicrosecond(None, None));
        assert_eq!(datum, None);

        // Note: TimestampSecond and TimestampMillisecond are not supported because
        // DataFusion's type coercion converts them to TimestampMicrosecond or TimestampNanosecond
        // before they reach scalar_value_to_datum in SQL queries.
        //
        // These return None (not pushed down):
        let ts_seconds = 1672876800i64; // 2023-01-05 00:00:00 UTC in seconds
        let datum = super::scalar_value_to_datum(&ScalarValue::TimestampSecond(
            Some(ts_seconds),
            None,
        ));
        assert_eq!(datum, None);

        let ts_millis = 1672876800000i64; // 2023-01-05 00:00:00 UTC in milliseconds
        let datum = super::scalar_value_to_datum(&ScalarValue::TimestampMillisecond(
            Some(ts_millis),
            None,
        ));
        assert_eq!(datum, None);
    }

    #[test]
    fn test_scalar_value_to_datum_binary() {
        use datafusion::common::ScalarValue;

        let bytes = vec![1u8, 2u8, 3u8];
        let datum =
            super::scalar_value_to_datum(&ScalarValue::Binary(Some(bytes.clone())));
        assert_eq!(datum, Some(Datum::binary(bytes.clone())));

        let datum =
            super::scalar_value_to_datum(&ScalarValue::LargeBinary(Some(bytes.clone())));
        assert_eq!(datum, Some(Datum::binary(bytes)));

        let datum = super::scalar_value_to_datum(&ScalarValue::Binary(None));
        assert_eq!(datum, None);
    }

    #[test]
    fn test_predicate_conversion_with_binary() {
        let sql = "foo = 1 and bar = X'0102'";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        // Binary literals are converted to Datum::binary
        // Note: SQL literal 1 is converted to Long by DataFusion
        let expected_predicate = Reference::new("foo")
            .equal_to(Datum::long(1))
            .and(Reference::new("bar").equal_to(Datum::binary(vec![1u8, 2u8])));
        assert_eq!(predicate, expected_predicate);
    }

    #[test]
    fn test_scalar_value_to_datum_boolean() {
        use datafusion::common::ScalarValue;

        // Test boolean true
        let datum = super::scalar_value_to_datum(&ScalarValue::Boolean(Some(true)));
        assert_eq!(datum, Some(Datum::bool(true)));

        // Test boolean false
        let datum = super::scalar_value_to_datum(&ScalarValue::Boolean(Some(false)));
        assert_eq!(datum, Some(Datum::bool(false)));

        // Test None boolean
        let datum = super::scalar_value_to_datum(&ScalarValue::Boolean(None));
        assert_eq!(datum, None);
    }

    #[test]
    fn test_predicate_conversion_with_like_starts_with() {
        let sql = "bar LIKE 'test%'";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        assert_eq!(
            predicate,
            Reference::new("bar").starts_with(Datum::string("test"))
        );
    }

    #[test]
    fn test_predicate_conversion_with_not_like_starts_with() {
        let sql = "bar NOT LIKE 'test%'";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        assert_eq!(
            predicate,
            Reference::new("bar").not_starts_with(Datum::string("test"))
        );
    }

    #[test]
    fn test_predicate_conversion_with_like_empty_prefix() {
        let sql = "bar LIKE '%'";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        assert_eq!(
            predicate,
            Reference::new("bar").starts_with(Datum::string(""))
        );
    }

    #[test]
    fn test_predicate_conversion_with_like_complex_pattern() {
        // Patterns with wildcards in the middle cannot be pushed down
        let sql = "bar LIKE 'te%st'";
        let predicate = convert_to_iceberg_predicate(sql);
        assert_eq!(predicate, None);
    }

    #[test]
    fn test_predicate_conversion_with_like_underscore_wildcard() {
        // Patterns with underscore wildcard cannot be pushed down
        let sql = "bar LIKE 'test_'";
        let predicate = convert_to_iceberg_predicate(sql);
        assert_eq!(predicate, None);
    }

    #[test]
    fn test_predicate_conversion_with_like_no_wildcard() {
        // Patterns without trailing % cannot be pushed down as StartsWith
        let sql = "bar LIKE 'test'";
        let predicate = convert_to_iceberg_predicate(sql);
        assert_eq!(predicate, None);
    }

    #[test]
    fn test_predicate_conversion_with_ilike() {
        // Case-insensitive LIKE (ILIKE) is not supported
        let sql = "bar ILIKE 'test%'";
        let predicate = convert_to_iceberg_predicate(sql);
        assert_eq!(predicate, None);
    }

    #[test]
    fn test_predicate_conversion_with_like_escape_clause() {
        // An explicit ESCAPE clause is not supported
        let sql = "bar LIKE 'test%' ESCAPE '#'";
        let predicate = convert_to_iceberg_predicate(sql);
        assert_eq!(predicate, None);
    }

    #[test]
    fn test_predicate_conversion_with_like_and_other_conditions() {
        let sql = "bar LIKE 'test%' AND foo > 1";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        let expected_predicate = Predicate::and(
            Reference::new("bar").starts_with(Datum::string("test")),
            Reference::new("foo").greater_than(Datum::long(1)),
        );
        assert_eq!(predicate, expected_predicate);
    }

    #[test]
    fn test_predicate_conversion_with_like_special_characters() {
        // Test LIKE with special characters in prefix
        let sql = "bar LIKE 'test-abc_123%'";
        let predicate = convert_to_iceberg_predicate(sql);
        // This should not be pushed down because it contains underscore
        assert_eq!(predicate, None);
    }

    #[test]
    fn test_predicate_conversion_with_like_unicode() {
        // Test LIKE with unicode characters in prefix
        let sql = "bar LIKE '测试%'";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        assert_eq!(
            predicate,
            Reference::new("bar").starts_with(Datum::string("测试"))
        );
    }

    #[test]
    fn test_predicate_conversion_with_like_escaped_trailing_percent() {
        // 'a\%' matches only the literal string "a%", so it is not a prefix pattern
        let sql = r"bar LIKE 'a\%'";
        let predicate = convert_to_iceberg_predicate(sql);
        assert_eq!(predicate, None);
    }

    #[test]
    fn test_predicate_conversion_with_like_escaped_wildcards_in_prefix() {
        let sql = r"bar LIKE 'a\%b\_c%'";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        assert_eq!(
            predicate,
            Reference::new("bar").starts_with(Datum::string("a%b_c"))
        );
    }

    #[test]
    fn test_predicate_conversion_with_like_escaped_backslash() {
        let sql = r"bar LIKE 'a\\%'";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        assert_eq!(
            predicate,
            Reference::new("bar").starts_with(Datum::string(r"a\"))
        );
    }

    #[test]
    fn test_predicate_conversion_with_like_escaped_regular_character() {
        // An escaped character that isn't a wildcard matches itself
        let sql = r"bar LIKE 'a\b%'";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        assert_eq!(
            predicate,
            Reference::new("bar").starts_with(Datum::string("ab"))
        );
    }

    #[test]
    fn test_predicate_conversion_with_not_like_escaped_prefix() {
        let sql = r"bar NOT LIKE 'a\%%'";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        assert_eq!(
            predicate,
            Reference::new("bar").not_starts_with(Datum::string("a%"))
        );
    }

    #[test]
    fn test_predicate_conversion_with_isnan() {
        let predicate = convert_to_iceberg_predicate("isnan(qux)").unwrap();
        assert_eq!(predicate, Reference::new("qux").is_nan());
    }

    #[test]
    fn test_predicate_conversion_with_not_isnan() {
        let predicate = convert_to_iceberg_predicate("NOT isnan(qux)").unwrap();
        assert_eq!(predicate, !Reference::new("qux").is_nan());
    }

    #[test]
    fn test_predicate_conversion_with_isnan_and_other_condition() {
        let sql = "isnan(qux) AND foo > 1";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        let expected_predicate = Predicate::and(
            Reference::new("qux").is_nan(),
            Reference::new("foo").greater_than(Datum::long(1)),
        );
        assert_eq!(predicate, expected_predicate);
    }

    #[test]
    fn test_predicate_conversion_with_isnan_negation() {
        // -x is NaN iff x is NaN
        let predicate = convert_to_iceberg_predicate("isnan(-qux)").unwrap();
        assert_eq!(predicate, Reference::new("qux").is_nan());

        let predicate = convert_to_iceberg_predicate("NOT isnan(-qux)").unwrap();
        assert_eq!(predicate, !Reference::new("qux").is_nan());
    }

    #[test]
    fn test_predicate_conversion_with_isnan_abs() {
        // abs(x) is NaN iff x is NaN
        let predicate = convert_to_iceberg_predicate("isnan(abs(qux))").unwrap();
        assert_eq!(predicate, Reference::new("qux").is_nan());
    }

    #[test]
    fn test_predicate_conversion_with_isnan_additive() {
        // x + c, c + x, x - c, c - x are NaN iff x is NaN (for finite c)
        for sql in [
            "isnan(qux + 1)",
            "isnan(1 + qux)",
            "isnan(qux - 1)",
            "isnan(1 - qux)",
            "isnan(qux + 1.5)",
        ] {
            let predicate = convert_to_iceberg_predicate(sql).unwrap();
            assert_eq!(predicate, Reference::new("qux").is_nan(), "sql: {sql}");
        }
    }

    #[test]
    fn test_predicate_conversion_with_isnan_multiplicative() {
        // x * c, c * x, x / c are NaN iff x is NaN (for finite non-zero c)
        for sql in ["isnan(qux * 2)", "isnan(2 * qux)", "isnan(qux / 2)"] {
            let predicate = convert_to_iceberg_predicate(sql).unwrap();
            assert_eq!(predicate, Reference::new("qux").is_nan(), "sql: {sql}");
        }
    }

    #[test]
    fn test_predicate_conversion_with_isnan_nested_expr() {
        // Nested NaN-preserving transformations resolve to the inner column
        let predicate =
            convert_to_iceberg_predicate("isnan(-(abs(qux) + 1) * 3)").unwrap();
        assert_eq!(predicate, Reference::new("qux").is_nan());
    }

    #[test]
    fn test_predicate_conversion_with_isnan_and_other_complex_condition() {
        let sql = "isnan(qux + 1) AND foo > 1";
        let predicate = convert_to_iceberg_predicate(sql).unwrap();
        let expected_predicate = Predicate::and(
            Reference::new("qux").is_nan(),
            Reference::new("foo").greater_than(Datum::long(1)),
        );
        assert_eq!(predicate, expected_predicate);
    }

    #[test]
    fn test_predicate_conversion_with_isnan_unsupported_arg() {
        // Multiplying/dividing by zero does not preserve NaN-ness: `x * 0` is NaN
        // when `x` is ±inf, so it cannot be pushed down.
        assert_eq!(convert_to_iceberg_predicate("isnan(qux * 0)"), None);
        assert_eq!(convert_to_iceberg_predicate("isnan(qux / 0)"), None);

        // `c / x` is not NaN-preserving (e.g. `0 / 0` is NaN while `0` is not).
        assert_eq!(convert_to_iceberg_predicate("isnan(1 / qux)"), None);

        // Expressions referencing more than one column cannot be reduced to a
        // single column reference.
        assert_eq!(convert_to_iceberg_predicate("isnan(qux + foo)"), None);

        // Unknown scalar functions are not pushed down.
        assert_eq!(convert_to_iceberg_predicate("isnan(sqrt(qux))"), None);
    }
}
