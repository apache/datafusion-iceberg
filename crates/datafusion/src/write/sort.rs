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

//! Partition-based sorting for Iceberg tables.

use std::sync::Arc;

use datafusion::common::plan_datafusion_err;
use datafusion::error::Result;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::sorts::sort::SortExec;
use iceberg::arrow::PROJECTED_PARTITION_VALUE_COLUMN;

/// Sorts each partition of a write's input by the `_partition` column added by
/// [`project_with_partition`](super::project_with_partition), as the clustered
/// writer requires.
pub(crate) fn sort_by_partition(
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let (index, _) = input
        .schema()
        .column_with_name(PROJECTED_PARTITION_VALUE_COLUMN)
        .ok_or_else(|| plan_datafusion_err!(
            "Partition column '{PROJECTED_PARTITION_VALUE_COLUMN}' not found in schema. Ensure the plan has been extended with partition values using project_with_partition."
        ))?;
    let column = Column::new(PROJECTED_PARTITION_VALUE_COLUMN, index);
    let ordering = LexOrdering::from([PhysicalSortExpr::new_default(Arc::new(column))]);
    Ok(Arc::new(
        SortExec::new(ordering, input).with_preserve_partitioning(true),
    ))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema};
    use datafusion::physical_plan::displayable;
    use datafusion::physical_plan::empty::EmptyExec;

    use super::*;

    /// An input with an `id` column, followed by `_partition` if `with_partition`,
    /// so the partition column's index isn't 0.
    fn input(with_partition: bool) -> Arc<dyn ExecutionPlan> {
        let mut fields = vec![Field::new("id", DataType::Int64, false)];
        if with_partition {
            fields.push(Field::new(
                PROJECTED_PARTITION_VALUE_COLUMN,
                DataType::Struct(Fields::empty()),
                false,
            ));
        }
        Arc::new(EmptyExec::new(Arc::new(Schema::new(fields))))
    }

    #[test]
    fn test_sort_by_partition_sorts_each_partition() {
        let plan = sort_by_partition(input(true)).unwrap();
        insta::assert_snapshot!(displayable(plan.as_ref()).indent(true), @r"
        SortExec: expr=[_partition@1 ASC], preserve_partitioning=[true]
          EmptyExec
        ");
    }

    #[test]
    fn test_sort_by_partition_requires_partition_column() {
        let error = sort_by_partition(input(false)).unwrap_err();
        insta::assert_snapshot!(error, @"Error during planning: Partition column '_partition' not found in schema. Ensure the plan has been extended with partition values using project_with_partition.");
    }
}
