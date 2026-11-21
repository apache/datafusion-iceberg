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

use std::num::NonZeroUsize;
use std::sync::Arc;

use datafusion::arrow::datatypes::Schema as ArrowSchema;
use datafusion::common::plan_err;
use datafusion::error::Result;
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};
use iceberg::arrow::PROJECTED_PARTITION_VALUE_COLUMN;
use iceberg::spec::{PartitionSpec, Transform};

/// Spreads the input of a write across `target_partitions` partitions.
///
/// For a partitioned table, the input must carry the `_partition` column added by
/// [`project_with_partition`](super::project_with_partition). If any partition field
/// uses an identity or bucket transform, rows are hashed on that column, so all rows
/// of a table partition reach the same writer. Otherwise batches are distributed
/// round-robin.
pub(crate) fn repartition(
    input: Arc<dyn ExecutionPlan>,
    partition_spec: &PartitionSpec,
    target_partitions: NonZeroUsize,
) -> Result<Arc<dyn ExecutionPlan>> {
    let partitioning =
        partitioning(&input.schema(), partition_spec, target_partitions.get())?;
    Ok(Arc::new(RepartitionExec::try_new(input, partitioning)?))
}

fn partitioning(
    input_schema: &ArrowSchema,
    partition_spec: &PartitionSpec,
    count: usize,
) -> Result<Partitioning> {
    if partition_spec.is_unpartitioned() {
        return Ok(Partitioning::RoundRobinBatch(count));
    }
    let Ok(index) = input_schema.index_of(PROJECTED_PARTITION_VALUE_COLUMN) else {
        return plan_err!(
            "Partitioned table input missing {PROJECTED_PARTITION_VALUE_COLUMN} column. \
             Ensure projection happens before repartitioning."
        );
    };
    let hashable = partition_spec.fields().iter().any(|field| {
        matches!(field.transform, Transform::Identity | Transform::Bucket(_))
    });
    if !hashable {
        return Ok(Partitioning::RoundRobinBatch(count));
    }
    let column = Column::new(PROJECTED_PARTITION_VALUE_COLUMN, index);
    Ok(Partitioning::Hash(vec![Arc::new(column)], count))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Fields};
    use datafusion::physical_plan::empty::EmptyExec;
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};

    use super::*;

    fn table_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long))
                    .into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String))
                    .into(),
                NestedField::required(3, "ts", Type::Primitive(PrimitiveType::Timestamp))
                    .into(),
            ])
            .build()
            .unwrap()
    }

    fn partition_spec(fields: &[(&str, Transform)]) -> PartitionSpec {
        let mut builder = PartitionSpec::builder(table_schema());
        for (source, transform) in fields {
            builder = builder
                .add_partition_field(source, format!("{source}_{transform}"), *transform)
                .unwrap();
        }
        builder.build().unwrap()
    }

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
        Arc::new(EmptyExec::new(Arc::new(ArrowSchema::new(fields))))
    }

    fn partitioning_of(
        input: Arc<dyn ExecutionPlan>,
        fields: &[(&str, Transform)],
    ) -> Result<String> {
        let plan = repartition(
            input,
            &partition_spec(fields),
            NonZeroUsize::new(4).unwrap(),
        )?;
        Ok(plan.properties().output_partitioning().to_string())
    }

    #[test]
    fn test_partitioning_by_transform() {
        let cases: &[&[(&str, Transform)]] = &[
            &[],
            &[("id", Transform::Identity)],
            &[("id", Transform::Bucket(8))],
            &[("name", Transform::Truncate(2))],
            &[("ts", Transform::Year)],
            &[("ts", Transform::Month)],
            &[("ts", Transform::Day)],
            &[("ts", Transform::Hour)],
            &[("ts", Transform::Day), ("id", Transform::Identity)],
        ];
        let lines = cases
            .iter()
            .map(|fields| {
                let spec = fields
                    .iter()
                    .map(|(source, transform)| format!("{transform}({source})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let partitioning = partitioning_of(input(true), fields).unwrap();
                format!("[{spec}] => {partitioning}")
            })
            .collect::<Vec<_>>()
            .join("\n");
        insta::assert_snapshot!(lines, @r"
        [] => RoundRobinBatch(4)
        [identity(id)] => Hash([_partition@1], 4)
        [bucket[8](id)] => Hash([_partition@1], 4)
        [truncate[2](name)] => RoundRobinBatch(4)
        [year(ts)] => RoundRobinBatch(4)
        [month(ts)] => RoundRobinBatch(4)
        [day(ts)] => RoundRobinBatch(4)
        [hour(ts)] => RoundRobinBatch(4)
        [day(ts), identity(id)] => Hash([_partition@1], 4)
        ");
    }

    #[test]
    fn test_partition_column_is_required_only_for_partitioned_tables() {
        insta::assert_snapshot!(
            partitioning_of(input(false), &[]).unwrap(),
            @"RoundRobinBatch(4)"
        );
        insta::assert_snapshot!(
            partitioning_of(input(false), &[("id", Transform::Identity)]).unwrap_err(),
            @"Error during planning: Partitioned table input missing _partition column. Ensure projection happens before repartitioning."
        );
    }
}
