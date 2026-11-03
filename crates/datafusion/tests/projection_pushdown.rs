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

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::sync::Arc;

    use datafusion::arrow::array::{Array, StringArray};
    use datafusion::error::Result;
    use datafusion::prelude::SessionContext;
    use iceberg::TableIdent;
    use iceberg::spec::{NestedField, PrimitiveType, Schema, StructType, Type};
    use iceberg_datafusion::{IcebergCatalogProvider, IcebergTableProviderFactory};
    use iceberg_datafusion_test_tools::{
        FixtureCatalog, IcebergTestHarness, create_taxi_table,
    };
    #[tokio::test]
    async fn projects_only_columns_required_by_the_query() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query(
                r"SELECT pickup_date, COUNT(*) AS trips
               FROM taxi
               WHERE pickup_date >= DATE '2024-01-10'
               GROUP BY pickup_date
               ORDER BY pickup_date",
            )
            .await?;

        insta::assert_snapshot!(plan, @r"
        SortPreservingMergeExec: [pickup_date@0 ASC NULLS LAST]
          ProjectionExec: expr=[pickup_date@0 as pickup_date, count(Int64(1))@1 as trips]
            SortExec: expr=[pickup_date@0 ASC NULLS LAST], preserve_partitioning=[true]
              AggregateExec: mode=FinalPartitioned, gby=[pickup_date@0 as pickup_date], aggr=[count(Int64(1))]
                RepartitionExec: partitioning=Hash([pickup_date@0], 16), input_partitions=16
                  AggregateExec: mode=Partial, gby=[pickup_date@0 as pickup_date], aggr=[count(Int64(1))]
                    FilterExec: pickup_date@0 >= 2024-01-10
                      RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                        DataSourceExec: format=iceberg, projection=[pickup_date], predicate=pickup_date >= 2024-01-10
        ");
        insta::assert_snapshot!(batches, @r"
    +-------------+-------+
    | pickup_date | trips |
    +-------------+-------+
    | 2024-01-10  | 25000 |
    | 2024-01-11  | 25000 |
    | 2024-01-12  | 25000 |
    | 2024-01-13  | 25000 |
    | 2024-01-14  | 25000 |
    +-------------+-------+
    ");

        Ok(())
    }

    #[tokio::test]
    async fn includes_filter_columns_that_are_not_in_the_output() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query(
                r"SELECT vendor_id, pickup_location_id
                   FROM taxi
                   WHERE pickup_date = DATE '2024-01-10'
                   ORDER BY pickup_at, vendor_id
                   LIMIT 3",
            )
            .await?;

        insta::assert_snapshot!(plan, @r"
        ProjectionExec: expr=[vendor_id@0 as vendor_id, pickup_location_id@1 as pickup_location_id]
          SortPreservingMergeExec: [pickup_at@2 ASC NULLS LAST, vendor_id@0 ASC NULLS LAST], fetch=3
            SortExec: TopK(fetch=3), expr=[pickup_at@2 ASC NULLS LAST, vendor_id@0 ASC NULLS LAST], preserve_partitioning=[true]
              FilterExec: pickup_date@3 = 2024-01-10, projection=[vendor_id@0, pickup_location_id@2, pickup_at@1]
                RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                  DataSourceExec: format=iceberg, projection=[vendor_id, pickup_at, pickup_location_id, pickup_date], predicate=pickup_date = 2024-01-10
        ");
        insta::assert_snapshot!(batches, @r"
        +-----------+--------------------+
        | vendor_id | pickup_location_id |
        +-----------+--------------------+
        | 1         | 93                 |
        | 2         | 228                |
        | 2         | 8                  |
        +-----------+--------------------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn projects_source_columns_for_computed_expressions() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query(
                r"SELECT MAX(trip_distance * fare_amount) AS max_weighted_fare
                   FROM taxi
                   WHERE pickup_date = DATE '2024-01-10'",
            )
            .await?;

        insta::assert_snapshot!(plan, @r"
        ProjectionExec: expr=[max(taxi.trip_distance * taxi.fare_amount)@0 as max_weighted_fare]
          AggregateExec: mode=Final, gby=[], aggr=[max(taxi.trip_distance * taxi.fare_amount)]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[max(taxi.trip_distance * taxi.fare_amount)]
                FilterExec: pickup_date@2 = 2024-01-10, projection=[trip_distance@0, fare_amount@1]
                  RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                    DataSourceExec: format=iceberg, projection=[trip_distance, fare_amount, pickup_date], predicate=pickup_date = 2024-01-10
        ");
        insta::assert_snapshot!(batches, @r"
        +-------------------+
        | max_weighted_fare |
        +-------------------+
        | 5389.1892         |
        +-------------------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn reads_all_columns_when_selecting_star() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query(
                r"SELECT *
                   FROM taxi
                   WHERE pickup_date = DATE '2024-01-10'
                   ORDER BY pickup_at, vendor_id, pickup_location_id
                   LIMIT 1",
            )
            .await?;

        insta::assert_snapshot!(plan, @r"
        SortPreservingMergeExec: [pickup_at@1 ASC NULLS LAST, vendor_id@0 ASC NULLS LAST, pickup_location_id@5 ASC NULLS LAST], fetch=1
          SortExec: TopK(fetch=1), expr=[pickup_at@1 ASC NULLS LAST, vendor_id@0 ASC NULLS LAST, pickup_location_id@5 ASC NULLS LAST], preserve_partitioning=[true]
            FilterExec: pickup_date@12 = 2024-01-10
              RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                DataSourceExec: format=iceberg, projection=[vendor_id, pickup_at, dropoff_at, passenger_count, trip_distance, pickup_location_id, dropoff_location_id, payment_type, fare_amount, tip_amount, tolls_amount, total_amount, pickup_date], predicate=pickup_date = 2024-01-10
        ");
        insta::assert_snapshot!(batches, @r"
        +-----------+---------------------+---------------------+-----------------+---------------+--------------------+---------------------+--------------+-------------+------------+--------------+--------------+-------------+
        | vendor_id | pickup_at           | dropoff_at          | passenger_count | trip_distance | pickup_location_id | dropoff_location_id | payment_type | fare_amount | tip_amount | tolls_amount | total_amount | pickup_date |
        +-----------+---------------------+---------------------+-----------------+---------------+--------------------+---------------------+--------------+-------------+------------+--------------+--------------+-------------+
        | 1         | 2024-01-10T00:00:04 | 2024-01-10T00:03:44 | 1               | 0.68          | 93                 | 237                 | 1            | 6.17        | 1.05       | 0.0          | 10.72        | 2024-01-10  |
        +-----------+---------------------+---------------------+-----------------+---------------+--------------------+---------------------+--------------+-------------+------------+--------------+--------------+-------------+
        ");

        Ok(())
    }

    fn get_struct_type() -> StructType {
        StructType::new(vec![
            NestedField::required(4, "s_foo1", Type::Primitive(PrimitiveType::Int))
                .into(),
            NestedField::required(5, "s_foo2", Type::Primitive(PrimitiveType::String))
                .into(),
        ])
    }

    #[tokio::test]
    async fn test_table_projection() -> Result<(), Box<dyn Error>> {
        let fixtures = FixtureCatalog::try_new().await?;
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int))
                    .into(),
                NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String))
                    .into(),
                NestedField::optional(3, "foo3", Type::Struct(get_struct_type())).into(),
            ])
            .build()?;
        fixtures
            .table(TableIdent::from_strs(["ns", "t1"])?, schema)
            .create()
            .await?;
        let ctx = SessionContext::new();
        ctx.register_catalog(
            "catalog",
            Arc::new(IcebergCatalogProvider::try_new(fixtures.catalog()).await?),
        );
        let table_df = ctx.table("catalog.ns.t1").await.unwrap();

        let records = table_df
            .clone()
            .explain(false, false)
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(1, records.len());
        let record = &records[0];
        // the first column is plan_type, the second column plan string.
        let s = record
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(2, s.len());
        // the first row is logical_plan, the second row is physical_plan
        assert!(
            s.value(1).contains(
                "DataSourceExec: format=iceberg, projection=[foo1, foo2, foo3]"
            )
        );

        // datafusion doesn't support query foo3.s_foo1, use foo3 instead
        let records = table_df
            .select_columns(&["foo1", "foo3"])
            .unwrap()
            .explain(false, false)
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(1, records.len());
        let record = &records[0];
        let s = record
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(2, s.len());
        assert!(
            s.value(1)
                .contains("DataSourceExec: format=iceberg, projection=[foo1, foo3]")
        );

        Ok(())
    }
}
