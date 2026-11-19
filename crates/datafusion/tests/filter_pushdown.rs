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
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    use iceberg_datafusion::{IcebergCatalogProvider, IcebergTableProviderFactory};
    use iceberg_datafusion_test_tools::{
        FixtureCatalog, IcebergTestHarness, create_taxi_table,
    };
    #[tokio::test]
    async fn pushes_down_compound_predicates() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query(
                r"SELECT COUNT(*) AS trips
               FROM taxi
               WHERE pickup_date = DATE '2024-01-10'
                 AND payment_type IN (1, 2)
                 AND trip_distance >= 2.0",
            )
            .await?;

        insta::assert_snapshot!(plan, @r"
        ProjectionExec: expr=[count(Int64(1))@0 as trips]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                FilterExec: pickup_date@2 = 2024-01-10 AND (payment_type@1 = 1 OR payment_type@1 = 2) AND trip_distance@0 >= 2, projection=[]
                  RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                    DataSourceExec: format=iceberg, projection=[trip_distance, payment_type, pickup_date], predicate=((pickup_date = 2024-01-10) AND ((payment_type = 1) OR (payment_type = 2))) AND (trip_distance >= 2)
        ");
        insta::assert_snapshot!(batches, @r"
        +-------+
        | trips |
        +-------+
        | 10659 |
        +-------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn pushes_down_disjunctions() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query(
                r"SELECT pickup_date, COUNT(*) AS trips
                   FROM taxi
                   WHERE pickup_date = DATE '2024-01-10'
                      OR pickup_date = DATE '2024-01-11'
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
                    FilterExec: pickup_date@0 = 2024-01-10 OR pickup_date@0 = 2024-01-11
                      RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                        DataSourceExec: format=iceberg, projection=[pickup_date], predicate=(pickup_date = 2024-01-10) OR (pickup_date = 2024-01-11)
        ");
        insta::assert_snapshot!(batches, @r"
    +-------------+-------+
    | pickup_date | trips |
    +-------------+-------+
    | 2024-01-10  | 25000 |
    | 2024-01-11  | 25000 |
    +-------------+-------+
    ");

        Ok(())
    }

    #[tokio::test]
    async fn pushes_down_null_predicates() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query("SELECT COUNT(*) AS trips FROM taxi WHERE passenger_count IS NULL")
            .await?;

        insta::assert_snapshot!(plan, @r"
        ProjectionExec: expr=[count(Int64(1))@0 as trips]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                FilterExec: passenger_count@0 IS NULL, projection=[]
                  RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                    DataSourceExec: format=iceberg, projection=[passenger_count], predicate=passenger_count IS NULL
        ");
        insta::assert_snapshot!(batches, @r"
        +-------+
        | trips |
        +-------+
        | 8886  |
        +-------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn retains_unsupported_filter_after_safe_pushdown() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query(
                r"SELECT COUNT(*) AS trips
                   FROM taxi
                   WHERE pickup_date = DATE '2024-01-10'
                     AND trip_distance + 1.0 > 3.0",
            )
            .await?;

        insta::assert_snapshot!(plan, @r"
        ProjectionExec: expr=[count(Int64(1))@0 as trips]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                FilterExec: pickup_date@1 = 2024-01-10 AND trip_distance@0 + 1 > 3, projection=[]
                  RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                    DataSourceExec: format=iceberg, projection=[trip_distance, pickup_date], predicate=pickup_date = 2024-01-10
        ");
        insta::assert_snapshot!(batches, @r"
        +-------+
        | trips |
        +-------+
        | 11688 |
        +-------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn executes_wholly_unsupported_filters_without_pushdown() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_table(create_taxi_table().await?)
            .query("SELECT COUNT(*) AS trips FROM taxi WHERE trip_distance + 1.0 > 20.0")
            .await?;

        insta::assert_snapshot!(plan, @r"
        ProjectionExec: expr=[count(Int64(1))@0 as trips]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                FilterExec: trip_distance@0 + 1 > 20, projection=[]
                  RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                    DataSourceExec: format=iceberg, projection=[trip_distance]
        ");
        insta::assert_snapshot!(batches, @r"
        +-------+
        | trips |
        +-------+
        | 4462  |
        +-------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_table_predict_pushdown() -> Result<(), Box<dyn Error>> {
        let fixtures = FixtureCatalog::try_new().await?;
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int))
                    .into(),
                NestedField::optional(2, "bar", Type::Primitive(PrimitiveType::String))
                    .into(),
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
        let records = ctx
            .sql("select * from catalog.ns.t1 where (foo > 1 and length(bar) = 1 ) or bar is null")
            .await
            .unwrap()
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
        let expected = "predicate=(foo > 1) OR (bar IS NULL)";
        assert!(s.value(1).trim().contains(expected));
        Ok(())
    }
}
