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

    use datafusion::error::Result;
    use datafusion::prelude::SessionContext;
    use iceberg::TableIdent;
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    use iceberg_datafusion::{IcebergCatalogProvider, IcebergTableProviderFactory};
    use iceberg_datafusion_test_tools::{
        FixtureCatalog, IcebergTestHarness, create_taxi_table, display_schema,
    };

    #[tokio::test]
    async fn applies_metadata_table_projections() -> Result<()> {
        // Metadata tables resolve through the catalog, not the external table.
        let table = create_taxi_table().await?;
        let catalog = IcebergCatalogProvider::try_new(table.catalog()).await?;
        let harness = IcebergTestHarness::builder()
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_catalog("iceberg", Arc::new(catalog))
            .with_table(table)
            .build()
            .await?;

        let single_projected_column_result = harness
            .query("SELECT operation FROM iceberg.default.taxi$snapshots")
            .await?;
        insta::assert_snapshot!(single_projected_column_result.0, @"
           CooperativeExec
             DataSourceExec: format=iceberg_metadata
           ");
        insta::assert_snapshot!(single_projected_column_result.1, @"
           +-----------+
           | operation |
           +-----------+
           | append    |
           +-----------+
           ");

        let multiple_projected_columns_result = harness
            .query("SELECT operation, snapshot_id FROM iceberg.default.taxi$snapshots")
            .await?;

        insta::assert_snapshot!(multiple_projected_columns_result.0, @"
        ProjectionExec: expr=[operation@1 as operation, snapshot_id@0 as snapshot_id]
          CooperativeExec
            DataSourceExec: format=iceberg_metadata
        ");
        insta::assert_snapshot!(multiple_projected_columns_result.1, @r"
        +-----------+-------------+
        | operation | snapshot_id |
        +-----------+-------------+
        | append    | 1           |
        +-----------+-------------+
        ");

        let aggregated_columns_result = harness
            .query("SELECT COUNT(*) FROM iceberg.default.taxi$snapshots")
            .await?;

        insta::assert_snapshot!(aggregated_columns_result.0, @"
        ProjectionExec: expr=[count(Int64(1))@0 as count(*)]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                RepartitionExec: partitioning=RoundRobinBatch(16), input_partitions=1
                  DataSourceExec: format=iceberg_metadata
        ");
        insta::assert_snapshot!(aggregated_columns_result.1, @"
        +----------+
        | count(*) |
        +----------+
        | 1        |
        +----------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_metadata_table() -> Result<(), Box<dyn Error>> {
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
        let snapshots = ctx
            .sql("select * from catalog.ns.t1$snapshots")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        insta::assert_snapshot!(display_schema(&snapshots[0].schema()), @r#"
        Field { "committed_at": Timestamp(µs, "+00:00"), metadata: {"PARQUET:field_id": "1"} }
        Field { "snapshot_id": Int64, metadata: {"PARQUET:field_id": "2"} }
        Field { "parent_id": nullable Int64, metadata: {"PARQUET:field_id": "3"} }
        Field { "operation": nullable Utf8, metadata: {"PARQUET:field_id": "4"} }
        Field { "manifest_list": nullable Utf8, metadata: {"PARQUET:field_id": "5"} }
        Field { "summary": nullable Map("key_value": non-null Struct("key": non-null Utf8, metadata: {"PARQUET:field_id": "7"}, "value": Utf8, metadata: {"PARQUET:field_id": "8"}), unsorted), metadata: {"PARQUET:field_id": "6"} }
        "#);
        assert!(snapshots.iter().all(|batch| batch.num_rows() == 0));

        let manifests = ctx
            .sql("select * from catalog.ns.t1$manifests")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        insta::assert_snapshot!(display_schema(&manifests[0].schema()), @r#"
        Field { "content": Int32, metadata: {"PARQUET:field_id": "14"} }
        Field { "path": Utf8, metadata: {"PARQUET:field_id": "1"} }
        Field { "length": Int64, metadata: {"PARQUET:field_id": "2"} }
        Field { "partition_spec_id": Int32, metadata: {"PARQUET:field_id": "3"} }
        Field { "added_snapshot_id": Int64, metadata: {"PARQUET:field_id": "4"} }
        Field { "added_data_files_count": Int32, metadata: {"PARQUET:field_id": "5"} }
        Field { "existing_data_files_count": Int32, metadata: {"PARQUET:field_id": "6"} }
        Field { "deleted_data_files_count": Int32, metadata: {"PARQUET:field_id": "7"} }
        Field { "added_delete_files_count": Int32, metadata: {"PARQUET:field_id": "15"} }
        Field { "existing_delete_files_count": Int32, metadata: {"PARQUET:field_id": "16"} }
        Field { "deleted_delete_files_count": Int32, metadata: {"PARQUET:field_id": "17"} }
        Field { "partition_summaries": List(non-null Struct("contains_null": non-null Boolean, metadata: {"PARQUET:field_id": "10"}, "contains_nan": Boolean, metadata: {"PARQUET:field_id": "11"}, "lower_bound": Utf8, metadata: {"PARQUET:field_id": "12"}, "upper_bound": Utf8, metadata: {"PARQUET:field_id": "13"}), metadata: {"PARQUET:field_id": "9"}), metadata: {"PARQUET:field_id": "8"} }
        "#);
        assert!(manifests.iter().all(|batch| batch.num_rows() == 0));

        Ok(())
    }
}
