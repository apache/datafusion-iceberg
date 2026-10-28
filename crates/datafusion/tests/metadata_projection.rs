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
    use std::sync::Arc;

    use datafusion::error::Result;
    use iceberg_datafusion::{IcebergCatalogProvider, IcebergTableProviderFactory};
    use iceberg_datafusion_test_tools::{IcebergTestHarness, create_taxi_table};

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
}
