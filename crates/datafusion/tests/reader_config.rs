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

use std::sync::Arc;
use std::time::Duration;

use datafusion::arrow::array::{ArrayRef, Int64Array, RecordBatch};
use datafusion::error::Result;
use iceberg::TableIdent;
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg_datafusion::{IcebergDataFusionConfig, IcebergTableProviderFactory};
use iceberg_datafusion_test_tools::{
    FixtureCatalog, IcebergTestHarness, ReadCountingStorageFactory, create_taxi_table,
};
use parquet::file::properties::WriterProperties;

#[tokio::test]
async fn parquet_pruning_controls_row_group_reads() -> Result<()> {
    let fixtures = FixtureCatalog::try_new().await?;
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .unwrap();
    let mut table = fixtures
        .table(TableIdent::from_strs(["default", "items"]).unwrap(), schema)
        .create()
        .await?;
    // One file with four sorted row groups. Only the first group contains id 0.
    table
        .append()
        .writer_properties(
            WriterProperties::builder()
                .set_max_row_group_row_count(Some(100))
                .build(),
        )
        .file(RecordBatch::try_from_iter([(
            "id",
            Arc::new(Int64Array::from_iter_values(0..400)) as ArrayRef,
        )])?)
        .commit()
        .await?;
    let storage = ReadCountingStorageFactory::default();
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(
            IcebergTableProviderFactory::new_with_storage_factory(Arc::new(
                storage.clone(),
            )),
        )
        .with_table(table)
        .with_config_option("datafusion.execution.parquet.enable_page_index", false)
        .with_config_option("datafusion.execution.parquet.pruning", true)
        .build()
        .await?;

    let (_, pruned_rows) = harness.query("SELECT id FROM items WHERE id = 0").await?;
    let pruned_reads = storage.take_reads();
    harness
        .query("SET datafusion.execution.parquet.pruning = false")
        .await?;
    let (_, unpruned_rows) = harness.query("SELECT id FROM items WHERE id = 0").await?;
    let unpruned_reads = storage.take_reads();

    insta::assert_snapshot!(pruned_rows, @"
    +----+
    | id |
    +----+
    | 0  |
    +----+
    ");
    assert_eq!(pruned_rows, unpruned_rows);
    assert!(
        pruned_reads < unpruned_reads,
        "pruning read {pruned_reads} ranges; disabling it read {unpruned_reads}"
    );
    Ok(())
}

#[tokio::test]
async fn zero_file_concurrency_still_completes_the_scan() -> Result<()> {
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_table(create_taxi_table().await?)
        .with_session_extension(|mut state| {
            state
                .config()
                .get_or_insert_default()
                .options_mut()
                .extensions
                .insert(IcebergDataFusionConfig::default());
            state
        })
        .with_config_option("iceberg.execution.data_file_concurrency_limit", 0)
        .build()
        .await?;
    let (_, rows) = tokio::time::timeout(
        Duration::from_secs(10),
        harness.query("SELECT pickup_date FROM taxi LIMIT 1"),
    )
    .await
    .expect("zero file concurrency must not stall the reader")?;

    insta::assert_snapshot!(rows, @"
    +-------------+
    | pickup_date |
    +-------------+
    | 2024-01-08  |
    +-------------+
    ");
    Ok(())
}

#[tokio::test]
async fn typed_reader_options_override_session_batch_size() -> Result<()> {
    let options = iceberg::arrow::ArrowReaderOptions::builder()
        .with_batch_size(137)
        .build();
    let harness = IcebergTestHarness::builder()
        .with_iceberg_table_factory(IcebergTableProviderFactory::new())
        .with_session_extension(move |mut state| {
            state
                .config()
                .get_or_insert_default()
                .set_extension(Arc::new(options));
            state
        })
        .with_config_option("datafusion.execution.batch_size", 2000)
        .with_table(create_taxi_table().await?)
        .build()
        .await?;
    let (_, batches) = harness
        .query_batches("SELECT pickup_date FROM taxi")
        .await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).max(), Some(137));
    Ok(())
}
