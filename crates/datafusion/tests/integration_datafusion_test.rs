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

//! Integration tests for Iceberg Datafusion with Hive Metastore.

use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::sync::Arc;
use std::vec;

use datafusion::arrow::array::{
    Array, AsArray, Int32Array, Int64Array, ListArray, RecordBatch, StringArray,
    StructArray, UInt64Array,
};
use datafusion::arrow::buffer::OffsetBuffer;
use datafusion::arrow::compute::{
    cast, concat_batches, sort_to_indices, take_record_batch,
};
use datafusion::arrow::datatypes::{
    DataType, Field, Int32Type, Int64Type, Schema as ArrowSchema,
};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::TableProvider;
use datafusion::execution::context::SessionContext;
use datafusion::parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::common::collect;
use datafusion::physical_plan::joins::SortMergeJoinExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{ExecutionPlan, displayable};
use datafusion::prelude::SessionConfig;
use datafusion_iceberg::physical_plan::{
    IcebergCommitExec, IcebergMetadataScan, IcebergTableScan, IcebergWriteExec,
};
use datafusion_iceberg::{
    IcebergCatalogProvider, IcebergDataFusionConfig, IcebergMetadataTableProvider,
    IcebergStaticTableProvider, IcebergTableProvider,
};
use expect_test::expect;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::scan::{FileScanTask, FileScanTaskDeleteFile};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, ListType, NestedField, NullOrder,
    PrimitiveType, Schema, SortDirection, SortField, SortOrder, Struct, StructType,
    Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::test_utils::check_record_batches;
use iceberg::transaction::{AddColumn, ApplyTransactionAction, Transaction};
use iceberg::{
    Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, Result as IcebergResult,
    TableCreation, TableIdent,
};
use tempfile::TempDir;

fn temp_path() -> String {
    let temp_dir = TempDir::new().unwrap();
    temp_dir.path().to_str().unwrap().to_string()
}

async fn get_iceberg_catalog() -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), temp_path())]),
        )
        .await
        .unwrap()
}

fn get_struct_type() -> StructType {
    StructType::new(vec![
        NestedField::required(4, "s_foo1", Type::Primitive(PrimitiveType::Int)).into(),
        NestedField::required(5, "s_foo2", Type::Primitive(PrimitiveType::String)).into(),
    ])
}

async fn set_test_namespace(
    catalog: &MemoryCatalog,
    namespace: &NamespaceIdent,
) -> IcebergResult<()> {
    let properties = HashMap::new();

    catalog.create_namespace(namespace, properties).await?;

    Ok(())
}

fn get_table_creation(
    location: impl ToString,
    name: impl ToString,
    schema: Option<Schema>,
) -> IcebergResult<TableCreation> {
    let schema = match schema {
        None => Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int))
                    .into(),
                NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String))
                    .into(),
            ])
            .build()?,
        Some(schema) => schema,
    };

    let creation = TableCreation::builder()
        .location(location.to_string())
        .name(name.to_string())
        .properties(HashMap::new())
        .schema(schema)
        .build();

    Ok(creation)
}

#[tokio::test]
async fn test_provider_plan_stream_schema() -> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_provider_get_table_schema".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    let provider = ctx.catalog("catalog").unwrap();
    let schema = provider.schema("test_provider_get_table_schema").unwrap();

    let table = schema.table("my_table").await.unwrap().unwrap();
    let table_schema = table.schema();

    let expected = [("foo1", &DataType::Int32), ("foo2", &DataType::Utf8)];

    for (field, exp) in table_schema.fields().iter().zip(expected.iter()) {
        assert_eq!(field.name(), exp.0);
        assert_eq!(field.data_type(), exp.1);
        assert!(!field.is_nullable())
    }

    let df = ctx
        .sql("select foo2 from catalog.test_provider_get_table_schema.my_table")
        .await
        .unwrap();

    let task_ctx = Arc::new(df.task_ctx());
    let plan = df.create_physical_plan().await.unwrap();
    let stream = plan.execute(1, task_ctx).unwrap();

    // Ensure both the plan and the stream conform to the same schema
    assert_eq!(plan.schema(), stream.schema());
    assert_eq!(
        stream.schema().as_ref(),
        &ArrowSchema::new(vec![
            Field::new("foo2", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )]))
        ]),
    );

    Ok(())
}

#[tokio::test]
async fn test_provider_list_table_names() -> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_provider_list_table_names".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    let provider = ctx.catalog("catalog").unwrap();
    let schema = provider.schema("test_provider_list_table_names").unwrap();

    let result = schema.table_names();

    expect![[r#"
        [
            "my_table",
            "my_table$snapshots",
            "my_table$manifests",
            "my_table$history",
        ]
    "#]]
    .assert_debug_eq(&result);

    Ok(())
}

#[tokio::test]
async fn test_provider_list_schema_names() -> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_provider_list_schema_names".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    let provider = ctx.catalog("catalog").unwrap();

    let expected = ["test_provider_list_schema_names"];
    let result = provider.schema_names();

    assert!(
        expected
            .iter()
            .all(|item| result.contains(&item.to_string()))
    );
    Ok(())
}

#[tokio::test]
async fn test_table_projection() -> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("ns".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String))
                .into(),
            NestedField::optional(3, "foo3", Type::Struct(get_struct_type())).into(),
        ])
        .build()?;
    let creation = get_table_creation(temp_path(), "t1", Some(schema))?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);
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
    assert!(s.value(1).contains("projection:[foo1,foo2,foo3]"));

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
            .contains("IcebergTableScan projection:[foo1,foo3]")
    );

    Ok(())
}

#[tokio::test]
async fn test_table_predict_pushdown() -> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("ns".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "bar", Type::Primitive(PrimitiveType::String))
                .into(),
        ])
        .build()?;
    let creation = get_table_creation(temp_path(), "t1", Some(schema))?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);
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
    let expected = "predicate:[(foo > 1) OR (bar IS NULL)]";
    assert!(s.value(1).trim().contains(expected));
    Ok(())
}

#[tokio::test]
async fn test_metadata_table() -> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("ns".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "bar", Type::Primitive(PrimitiveType::String))
                .into(),
        ])
        .build()?;
    let creation = get_table_creation(temp_path(), "t1", Some(schema))?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);
    let snapshots = ctx
        .sql("select * from catalog.ns.t1$snapshots")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    check_record_batches(
        snapshots,
        expect![[r#"
            Field { "committed_at": Timestamp(µs, "+00:00"), metadata: {"PARQUET:field_id": "1"} },
            Field { "snapshot_id": Int64, metadata: {"PARQUET:field_id": "2"} },
            Field { "parent_id": nullable Int64, metadata: {"PARQUET:field_id": "3"} },
            Field { "operation": nullable Utf8, metadata: {"PARQUET:field_id": "4"} },
            Field { "manifest_list": nullable Utf8, metadata: {"PARQUET:field_id": "5"} },
            Field { "summary": nullable Map("key_value": non-null Struct("key": non-null Utf8, metadata: {"PARQUET:field_id": "7"}, "value": Utf8, metadata: {"PARQUET:field_id": "8"}), unsorted), metadata: {"PARQUET:field_id": "6"} }"#]],
        expect![[r#"
            committed_at: PrimitiveArray<Timestamp(µs, "+00:00")>
            [
            ],
            snapshot_id: PrimitiveArray<Int64>
            [
            ],
            parent_id: PrimitiveArray<Int64>
            [
            ],
            operation: StringArray
            [
            ],
            manifest_list: StringArray
            [
            ],
            summary: MapArray
            [
            ]"#]],
        &[],
        None,
    );

    let manifests = ctx
        .sql("select * from catalog.ns.t1$manifests")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    check_record_batches(
        manifests,
        expect![[r#"
            Field { "content": Int32, metadata: {"PARQUET:field_id": "14"} },
            Field { "path": Utf8, metadata: {"PARQUET:field_id": "1"} },
            Field { "length": Int64, metadata: {"PARQUET:field_id": "2"} },
            Field { "partition_spec_id": Int32, metadata: {"PARQUET:field_id": "3"} },
            Field { "added_snapshot_id": Int64, metadata: {"PARQUET:field_id": "4"} },
            Field { "added_data_files_count": Int32, metadata: {"PARQUET:field_id": "5"} },
            Field { "existing_data_files_count": Int32, metadata: {"PARQUET:field_id": "6"} },
            Field { "deleted_data_files_count": Int32, metadata: {"PARQUET:field_id": "7"} },
            Field { "added_delete_files_count": Int32, metadata: {"PARQUET:field_id": "15"} },
            Field { "existing_delete_files_count": Int32, metadata: {"PARQUET:field_id": "16"} },
            Field { "deleted_delete_files_count": Int32, metadata: {"PARQUET:field_id": "17"} },
            Field { "partition_summaries": List(non-null Struct("contains_null": non-null Boolean, metadata: {"PARQUET:field_id": "10"}, "contains_nan": Boolean, metadata: {"PARQUET:field_id": "11"}, "lower_bound": Utf8, metadata: {"PARQUET:field_id": "12"}, "upper_bound": Utf8, metadata: {"PARQUET:field_id": "13"}), metadata: {"PARQUET:field_id": "9"}), metadata: {"PARQUET:field_id": "8"} }"#]],
        expect![[r#"
            content: PrimitiveArray<Int32>
            [
            ],
            path: StringArray
            [
            ],
            length: PrimitiveArray<Int64>
            [
            ],
            partition_spec_id: PrimitiveArray<Int32>
            [
            ],
            added_snapshot_id: PrimitiveArray<Int64>
            [
            ],
            added_data_files_count: PrimitiveArray<Int32>
            [
            ],
            existing_data_files_count: PrimitiveArray<Int32>
            [
            ],
            deleted_data_files_count: PrimitiveArray<Int32>
            [
            ],
            added_delete_files_count: PrimitiveArray<Int32>
            [
            ],
            existing_delete_files_count: PrimitiveArray<Int32>
            [
            ],
            deleted_delete_files_count: PrimitiveArray<Int32>
            [
            ],
            partition_summaries: ListArray
            [
            ]"#]],
        &[],
        None,
    );

    Ok(())
}

#[tokio::test]
async fn test_insert_into() -> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_insert_into".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Verify table schema
    let provider = ctx.catalog("catalog").unwrap();
    let schema = provider.schema("test_insert_into").unwrap();
    let table = schema.table("my_table").await.unwrap().unwrap();
    let table_schema = table.schema();

    let expected = [("foo1", &DataType::Int32), ("foo2", &DataType::Utf8)];
    for (field, exp) in table_schema.fields().iter().zip(expected.iter()) {
        assert_eq!(field.name(), exp.0);
        assert_eq!(field.data_type(), exp.1);
        assert!(!field.is_nullable())
    }

    // Insert data into the table
    let df = ctx
        .sql("INSERT INTO catalog.test_insert_into.my_table VALUES (1, 'alan'), (2, 'turing')")
        .await
        .unwrap();

    // Verify the insert operation result
    let batches = df.collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert!(
        batch.num_rows() == 1 && batch.num_columns() == 1,
        "Results should only have one row and one column that has the number of rows inserted"
    );
    // Verify the number of rows inserted
    let rows_inserted = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(rows_inserted.value(0), 2);

    // Query the table to verify the inserted data
    let df = ctx
        .sql("SELECT * FROM catalog.test_insert_into.my_table")
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();

    // Use check_record_batches to verify the data
    check_record_batches(
        batches,
        expect![[r#"
            Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }"#]],
        expect![[r#"
            foo1: PrimitiveArray<Int32>
            [
              1,
              2,
            ],
            foo2: StringArray
            [
              "alan",
              "turing",
            ]"#]],
        &[],
        Some("foo1"),
    );

    Ok(())
}

fn get_nested_struct_type() -> StructType {
    // Create a nested struct type with:
    // - address: STRUCT<street: STRING, city: STRING, zip: INT>
    // - contact: STRUCT<email: STRING, phone: STRING>
    StructType::new(vec![
        NestedField::optional(
            10,
            "address",
            Type::Struct(StructType::new(vec![
                NestedField::optional(
                    11,
                    "street",
                    Type::Primitive(PrimitiveType::String),
                )
                .into(),
                NestedField::optional(12, "city", Type::Primitive(PrimitiveType::String))
                    .into(),
                NestedField::optional(13, "zip", Type::Primitive(PrimitiveType::Int))
                    .into(),
            ])),
        )
        .into(),
        NestedField::optional(
            20,
            "contact",
            Type::Struct(StructType::new(vec![
                NestedField::optional(
                    21,
                    "email",
                    Type::Primitive(PrimitiveType::String),
                )
                .into(),
                NestedField::optional(
                    22,
                    "phone",
                    Type::Primitive(PrimitiveType::String),
                )
                .into(),
            ])),
        )
        .into(),
    ])
}

#[tokio::test]
async fn test_insert_into_nested() -> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_insert_nested".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let table_name = "nested_table";

    // Create a schema with nested fields
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String))
                .into(),
            NestedField::optional(3, "profile", Type::Struct(get_nested_struct_type()))
                .into(),
        ])
        .build()?;

    // Create the table with the nested schema
    let creation = get_table_creation(temp_path(), table_name, Some(schema))?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Verify table schema
    let provider = ctx.catalog("catalog").unwrap();
    let schema = provider.schema("test_insert_nested").unwrap();
    let table = schema.table("nested_table").await.unwrap().unwrap();
    let table_schema = table.schema();

    // Verify the schema has the expected structure
    assert_eq!(table_schema.fields().len(), 3);
    assert_eq!(table_schema.field(0).name(), "id");
    assert_eq!(table_schema.field(1).name(), "name");
    assert_eq!(table_schema.field(2).name(), "profile");
    assert!(matches!(
        table_schema.field(2).data_type(),
        DataType::Struct(_)
    ));

    // In DataFusion, we need to use named_struct to create struct values
    // Insert data with nested structs
    let insert_sql = r#"
    INSERT INTO catalog.test_insert_nested.nested_table
    SELECT 
        1 as id, 
        'Alice' as name,
        named_struct(
            'address', named_struct(
                'street', '123 Main St',
                'city', 'San Francisco',
                'zip', 94105
            ),
            'contact', named_struct(
                'email', 'alice@example.com',
                'phone', '555-1234'
            )
        ) as profile
    UNION ALL
    SELECT 
        2 as id, 
        'Bob' as name,
        named_struct(
            'address', named_struct(
                'street', '456 Market St',
                'city', 'San Jose',
                'zip', 95113
            ),
            'contact', named_struct(
                'email', 'bob@example.com',
                'phone', NULL
            )
        ) as profile
    "#;

    // Execute the insert
    let df = ctx.sql(insert_sql).await.unwrap();
    let batches = df.collect().await.unwrap();

    // Verify the insert operation result
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert!(batch.num_rows() == 1 && batch.num_columns() == 1);

    // Verify the number of rows inserted
    let rows_inserted = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(rows_inserted.value(0), 2);

    // Query the table to verify the inserted data
    let df = ctx
        .sql("SELECT * FROM catalog.test_insert_nested.nested_table ORDER BY id")
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();

    // Use check_record_batches to verify the data
    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "name": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "profile": nullable Struct("address": Struct("street": Utf8, metadata: {"PARQUET:field_id": "6"}, "city": Utf8, metadata: {"PARQUET:field_id": "7"}, "zip": Int32, metadata: {"PARQUET:field_id": "8"}), metadata: {"PARQUET:field_id": "4"}, "contact": Struct("email": Utf8, metadata: {"PARQUET:field_id": "9"}, "phone": Utf8, metadata: {"PARQUET:field_id": "10"}), metadata: {"PARQUET:field_id": "5"}), metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
            ],
            name: StringArray
            [
              "Alice",
              "Bob",
            ],
            profile: StructArray
            -- validity:
            [
              valid,
              valid,
            ]
            [
            -- child 0: "address" (Struct([Field { name: "street", data_type: Utf8, nullable: true, metadata: {"PARQUET:field_id": "6"} }, Field { name: "city", data_type: Utf8, nullable: true, metadata: {"PARQUET:field_id": "7"} }, Field { name: "zip", data_type: Int32, nullable: true, metadata: {"PARQUET:field_id": "8"} }]))
            StructArray
            -- validity:
            [
              valid,
              valid,
            ]
            [
            -- child 0: "street" (Utf8)
            StringArray
            [
              "123 Main St",
              "456 Market St",
            ]
            -- child 1: "city" (Utf8)
            StringArray
            [
              "San Francisco",
              "San Jose",
            ]
            -- child 2: "zip" (Int32)
            PrimitiveArray<Int32>
            [
              94105,
              95113,
            ]
            ]
            -- child 1: "contact" (Struct([Field { name: "email", data_type: Utf8, nullable: true, metadata: {"PARQUET:field_id": "9"} }, Field { name: "phone", data_type: Utf8, nullable: true, metadata: {"PARQUET:field_id": "10"} }]))
            StructArray
            -- validity:
            [
              valid,
              valid,
            ]
            [
            -- child 0: "email" (Utf8)
            StringArray
            [
              "alice@example.com",
              "bob@example.com",
            ]
            -- child 1: "phone" (Utf8)
            StringArray
            [
              "555-1234",
              null,
            ]
            ]
            ]"#]],
        &[],
        Some("id"),
    );

    // Query with explicit field access to verify nested data
    let df = ctx
        .sql(
            r#"
            SELECT 
                id, 
                name,
                profile.address.street,
                profile.address.city,
                profile.address.zip,
                profile.contact.email,
                profile.contact.phone
            FROM catalog.test_insert_nested.nested_table 
            ORDER BY id
        "#,
        )
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();

    // Use check_record_batches to verify the flattened data
    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "name": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "catalog.test_insert_nested.nested_table.profile[address][street]": nullable Utf8, metadata: {"PARQUET:field_id": "6"} },
            Field { "catalog.test_insert_nested.nested_table.profile[address][city]": nullable Utf8, metadata: {"PARQUET:field_id": "7"} },
            Field { "catalog.test_insert_nested.nested_table.profile[address][zip]": nullable Int32, metadata: {"PARQUET:field_id": "8"} },
            Field { "catalog.test_insert_nested.nested_table.profile[contact][email]": nullable Utf8, metadata: {"PARQUET:field_id": "9"} },
            Field { "catalog.test_insert_nested.nested_table.profile[contact][phone]": nullable Utf8, metadata: {"PARQUET:field_id": "10"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
            ],
            name: StringArray
            [
              "Alice",
              "Bob",
            ],
            catalog.test_insert_nested.nested_table.profile[address][street]: StringArray
            [
              "123 Main St",
              "456 Market St",
            ],
            catalog.test_insert_nested.nested_table.profile[address][city]: StringArray
            [
              "San Francisco",
              "San Jose",
            ],
            catalog.test_insert_nested.nested_table.profile[address][zip]: PrimitiveArray<Int32>
            [
              94105,
              95113,
            ],
            catalog.test_insert_nested.nested_table.profile[contact][email]: StringArray
            [
              "alice@example.com",
              "bob@example.com",
            ],
            catalog.test_insert_nested.nested_table.profile[contact][phone]: StringArray
            [
              "555-1234",
              null,
            ]"#]],
        &[],
        Some("id"),
    );

    Ok(())
}

#[tokio::test]
async fn test_insert_into_partitioned() -> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_partitioned_write".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    // Create a schema with a partition column
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String))
                .into(),
            NestedField::required(3, "value", Type::Primitive(PrimitiveType::String))
                .into(),
        ])
        .build()?;

    // Create partition spec with identity transform on category
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category", Transform::Identity)?
        .build();

    // Create the partitioned table
    let creation = TableCreation::builder()
        .name("partitioned_table".to_string())
        .location(temp_path())
        .schema(schema)
        .partition_spec(partition_spec)
        .properties(HashMap::new())
        .build();

    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Insert data with multiple partition values in a single batch
    let df = ctx
        .sql(
            r#"
            INSERT INTO catalog.test_partitioned_write.partitioned_table 
            VALUES 
                (1, 'electronics', 'laptop'),
                (2, 'electronics', 'phone'),
                (3, 'books', 'novel'),
                (4, 'books', 'textbook'),
                (5, 'clothing', 'shirt')
            "#,
        )
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    let rows_inserted = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(rows_inserted.value(0), 5);

    // Query the table to verify data
    let df = ctx
        .sql("SELECT * FROM catalog.test_partitioned_write.partitioned_table ORDER BY id")
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();

    // Verify the data - note that _partition column should NOT be present
    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
              3,
              4,
              5,
            ],
            category: StringArray
            [
              "electronics",
              "electronics",
              "books",
              "books",
              "clothing",
            ],
            value: StringArray
            [
              "laptop",
              "phone",
              "novel",
              "textbook",
              "shirt",
            ]"#]],
        &[],
        Some("id"),
    );

    // Verify that data files exist under correct partition paths
    let table_ident = TableIdent::new(namespace.clone(), "partitioned_table".to_string());
    let table = client.load_table(&table_ident).await?;
    let table_location = table.metadata().location();
    let file_io = table.file_io();

    // List files under each expected partition path
    let electronics_path = format!("{table_location}/data/category=electronics");
    let books_path = format!("{table_location}/data/category=books");
    let clothing_path = format!("{table_location}/data/category=clothing");

    // Verify partition directories exist and contain data files
    assert!(
        file_io.exists(&electronics_path).await?,
        "Expected partition directory: {electronics_path}"
    );
    assert!(
        file_io.exists(&books_path).await?,
        "Expected partition directory: {books_path}"
    );
    assert!(
        file_io.exists(&clothing_path).await?,
        "Expected partition directory: {clothing_path}"
    );

    Ok(())
}

/// Executes `plan`, which must have a single partition, and returns its rows.
async fn run_batches(
    plan: &dyn ExecutionPlan,
    ctx: &SessionContext,
) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
    assert_eq!(plan.properties().partitioning.partition_count(), 1);
    let stream = plan.execute(0, ctx.task_ctx())?;
    Ok(collect(stream).await?)
}

/// Executes `plan`, which must have a single partition, and renders its rows
/// as a table.
async fn run(
    plan: &dyn ExecutionPlan,
    ctx: &SessionContext,
) -> Result<String, Box<dyn Error>> {
    Ok(pretty_format_batches(&run_batches(plan, ctx).await?)?.to_string())
}

/// Rebuilds `scan` from its accessors alone, as a codec would.
fn rebuild_scan(scan: &IcebergTableScan) -> IcebergTableScan {
    IcebergTableScan::new_with_predicate(
        scan.table().clone(),
        scan.snapshot_id(),
        scan.schema(),
        scan.predicates().cloned(),
        scan.limit(),
    )
}

/// Returns the first node of type `T` in `plan`, depth first.
fn find_node<T: ExecutionPlan + 'static>(plan: &Arc<dyn ExecutionPlan>) -> Option<&T> {
    plan.downcast_ref::<T>()
        .or_else(|| plan.children().into_iter().find_map(find_node::<T>))
}

/// The plan nodes and providers can be named and inspected from outside this
/// crate, and rebuilt from their parts, as a codec that serializes them does.
#[tokio::test]
async fn test_plan_nodes_are_inspectable() -> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_plan_nodes".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let ident = TableIdent::new(namespace.clone(), "my_table".to_string());
    let client: Arc<dyn Catalog> = Arc::new(iceberg_catalog);

    let ctx = SessionContext::new();
    let catalog = IcebergCatalogProvider::try_new(client.clone()).await?;
    ctx.register_catalog("catalog", Arc::new(catalog));
    let provider = ctx
        .table_provider("catalog.test_plan_nodes.my_table")
        .await?;
    let provider = provider
        .downcast_ref::<IcebergTableProvider>()
        .expect("a catalog-backed provider");
    assert_eq!(provider.table_ident(), &ident);
    assert!(Arc::ptr_eq(provider.catalog(), &client));
    let rebuilt = IcebergTableProvider::try_new(
        provider.catalog().clone(),
        provider.table_ident().namespace().clone(),
        provider.table_ident().name(),
    )
    .await?;
    assert_eq!(rebuilt.table_ident(), &ident);
    assert_eq!(rebuilt.schema(), provider.schema());

    // Write path: a commit above a write, both holding the table, and the
    // commit going through the provider's catalog. The plan that runs is
    // rebuilt from their accessors and children alone. The optimizer drops
    // the coalesce above a single-partition write, so the rebuilt commit
    // always gets one, as a codec would.
    let insert = ctx
        .sql("INSERT INTO catalog.test_plan_nodes.my_table VALUES (1, 'alan'), (2, 'turing')")
        .await?
        .create_physical_plan()
        .await?;
    let commit = insert
        .downcast_ref::<IcebergCommitExec>()
        .expect("the insert plan is rooted at a commit");
    assert_eq!(commit.table().identifier(), &ident);
    assert!(Arc::ptr_eq(commit.catalog(), &client));
    let write = find_node::<IcebergWriteExec>(&insert).expect("a write below the commit");
    assert_eq!(write.table().identifier(), &ident);
    let rebuilt_write: Arc<dyn ExecutionPlan> = Arc::new(IcebergWriteExec::new(
        write.table().clone(),
        write.children()[0].clone(),
    ));
    let rebuilt_commit = IcebergCommitExec::new(
        commit.table().clone(),
        commit.catalog().clone(),
        Arc::new(CoalescePartitionsExec::new(rebuilt_write)),
    );
    expect![[r#"
        +-------+
        | count |
        +-------+
        | 2     |
        +-------+"#]]
    .assert_eq(&run(&rebuilt_commit, &ctx).await?);

    // Read path: a scan pinned to a snapshot, rebuilt from its accessors,
    // returns the same rows.
    let table = client.load_table(&ident).await?;
    let snapshot_id = table.metadata().current_snapshot_id().unwrap();
    let pinned =
        IcebergStaticTableProvider::try_new_from_table_snapshot(table, snapshot_id)
            .await?;
    assert_eq!(pinned.snapshot_id(), Some(snapshot_id));
    ctx.register_table("pinned", Arc::new(pinned.clone()))?;
    // A later write, so a scan reading the current snapshot rather than the
    // pinned one would return its row too.
    ctx.sql("INSERT INTO catalog.test_plan_nodes.my_table VALUES (3, 'hopper')")
        .await?
        .collect()
        .await?;
    let latest_snapshot_id = client
        .load_table(&ident)
        .await?
        .metadata()
        .current_snapshot_id()
        .unwrap();
    assert_ne!(latest_snapshot_id, snapshot_id);
    let plan = ctx
        .sql("SELECT foo2 FROM pinned WHERE foo1 = 1")
        .await?
        .create_physical_plan()
        .await?;
    let scan = find_node::<IcebergTableScan>(&plan).expect("a scan");
    assert_eq!(
        scan.predicates().map(ToString::to_string).as_deref(),
        Some("foo1 = 1")
    );
    let rebuilt = rebuild_scan(scan);
    assert_eq!(rebuilt.schema(), scan.schema());
    assert_eq!(rebuilt.projection(), scan.projection());
    let expected = run(scan, &ctx).await?;
    expect![[r#"
        +------+------+
        | foo1 | foo2 |
        +------+------+
        | 1    | alan |
        +------+------+"#]]
    .assert_eq(&expected);
    assert_eq!(run(&rebuilt, &ctx).await?, expected);

    // Without a projection the scan reads every column by name, and its limit
    // is kept.
    let plan = pinned.scan(&ctx.state(), None, &[], Some(1)).await?;
    let scan = plan.downcast_ref::<IcebergTableScan>().expect("a scan");
    assert_eq!(scan.projection(), ["foo1".to_string(), "foo2".to_string()]);
    let rebuilt = rebuild_scan(scan);
    assert_eq!(rebuilt.limit(), Some(1));
    let expected = run(scan, &ctx).await?;
    expect![[r#"
        +------+------+
        | foo1 | foo2 |
        +------+------+
        | 1    | alan |
        +------+------+"#]]
    .assert_eq(&expected);
    assert_eq!(run(&rebuilt, &ctx).await?, expected);

    // A scan reads the columns of its schema and no others, so one built over
    // part of the table returns only those columns, from the pinned snapshot.
    let foo2_only = Arc::new(pinned.schema().project(&[1])?);
    let partial = IcebergTableScan::new_with_predicate(
        pinned.table().clone(),
        Some(snapshot_id),
        foo2_only,
        None,
        None,
    );
    expect![[r#"
        +--------+
        | foo2   |
        +--------+
        | alan   |
        | turing |
        +--------+"#]]
    .assert_eq(&run(&partial, &ctx).await?);

    // Metadata tables: a scan rebuilt from a metadata scan's parts reads the
    // same rows.
    let plan = ctx
        .sql("SELECT * FROM catalog.test_plan_nodes.\"my_table$snapshots\"")
        .await?
        .create_physical_plan()
        .await?;
    let metadata_scan = find_node::<IcebergMetadataScan>(&plan).expect("a metadata scan");
    let provider = metadata_scan.provider();
    assert_eq!(provider.table().identifier(), &ident);
    let rebuilt = IcebergMetadataScan::new(IcebergMetadataTableProvider::new(
        provider.table().clone(),
        provider.metadata_type().clone(),
    ));
    let batches = run_batches(metadata_scan, &ctx).await?;
    // Snapshots come back in no set order, so put the first, which has no
    // parent, first. Their ids, times and paths differ on every run, so each
    // row is checked against its snapshot's metadata.
    let batch = concat_batches(&batches[0].schema(), &batches)?;
    let order = sort_to_indices(batch.column_by_name("parent_id").unwrap(), None, None)?;
    let batch = take_record_batch(&batch, &order)?;
    let column = |name: &str| batch.column_by_name(name).unwrap().clone();
    let longs = |name: &str| -> Result<Vec<Option<i64>>, Box<dyn Error>> {
        Ok(cast(&column(name), &DataType::Int64)?
            .as_primitive::<Int64Type>()
            .iter()
            .collect())
    };
    let strings = |name: &str| -> Vec<Option<String>> {
        column(name)
            .as_string::<i32>()
            .iter()
            .map(|value| value.map(str::to_string))
            .collect()
    };
    let metadata = provider.table().metadata();
    let snapshots = [snapshot_id, latest_snapshot_id].map(|id| {
        metadata
            .snapshot_by_id(id)
            .expect("a snapshot of the table")
    });
    assert_eq!(
        batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        [
            "committed_at",
            "snapshot_id",
            "parent_id",
            "operation",
            "manifest_list",
            "summary"
        ]
    );
    assert_eq!(
        longs("committed_at")?,
        snapshots.map(|snapshot| Some(snapshot.timestamp_ms() * 1000))
    );
    assert_eq!(
        longs("snapshot_id")?,
        [Some(snapshot_id), Some(latest_snapshot_id)]
    );
    assert_eq!(longs("parent_id")?, [None, Some(snapshot_id)]);
    assert_eq!(
        strings("operation"),
        [Some("append".to_string()), Some("append".to_string())]
    );
    assert_eq!(
        strings("manifest_list"),
        snapshots.map(|snapshot| Some(snapshot.manifest_list().to_string()))
    );
    // The summaries print their keys in no set order, so sort them.
    let summaries = column("summary");
    let summaries = summaries.as_map();
    let summaries = (0..summaries.len())
        .map(|row| {
            let entries = summaries.value(row);
            let keys = entries.column(0).as_string::<i32>();
            let values = entries.column(1).as_string::<i32>();
            (0..entries.len())
                .map(|entry| format!("{}: {}", keys.value(entry), values.value(entry)))
                .collect::<BTreeSet<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        summaries,
        snapshots.map(|snapshot| {
            snapshot
                .summary()
                .additional_properties
                .iter()
                .map(|(key, value)| format!("{key}: {value}"))
                .collect::<BTreeSet<_>>()
        })
    );
    // The rebuilt scan holds the same table, so lists its snapshots in the
    // same order.
    assert_eq!(
        pretty_format_batches(&run_batches(&rebuilt, &ctx).await?)?.to_string(),
        pretty_format_batches(&batches)?.to_string()
    );

    Ok(())
}

/// A catalog-backed provider keeps the schema it was built with, but scans the
/// table as it is now. After a column is added to the table and written to, a
/// scan with no projection still returns only the provider's columns, matching
/// its schema.
#[tokio::test]
async fn test_scan_after_schema_evolution_reads_provider_columns()
-> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_schema_evolution".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let ident = TableIdent::new(namespace.clone(), "my_table".to_string());
    let client: Arc<dyn Catalog> = Arc::new(iceberg_catalog);

    let provider = Arc::new(
        IcebergTableProvider::try_new(client.clone(), namespace, "my_table").await?,
    );
    let ctx = SessionContext::new();
    ctx.register_table("t", provider.clone())?;
    ctx.sql("INSERT INTO t VALUES (1, 'alan')")
        .await?
        .collect()
        .await?;

    let table = client.load_table(&ident).await?;
    let tx = Transaction::new(&table);
    tx.update_schema()
        .add_column(AddColumn::optional(
            "foo3",
            Type::Primitive(PrimitiveType::Int),
        ))
        .apply(tx)?
        .commit(client.as_ref())
        .await?;

    // A scan reads the schema of the snapshot it reads, so write a snapshot
    // with the new column, through a provider that sees it.
    let evolved = IcebergTableProvider::try_new(
        client.clone(),
        ident.namespace().clone(),
        ident.name(),
    )
    .await?;
    expect![[r#"
        Schema {
            fields: [
                Field {
                    name: "foo1",
                    data_type: Int32,
                    metadata: {
                        "PARQUET:field_id": "1",
                    },
                },
                Field {
                    name: "foo2",
                    data_type: Utf8,
                    metadata: {
                        "PARQUET:field_id": "2",
                    },
                },
                Field {
                    name: "foo3",
                    data_type: Int32,
                    nullable: true,
                    metadata: {
                        "PARQUET:field_id": "3",
                    },
                },
            ],
            metadata: {},
        }
    "#]]
    .assert_debug_eq(&evolved.schema());
    ctx.register_table("evolved", Arc::new(evolved))?;
    ctx.sql("INSERT INTO evolved VALUES (2, 'turing', 3)")
        .await?
        .collect()
        .await?;

    let plan = provider.scan(&ctx.state(), None, &[], None).await?;
    assert_eq!(plan.schema(), provider.schema());
    // The rows come from two data files, which may be read in either order.
    // They are joined under their own schema, not the plan's, so that a column
    // the plan does not report would show.
    let batches = run_batches(plan.as_ref(), &ctx).await?;
    let batch = concat_batches(&batches[0].schema(), &batches)?;
    let order = sort_to_indices(batch.column_by_name("foo1").unwrap(), None, None)?;
    let sorted = take_record_batch(&batch, &order)?;
    expect![[r#"
        +------+--------+
        | foo1 | foo2   |
        +------+--------+
        | 1    | alan   |
        | 2    | turing |
        +------+--------+"#]]
    .assert_eq(&pretty_format_batches(&[sorted])?.to_string());

    Ok(())
}

/// The id of the sort order of the tables `create_sorted_table` creates.
const SORTED: Option<i32> = Some(1);

/// Creates table `name`, sorted by `id` ascending, with columns `id`, `data`,
/// a struct `info` and a list `tags`, whose values are derived from `id`. Each
/// entry of `files` becomes one data file of the given ids, in the given order,
/// which records the given sort order id.
async fn create_sorted_table(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    name: &str,
    files: &[(Option<i32>, &[i32])],
) -> Result<Table, Box<dyn Error>> {
    let sort_order = SortOrder::builder()
        .with_sort_field(
            SortField::builder()
                .source_id(1)
                .transform(Transform::Identity)
                .direction(SortDirection::Ascending)
                .null_order(NullOrder::First)
                .build(),
        )
        .build_unbound()?;
    let creation = TableCreation::builder()
        .location(temp_path())
        .name(name.to_string())
        .schema(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
                        .into(),
                    NestedField::required(
                        2,
                        "data",
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                    NestedField::optional(
                        3,
                        "info",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(
                                4,
                                "tag",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                        ])),
                    )
                    .into(),
                    NestedField::optional(
                        5,
                        "tags",
                        Type::List(ListType::new(
                            NestedField::list_element(
                                6,
                                Type::Primitive(PrimitiveType::Int),
                                false,
                            )
                            .into(),
                        )),
                    )
                    .into(),
                ])
                .build()?,
        )
        .sort_order(sort_order)
        .build();
    let table = catalog.create_table(namespace, creation).await?;
    assert_eq!(
        Some(table.metadata().default_sort_order_id() as i32),
        SORTED
    );

    let arrow_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema())?);
    let (DataType::Struct(info_fields), DataType::List(tags_field)) = (
        arrow_schema.field(2).data_type().clone(),
        arrow_schema.field(3).data_type().clone(),
    ) else {
        unreachable!("info is a struct and tags a list");
    };
    let data_dir = format!("{}/data", table.metadata().location());
    std::fs::create_dir_all(&data_dir)?;
    let mut data_files = Vec::new();
    for (i, (sort_order_id, ids)) in files.iter().enumerate() {
        let path = format!("{data_dir}/{i}.parquet");
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from_iter_values(
                    ids.iter().map(|id| format!("row {id}")),
                )),
                Arc::new(StructArray::new(
                    info_fields.clone(),
                    vec![Arc::new(StringArray::from_iter_values(
                        ids.iter().map(|id| format!("tag {id}")),
                    ))],
                    None,
                )),
                Arc::new(ListArray::new(
                    tags_field.clone(),
                    OffsetBuffer::from_lengths(ids.iter().map(|_| 2)),
                    Arc::new(Int32Array::from_iter_values(
                        ids.iter().flat_map(|id| [*id, -id]),
                    )),
                    None,
                )),
            ],
        )?;
        let mut writer = ArrowWriter::try_new(
            std::fs::File::create(&path)?,
            arrow_schema.clone(),
            None,
        )?;
        writer.write(&batch)?;
        writer.close()?;

        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(path.clone())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(std::fs::metadata(&path)?.len())
            .record_count(ids.len() as u64)
            .partition_spec_id(0)
            .partition(Struct::empty());
        if let Some(sort_order_id) = sort_order_id {
            builder.sort_order_id(*sort_order_id);
        }
        data_files.push(builder.build()?);
    }
    let tx = Transaction::new(&table);
    let tx = tx.fast_append().add_data_files(data_files).apply(tx)?;
    Ok(tx.commit(catalog.as_ref()).await?)
}

/// A session that registers the Iceberg options, with
/// `iceberg.planning.preserve_data_ordering` set to `preserve`.
async fn ordering_session(
    preserve: bool,
    target_partitions: usize,
) -> Result<SessionContext, Box<dyn Error>> {
    let config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_option_extension(IcebergDataFusionConfig::default());
    let ctx = SessionContext::new_with_config(config);
    ctx.sql(&format!(
        "SET iceberg.planning.preserve_data_ordering = {preserve}"
    ))
    .await?
    .collect()
    .await?;
    Ok(ctx)
}

/// Plans `table` as a full scan in `ctx`.
async fn plan_sorted_scan(
    table: &Table,
    ctx: &SessionContext,
) -> Result<Arc<dyn ExecutionPlan>, Box<dyn Error>> {
    let provider = IcebergStaticTableProvider::try_new_from_table(table.clone()).await?;
    Ok(provider.scan(&ctx.state(), None, &[], None).await?)
}

/// The values of the `id` column, which must be the first, of `batches`.
fn ids(batches: &[RecordBatch]) -> Vec<i32> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_primitive::<Int32Type>()
                .values()
                .to_vec()
        })
        .collect()
}

async fn sorted_namespace() -> Result<(Arc<dyn Catalog>, NamespaceIdent), Box<dyn Error>>
{
    let catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("sorted".to_string());
    set_test_namespace(&catalog, &namespace).await?;
    Ok((Arc::new(catalog), namespace))
}

#[tokio::test]
async fn test_sorted_scan_merges_files_with_overlapping_ranges()
-> Result<(), Box<dyn Error>> {
    let (catalog, namespace) = sorted_namespace().await?;
    let table = create_sorted_table(
        &catalog,
        &namespace,
        "t",
        &[
            (SORTED, &[1, 4, 7, 7]),
            (SORTED, &[2, 4, 8]),
            (SORTED, &[3, 5, 9]),
        ],
    )
    .await?;
    let ctx = ordering_session(true, 4).await?;

    let plan = plan_sorted_scan(&table, &ctx).await?;
    let scan = plan.downcast_ref::<IcebergTableScan>().unwrap();
    assert_eq!(scan.sorted_tasks().map(<[_]>::len), Some(3));
    assert_eq!(
        plan.properties()
            .output_ordering()
            .map(|ordering| ordering.to_string()),
        Some("id@0 ASC".to_string())
    );
    expect!["IcebergTableScan projection:[id,data,info,tags] predicate:[] output_ordering:[id@0 ASC] files:[3]"]
        .assert_eq(displayable(plan.as_ref()).one_line().to_string().trim_end());

    let batches = run_batches(plan.as_ref(), &ctx).await?;
    assert_eq!(ids(&batches), vec![1, 2, 3, 4, 4, 5, 7, 7, 8, 9]);
    let batch = concat_batches(&batches[0].schema(), &batches)?;
    for row in 0..batch.num_rows() {
        let id = batch.column(0).as_primitive::<Int32Type>().value(row);
        assert_eq!(
            batch.column(1).as_string::<i32>().value(row),
            format!("row {id}")
        );
        assert_eq!(
            batch
                .column(2)
                .as_struct()
                .column(0)
                .as_string::<i32>()
                .value(row),
            format!("tag {id}")
        );
        assert_eq!(
            batch
                .column(3)
                .as_list::<i32>()
                .value(row)
                .as_primitive::<Int32Type>()
                .values(),
            &[id, -id]
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_sorted_scan_merge_honors_limit() -> Result<(), Box<dyn Error>> {
    let (catalog, namespace) = sorted_namespace().await?;
    let table = create_sorted_table(
        &catalog,
        &namespace,
        "t",
        &[
            (SORTED, &[1, 4, 7]),
            (SORTED, &[2, 5, 8]),
            (SORTED, &[3, 6, 9]),
        ],
    )
    .await?;
    let ctx = ordering_session(true, 4).await?;
    let provider = IcebergStaticTableProvider::try_new_from_table(table).await?;

    let plan = provider.scan(&ctx.state(), None, &[], Some(4)).await?;
    assert!(plan.properties().output_ordering().is_some());
    assert_eq!(
        ids(&run_batches(plan.as_ref(), &ctx).await?),
        vec![1, 2, 3, 4]
    );
    Ok(())
}

/// A scan reports no order, and is planned as it is without the option, when
/// its files do not all record the same resolvable sort order.
#[tokio::test]
async fn test_sorted_scan_reports_no_order_unless_files_agree()
-> Result<(), Box<dyn Error>> {
    let (catalog, namespace) = sorted_namespace().await?;
    let cases: [(&str, [Option<i32>; 2]); 5] = [
        ("unresolvable", [SORTED, Some(99)]),
        ("missing", [SORTED, None]),
        ("unsorted", [SORTED, Some(0)]),
        ("all_unsorted", [Some(0), Some(0)]),
        ("all_missing", [None, None]),
    ];
    let on = ordering_session(true, 4).await?;
    let off = ordering_session(false, 4).await?;
    for (name, [first, second]) in cases {
        let table = create_sorted_table(
            &catalog,
            &namespace,
            name,
            &[(first, &[1, 3]), (second, &[2, 4])],
        )
        .await?;

        let plan = plan_sorted_scan(&table, &on).await?;
        let scan = plan.downcast_ref::<IcebergTableScan>().unwrap();
        assert!(scan.sorted_tasks().is_none(), "{name}");
        assert!(plan.properties().output_ordering().is_none(), "{name}");
        assert!(plan.metrics().is_none(), "{name}");
        let unordered = plan_sorted_scan(&table, &off).await?;
        assert_eq!(
            displayable(plan.as_ref()).indent(true).to_string(),
            displayable(unordered.as_ref()).indent(true).to_string(),
            "{name}"
        );

        let mut rows = ids(&run_batches(plan.as_ref(), &on).await?);
        rows.sort();
        assert_eq!(rows, vec![1, 2, 3, 4], "{name}");
    }
    Ok(())
}

#[tokio::test]
async fn test_sorted_scan_is_opt_in_and_capped() -> Result<(), Box<dyn Error>> {
    let (catalog, namespace) = sorted_namespace().await?;
    let table = create_sorted_table(
        &catalog,
        &namespace,
        "t",
        &[(SORTED, &[1]), (SORTED, &[2]), (SORTED, &[3])],
    )
    .await?;

    let unregistered = SessionContext::new();
    let plan = plan_sorted_scan(&table, &unregistered).await?;
    assert!(plan.properties().output_ordering().is_none());

    let off = ordering_session(false, 4).await?;
    let plan = plan_sorted_scan(&table, &off).await?;
    assert!(plan.properties().output_ordering().is_none());

    let capped = ordering_session(true, 4).await?;
    capped
        .sql("SET iceberg.planning.max_merge_files = 2")
        .await?
        .collect()
        .await?;
    let plan = plan_sorted_scan(&table, &capped).await?;
    assert!(plan.properties().output_ordering().is_none());

    capped
        .sql("SET iceberg.planning.max_merge_files = 3")
        .await?
        .collect()
        .await?;
    let plan = plan_sorted_scan(&table, &capped).await?;
    assert!(plan.properties().output_ordering().is_some());
    Ok(())
}

/// Plans and runs `sql` in `ctx`, returning the plan that ran and the ids it
/// returned.
async fn plan_and_run(
    ctx: &SessionContext,
    sql: &str,
) -> Result<(Arc<dyn ExecutionPlan>, Vec<i32>), Box<dyn Error>> {
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    let batches =
        datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx()).await?;
    Ok((plan, ids(&batches)))
}

#[tokio::test]
async fn test_sorted_scan_removes_sort_from_order_by() -> Result<(), Box<dyn Error>> {
    let (catalog, namespace) = sorted_namespace().await?;
    let table = create_sorted_table(
        &catalog,
        &namespace,
        "t",
        &[
            (SORTED, &[1, 4, 7]),
            (SORTED, &[2, 5, 8]),
            (SORTED, &[3, 6, 9]),
        ],
    )
    .await?;
    let provider = Arc::new(IcebergStaticTableProvider::try_new_from_table(table).await?);
    let on = ordering_session(true, 4).await?;
    on.register_table("t", provider.clone())?;
    let off = ordering_session(false, 4).await?;
    off.register_table("t", provider)?;

    let (plan, rows) = plan_and_run(&on, "SELECT id FROM t ORDER BY id").await?;
    assert!(find_node::<SortExec>(&plan).is_none());
    assert_eq!(rows, (1..=9).collect::<Vec<_>>());

    let (plan, rows) = plan_and_run(&on, "SELECT id FROM t ORDER BY id LIMIT 4").await?;
    assert!(find_node::<SortExec>(&plan).is_none());
    assert_eq!(rows, vec![1, 2, 3, 4]);

    let (plan, rows) = plan_and_run(&on, "SELECT id FROM t ORDER BY id DESC").await?;
    assert!(find_node::<SortExec>(&plan).is_some());
    assert_eq!(rows, (1..=9).rev().collect::<Vec<_>>());

    let (plan, rows) = plan_and_run(&off, "SELECT id FROM t ORDER BY id").await?;
    assert!(find_node::<SortExec>(&plan).is_some());
    assert_eq!(rows, (1..=9).collect::<Vec<_>>());
    Ok(())
}

#[tokio::test]
async fn test_sorted_scan_removes_sort_from_sort_merge_join() -> Result<(), Box<dyn Error>>
{
    let (catalog, namespace) = sorted_namespace().await?;
    let left = create_sorted_table(
        &catalog,
        &namespace,
        "l",
        &[(SORTED, &[1, 3, 5]), (SORTED, &[2, 4, 6])],
    )
    .await?;
    let right = create_sorted_table(
        &catalog,
        &namespace,
        "r",
        &[(SORTED, &[2, 5]), (SORTED, &[3, 6, 7])],
    )
    .await?;
    let sql = "SELECT l.id FROM l JOIN r ON l.id = r.id ORDER BY l.id";
    for preserve in [true, false] {
        let ctx = ordering_session(preserve, 4).await?;
        ctx.sql("SET datafusion.optimizer.prefer_hash_join = false")
            .await?
            .collect()
            .await?;
        for (name, table) in [("l", &left), ("r", &right)] {
            ctx.register_table(
                name,
                Arc::new(
                    IcebergStaticTableProvider::try_new_from_table(table.clone()).await?,
                ),
            )?;
        }

        let (plan, rows) = plan_and_run(&ctx, sql).await?;
        assert!(find_node::<SortMergeJoinExec>(&plan).is_some());
        assert_eq!(find_node::<SortExec>(&plan).is_none(), preserve);
        assert_eq!(rows, vec![2, 3, 5, 6]);
    }
    Ok(())
}

/// A scan rebuilt from the accessors of a sorted scan, and given its sorted
/// tasks, reports and keeps the same order.
#[tokio::test]
async fn test_rebuilt_sorted_scan_keeps_order() -> Result<(), Box<dyn Error>> {
    let (catalog, namespace) = sorted_namespace().await?;
    let table = create_sorted_table(
        &catalog,
        &namespace,
        "t",
        &[(SORTED, &[1, 3]), (SORTED, &[2, 4])],
    )
    .await?;
    let ctx = ordering_session(true, 4).await?;
    let plan = plan_sorted_scan(&table, &ctx).await?;
    let scan = plan.downcast_ref::<IcebergTableScan>().unwrap();
    let tasks = scan.sorted_tasks().unwrap().to_vec();

    let rebuilt = rebuild_scan(scan).with_sorted_tasks(tasks)?;
    assert_eq!(
        rebuilt.properties().output_ordering(),
        plan.properties().output_ordering()
    );
    assert_eq!(ids(&run_batches(&rebuilt, &ctx).await?), vec![1, 2, 3, 4]);

    assert!(rebuild_scan(scan).with_sorted_tasks(vec![]).is_err());
    Ok(())
}

/// Returns `task` with `deletes` attached.
fn with_deletes(
    task: &FileScanTask,
    deletes: Vec<FileScanTaskDeleteFile>,
) -> Result<FileScanTask, Box<dyn Error>> {
    Ok(FileScanTask::builder()
        .with_file_size_in_bytes(task.file_size_in_bytes())
        .with_start(task.start())
        .with_length(task.length())
        .with_record_count(task.record_count())
        .with_data_file_path(task.data_file_path().to_string())
        .with_data_file_format(task.data_file_format())
        .with_schema(task.schema_ref())
        .with_project_field_ids(task.project_field_ids().to_vec())
        .with_predicate(task.predicate().cloned())
        .with_deletes(deletes)
        .with_partition(task.partition().cloned())
        .with_partition_spec(task.partition_spec().cloned())
        .with_name_mapping(task.name_mapping().cloned())
        .with_sort_order_id(task.sort_order_id())
        .with_sort_order(task.sort_order().cloned())
        .with_case_sensitive(task.case_sensitive())
        .build()?)
}

#[tokio::test]
async fn test_sorted_scan_applies_position_deletes() -> Result<(), Box<dyn Error>> {
    let (catalog, namespace) = sorted_namespace().await?;
    let table = create_sorted_table(
        &catalog,
        &namespace,
        "t",
        &[(SORTED, &[1, 4, 7]), (SORTED, &[2, 5, 8])],
    )
    .await?;
    let ctx = ordering_session(true, 4).await?;
    let plan = plan_sorted_scan(&table, &ctx).await?;
    let scan = plan.downcast_ref::<IcebergTableScan>().unwrap();
    let tasks = scan.sorted_tasks().unwrap();

    // Deletes row 1 of each data file, ids 4 and 5.
    let delete_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2147483546".to_string(),
        )])),
        Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2147483545".to_string(),
        )])),
    ]));
    let mut paths: Vec<&str> = tasks.iter().map(|task| task.data_file_path()).collect();
    paths.sort();
    let delete_path = format!("{}/data/deletes.parquet", table.metadata().location());
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&delete_path)?,
        delete_schema.clone(),
        None,
    )?;
    writer.write(&RecordBatch::try_new(
        delete_schema,
        vec![
            Arc::new(StringArray::from(paths)),
            Arc::new(Int64Array::from(vec![1, 1])),
        ],
    )?)?;
    writer.close()?;
    let delete = FileScanTaskDeleteFile::builder()
        .with_file_path(delete_path.clone())
        .with_file_size_in_bytes(std::fs::metadata(&delete_path)?.len())
        .with_file_type(DataContentType::PositionDeletes)
        .with_file_format(DataFileFormat::Parquet)
        .with_partition_spec_id(0)
        .build();
    let tasks = tasks
        .iter()
        .map(|task| with_deletes(task, vec![delete.clone()]))
        .collect::<Result<Vec<_>, _>>()?;

    let rebuilt = rebuild_scan(scan).with_sorted_tasks(tasks)?;
    assert_eq!(ids(&run_batches(&rebuilt, &ctx).await?), vec![1, 2, 7, 8]);
    Ok(())
}
