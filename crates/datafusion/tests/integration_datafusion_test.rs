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

use std::collections::{BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::sync::Arc;
use std::vec;

use datafusion::arrow::array::{
    Array, AsArray, Int32Array, RecordBatch, StringArray, UInt64Array,
};
use datafusion::arrow::compute::{
    cast, concat_batches, sort_to_indices, take_record_batch,
};
use datafusion::arrow::datatypes::{DataType, Field, Int64Type, Schema as ArrowSchema};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::common::collect;
use datafusion::physical_plan::{ExecutionPlan, displayable};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    Literal, NestedField, PrimitiveType, Schema, Struct, StructType, Transform, Type,
    UnboundPartitionField, UnboundPartitionSpec,
};
use iceberg::transaction::{AddColumn, ApplyTransactionAction, Transaction};
use iceberg::{
    Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, Result as IcebergResult,
    TableCreation, TableIdent,
};
use iceberg_datafusion::metadata::{
    IcebergMetadataDataSource, IcebergMetadataTableProvider,
};
use iceberg_datafusion::read::IcebergDataSource;
use iceberg_datafusion::write::{IcebergCommitExec, IcebergWriteExec};
use iceberg_datafusion::{
    IcebergCatalogProvider, IcebergCatalogTableProvider, IcebergStaticTableProvider,
};
use tempfile::TempDir;

fn display_schema(schema: &ArrowSchema) -> String {
    schema
        .fields()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

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

    insta::assert_debug_snapshot!(&result, @r#"
        [
            "my_table",
            "my_table$snapshots",
            "my_table$manifests",
            "my_table$history",
        ]
    "#);

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
    assert!(
        s.value(1)
            .contains("DataSourceExec: format=iceberg, projection=[foo1, foo2, foo3]")
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
    let expected = "predicate=(foo > 1) OR (bar IS NULL)";
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
        .sql("SELECT * FROM catalog.test_insert_into.my_table ORDER BY foo1")
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();

    // Verify the data
    insta::assert_snapshot!(display_schema(&batches[0].schema()), @r#"
    Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} }
    Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }
    "#);
    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @r"
    +------+--------+
    | foo1 | foo2   |
    +------+--------+
    | 1    | alan   |
    | 2    | turing |
    +------+--------+
    ");

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

    // Verify the data
    insta::assert_snapshot!(display_schema(&batches[0].schema()), @r#"
    Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} }
    Field { "name": Utf8, metadata: {"PARQUET:field_id": "2"} }
    Field { "profile": nullable Struct("address": Struct("street": Utf8, metadata: {"PARQUET:field_id": "6"}, "city": Utf8, metadata: {"PARQUET:field_id": "7"}, "zip": Int32, metadata: {"PARQUET:field_id": "8"}), metadata: {"PARQUET:field_id": "4"}, "contact": Struct("email": Utf8, metadata: {"PARQUET:field_id": "9"}, "phone": Utf8, metadata: {"PARQUET:field_id": "10"}), metadata: {"PARQUET:field_id": "5"}), metadata: {"PARQUET:field_id": "3"} }
    "#);
    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @r"
    +----+-------+-------------------------------------------------------------------------------------------------------------------------+
    | id | name  | profile                                                                                                                 |
    +----+-------+-------------------------------------------------------------------------------------------------------------------------+
    | 1  | Alice | {address: {street: 123 Main St, city: San Francisco, zip: 94105}, contact: {email: alice@example.com, phone: 555-1234}} |
    | 2  | Bob   | {address: {street: 456 Market St, city: San Jose, zip: 95113}, contact: {email: bob@example.com, phone: }}              |
    +----+-------+-------------------------------------------------------------------------------------------------------------------------+
    ");

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

    // Verify the flattened data
    insta::assert_snapshot!(display_schema(&batches[0].schema()), @r#"
    Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} }
    Field { "name": Utf8, metadata: {"PARQUET:field_id": "2"} }
    Field { "catalog.test_insert_nested.nested_table.profile[address][street]": nullable Utf8, metadata: {"PARQUET:field_id": "6"} }
    Field { "catalog.test_insert_nested.nested_table.profile[address][city]": nullable Utf8, metadata: {"PARQUET:field_id": "7"} }
    Field { "catalog.test_insert_nested.nested_table.profile[address][zip]": nullable Int32, metadata: {"PARQUET:field_id": "8"} }
    Field { "catalog.test_insert_nested.nested_table.profile[contact][email]": nullable Utf8, metadata: {"PARQUET:field_id": "9"} }
    Field { "catalog.test_insert_nested.nested_table.profile[contact][phone]": nullable Utf8, metadata: {"PARQUET:field_id": "10"} }
    "#);
    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @r"
    +----+-------+------------------------------------------------------------------+----------------------------------------------------------------+---------------------------------------------------------------+-----------------------------------------------------------------+-----------------------------------------------------------------+
    | id | name  | catalog.test_insert_nested.nested_table.profile[address][street] | catalog.test_insert_nested.nested_table.profile[address][city] | catalog.test_insert_nested.nested_table.profile[address][zip] | catalog.test_insert_nested.nested_table.profile[contact][email] | catalog.test_insert_nested.nested_table.profile[contact][phone] |
    +----+-------+------------------------------------------------------------------+----------------------------------------------------------------+---------------------------------------------------------------+-----------------------------------------------------------------+-----------------------------------------------------------------+
    | 1  | Alice | 123 Main St                                                      | San Francisco                                                  | 94105                                                         | alice@example.com                                               | 555-1234                                                        |
    | 2  | Bob   | 456 Market St                                                    | San Jose                                                       | 95113                                                         | bob@example.com                                                 |                                                                 |
    +----+-------+------------------------------------------------------------------+----------------------------------------------------------------+---------------------------------------------------------------+-----------------------------------------------------------------+-----------------------------------------------------------------+
    ");

    Ok(())
}

#[tokio::test]
async fn test_insert_into_partitioned() -> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_partitioned_write".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    // Create a schema with string and nullable float/double partition columns
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String))
                .into(),
            NestedField::required(3, "value", Type::Primitive(PrimitiveType::String))
                .into(),
            NestedField::optional(
                4,
                "float_partition",
                Type::Primitive(PrimitiveType::Float),
            )
            .into(),
            NestedField::optional(
                5,
                "double_partition",
                Type::Primitive(PrimitiveType::Double),
            )
            .into(),
        ])
        .build()?;

    // Identity-partition on category and both floating-point values
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(
            UnboundPartitionField::builder()
                .source_ids(vec![2])
                .name("category")
                .transform(Transform::Identity)
                .build()?,
        )?
        .add_partition_field(
            UnboundPartitionField::builder()
                .source_ids(vec![4])
                .name("float_partition")
                .transform(Transform::Identity)
                .build()?,
        )?
        .add_partition_field(
            UnboundPartitionField::builder()
                .source_ids(vec![5])
                .name("double_partition")
                .transform(Transform::Identity)
                .build()?,
        )?
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

    // One task must return all five partition files in the same Avro payload.
    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    ctx.register_catalog("catalog", catalog);

    // Insert data with multiple partition values in a single batch
    let df = ctx
        .sql(
            r#"
            INSERT INTO catalog.test_partitioned_write.partitioned_table
            VALUES
                (1, 'electronics', 'laptop', CAST(123.456 AS REAL), CAST(123.456 AS DOUBLE)),
                (2, 'electronics', 'phone', CAST(123.456 AS REAL), CAST(123.456 AS DOUBLE)),
                (3, 'books', 'novel', CAST('NaN' AS REAL), CAST('NaN' AS DOUBLE)),
                (4, 'books', 'textbook', CAST('Infinity' AS REAL), CAST('Infinity' AS DOUBLE)),
                (5, 'clothing', 'shirt', CAST('-Infinity' AS REAL), CAST('-Infinity' AS DOUBLE)),
                (6, 'clothing', 'coat', CAST(NULL AS REAL), CAST(NULL AS DOUBLE))
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
    assert_eq!(rows_inserted.value(0), 6);

    // Read all rows without a filter, including NaN, infinities, and null.
    let df = ctx
        .sql("SELECT * FROM catalog.test_partitioned_write.partitioned_table ORDER BY id")
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();

    // Verify the data - note that _partition column should NOT be present
    insta::assert_snapshot!(display_schema(&batches[0].schema()), @r#"
    Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} }
    Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} }
    Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }
    Field { "float_partition": nullable Float32, metadata: {"PARQUET:field_id": "4"} }
    Field { "double_partition": nullable Float64, metadata: {"PARQUET:field_id": "5"} }
    "#);
    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @r"
    +----+-------------+----------+-----------------+------------------+
    | id | category    | value    | float_partition | double_partition |
    +----+-------------+----------+-----------------+------------------+
    | 1  | electronics | laptop   | 123.456         | 123.456          |
    | 2  | electronics | phone    | 123.456         | 123.456          |
    | 3  | books       | novel    | NaN             | NaN              |
    | 4  | books       | textbook | inf             | inf              |
    | 5  | clothing    | shirt    | -inf            | -inf             |
    | 6  | clothing    | coat     |                 |                  |
    +----+-------------+----------+-----------------+------------------+
    ");

    // Verify the committed files, rows, and partition values.
    let table_ident = TableIdent::new(namespace.clone(), "partitioned_table".to_string());
    let table = client.load_table(&table_ident).await?;

    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = table.manifest_list_reader(snapshot).load().await?;
    let mut file_count = 0;
    let mut row_count = 0;
    let mut partitions = HashSet::new();
    for entry in manifest_list.entries() {
        let manifest = table.manifest_reader().read(entry).await?;
        for entry in manifest.entries() {
            file_count += 1;
            row_count += entry.data_file().record_count();
            partitions.insert(entry.data_file().partition().clone());
        }
    }
    assert_eq!(file_count, 5);
    assert_eq!(row_count, 6);
    let partition = |category: &str, float: Option<f32>, double: Option<f64>| {
        Struct::from_iter([
            Some(Literal::string(category)),
            float.map(Literal::float),
            double.map(Literal::double),
        ])
    };
    assert_eq!(
        partitions,
        HashSet::from([
            partition("electronics", Some(123.456), Some(123.456)),
            partition("books", Some(f32::NAN), Some(f64::NAN)),
            partition("books", Some(f32::INFINITY), Some(f64::INFINITY)),
            partition("clothing", Some(f32::NEG_INFINITY), Some(f64::NEG_INFINITY)),
            partition("clothing", None, None),
        ])
    );

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
fn rebuild_scan(scan: &IcebergDataSource) -> IcebergDataSource {
    IcebergDataSource::builder(scan.table().clone(), scan.schema())
        .with_snapshot_id(scan.snapshot_id())
        .with_predicate(scan.predicate().cloned())
        .with_fetch(scan.fetch())
        .build()
        .unwrap()
}

async fn run_source(
    source: &IcebergDataSource,
    ctx: &SessionContext,
) -> Result<String, Box<dyn Error>> {
    let exec = DataSourceExec::from_data_source(source.clone());
    run(exec.as_ref(), ctx).await
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
        .downcast_ref::<IcebergCatalogTableProvider>()
        .expect("a catalog-backed provider");
    assert_eq!(provider.table_ident(), &ident);
    assert!(Arc::ptr_eq(provider.catalog(), &client));
    let rebuilt = IcebergCatalogTableProvider::try_new(
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
    insta::assert_snapshot!(&run(&rebuilt_commit, &ctx).await?, @r#"
        +-------+
        | count |
        +-------+
        | 2     |
        +-------+"#);

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
    let scan_exec = find_node::<DataSourceExec>(&plan).expect("a data source");
    let scan = scan_exec
        .data_source()
        .downcast_ref::<IcebergDataSource>()
        .expect("an Iceberg data source");
    assert_eq!(
        scan.predicate().map(ToString::to_string).as_deref(),
        Some("foo1 = 1")
    );
    let rebuilt = rebuild_scan(scan);
    assert_eq!(rebuilt.schema(), scan.schema());
    assert_eq!(rebuilt.projection(), scan.projection());
    let expected = run_source(scan, &ctx).await?;
    insta::assert_snapshot!(&expected, @r#"
        +------+------+
        | foo1 | foo2 |
        +------+------+
        | 1    | alan |
        +------+------+"#);
    assert_eq!(run_source(&rebuilt, &ctx).await?, expected);

    // Without a projection the scan reads every column by name, and its limit
    // is kept.
    let plan = pinned.scan(&ctx.state(), None, &[], Some(1)).await?;
    let scan_exec = plan
        .downcast_ref::<DataSourceExec>()
        .expect("a data source");
    let scan = scan_exec
        .data_source()
        .downcast_ref::<IcebergDataSource>()
        .expect("an Iceberg data source");
    assert_eq!(scan.projection(), ["foo1".to_string(), "foo2".to_string()]);
    let rebuilt = rebuild_scan(scan);
    assert_eq!(rebuilt.fetch(), Some(1));
    let expected = run_source(scan, &ctx).await?;
    insta::assert_snapshot!(&expected, @r#"
        +------+------+
        | foo1 | foo2 |
        +------+------+
        | 1    | alan |
        +------+------+"#);
    assert_eq!(run_source(&rebuilt, &ctx).await?, expected);

    // A scan reads the columns of its schema and no others, so one built over
    // part of the table returns only those columns, from the pinned snapshot.
    let foo2_only = Arc::new(pinned.schema().project(&[1])?);
    let partial = IcebergDataSource::builder(pinned.table().clone(), foo2_only)
        .with_snapshot_id(Some(snapshot_id))
        .build()?;
    insta::assert_snapshot!(&run_source(&partial, &ctx).await?, @r#"
        +--------+
        | foo2   |
        +--------+
        | alan   |
        | turing |
        +--------+"#);

    // Metadata tables: a scan rebuilt from a metadata scan's parts reads the
    // same rows.
    let plan = ctx
        .sql("SELECT * FROM catalog.test_plan_nodes.\"my_table$snapshots\"")
        .await?
        .create_physical_plan()
        .await?;
    let metadata_exec =
        find_node::<DataSourceExec>(&plan).expect("a metadata data source");
    let metadata_scan = metadata_exec
        .data_source()
        .downcast_ref::<IcebergMetadataDataSource>()
        .expect("an Iceberg metadata source");
    let provider = metadata_scan.provider();
    assert_eq!(provider.table().identifier(), &ident);
    let rebuilt = IcebergMetadataDataSource::try_new(
        IcebergMetadataTableProvider::new(
            provider.table().clone(),
            provider.metadata_type().clone(),
        ),
        None,
    )?;
    let batches = run_batches(metadata_exec, &ctx).await?;
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
    let rebuilt_exec = DataSourceExec::from_data_source(rebuilt);
    assert_eq!(
        pretty_format_batches(&run_batches(rebuilt_exec.as_ref(), &ctx).await?)?
            .to_string(),
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
        IcebergCatalogTableProvider::try_new(client.clone(), namespace, "my_table")
            .await?,
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
    let evolved = IcebergCatalogTableProvider::try_new(
        client.clone(),
        ident.namespace().clone(),
        ident.name(),
    )
    .await?;
    insta::assert_snapshot!(display_schema(&evolved.schema()), @r#"
    Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} }
    Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }
    Field { "foo3": nullable Int32, metadata: {"PARQUET:field_id": "3"} }
    "#);
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
    insta::assert_snapshot!(&pretty_format_batches(&[sorted])?.to_string(), @r#"
        +------+--------+
        | foo1 | foo2   |
        +------+--------+
        | 1    | alan   |
        | 2    | turing |
        +------+--------+"#);

    Ok(())
}

/// An INSERT whose input ends in a projection that renames and reorders columns gets
/// the partition projection merged into it by the optimizer. The partition values
/// must still come from the columns being written (apache/datafusion-iceberg#35).
#[tokio::test]
async fn test_insert_into_partitioned_through_reordering_projection()
-> Result<(), Box<dyn Error>> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_reordered_insert".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "val", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()?;
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(
            UnboundPartitionField::builder()
                .source_ids(vec![1])
                .name("id")
                .transform(Transform::Identity)
                .build()?,
        )?
        .build();
    let creation = TableCreation::builder()
        .name("partitioned_by_id".to_string())
        .location(temp_path())
        .schema(schema)
        .partition_spec(partition_spec)
        .properties(HashMap::new())
        .build();
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    // One partition keeps repartitioning out of the plan.
    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    ctx.register_catalog("catalog", catalog);

    let source_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]));
    let source = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![100, 200])),
        ],
    )?;
    ctx.register_table(
        "source",
        Arc::new(MemTable::try_new(source_schema, vec![vec![source]])?),
    )?;

    // `id` is written from `b`, the second source column, so the partition values
    // must read `b@1` and not the first column of the merged projection's input.
    let plan = ctx
        .sql(
            "INSERT INTO catalog.test_reordered_insert.partitioned_by_id
             SELECT b AS id, a AS val FROM source",
        )
        .await?
        .create_physical_plan()
        .await?;
    insta::assert_snapshot!(displayable(plan.as_ref()).indent(false), @r"
    IcebergCommitExec: table=test_reordered_insert.partitioned_by_id
      IcebergWriteExec: table=test_reordered_insert.partitioned_by_id
        ProjectionExec: expr=[b@1 as id, a@0 as val, iceberg_partition_values(id: identity(b@1)) as _partition]
          DataSourceExec: partitions=1, partition_sizes=[1]
    ");
    insta::assert_snapshot!(run(plan.as_ref(), &ctx).await?, @r"
    +-------+
    | count |
    +-------+
    | 2     |
    +-------+
    ");

    let table_ident = TableIdent::new(namespace, "partitioned_by_id".to_string());
    let table = client.load_table(&table_ident).await?;
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = table.manifest_list_reader(snapshot).load().await?;
    let mut partitions = HashSet::new();
    for entry in manifest_list.entries() {
        let manifest = table.manifest_reader().read(entry).await?;
        for entry in manifest.entries() {
            partitions.insert(entry.data_file().partition().clone());
        }
    }
    assert_eq!(
        partitions,
        HashSet::from([
            Struct::from_iter([Some(Literal::int(100))]),
            Struct::from_iter([Some(Literal::int(200))]),
        ])
    );

    // A filter on the partition column prunes by these values, so it only finds
    // the row if the values are right.
    let batches = ctx
        .sql("SELECT * FROM catalog.test_reordered_insert.partitioned_by_id WHERE id = 100")
        .await?
        .collect()
        .await?;
    insta::assert_snapshot!(pretty_format_batches(&batches)?.to_string(), @r"
    +-----+-----+
    | id  | val |
    +-----+-----+
    | 100 | 1   |
    +-----+-----+
    ");

    Ok(())
}
