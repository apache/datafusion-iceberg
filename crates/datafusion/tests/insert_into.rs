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

//! `INSERT INTO` through a catalog-backed provider: plain, nested and
//! partitioned tables.

use std::collections::HashSet;
use std::error::Error;
use std::sync::Arc;

use datafusion::arrow::array::{Array, UInt64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::execution::context::{SessionConfig, SessionContext};
use iceberg::TableIdent;
use iceberg::spec::{
    Literal, NestedField, PrimitiveType, Schema, Struct, StructType, Transform, Type,
};
use iceberg_datafusion::IcebergCatalogProvider;
use iceberg_datafusion_test_tools::{FixtureCatalog, display_schema};

#[tokio::test]
async fn test_insert_into() -> Result<(), Box<dyn Error>> {
    let fixtures = FixtureCatalog::try_new().await?;
    fixtures
        .table(
            TableIdent::from_strs(["test_insert_into", "my_table"])?,
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int))
                        .into(),
                    NestedField::required(
                        2,
                        "foo2",
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                ])
                .build()?,
        )
        .create()
        .await?;
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(fixtures.catalog()).await?),
    );

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
    let fixtures = FixtureCatalog::try_new().await?;

    // Create a schema with nested fields
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String))
                .into(),
            NestedField::optional(3, "profile", Type::Struct(get_nested_struct_type()))
                .into(),
        ])
        .build()?;

    // Create the table with the nested schema
    fixtures
        .table(
            TableIdent::from_strs(["test_insert_nested", "nested_table"])?,
            schema,
        )
        .create()
        .await?;
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(fixtures.catalog()).await?),
    );

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
    let fixtures = FixtureCatalog::try_new().await?;

    // Create a schema with string and nullable float/double partition columns
    let schema = Schema::builder()
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
    let table_ident =
        TableIdent::from_strs(["test_partitioned_write", "partitioned_table"])?;
    fixtures
        .table(table_ident.clone(), schema)
        .partition_by("category", Transform::Identity)
        .partition_by("float_partition", Transform::Identity)
        .partition_by("double_partition", Transform::Identity)
        .create()
        .await?;

    // One task must return all five partition files in the same Avro payload.
    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    ctx.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(fixtures.catalog()).await?),
    );

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
    let table = fixtures.catalog().load_table(&table_ident).await?;

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
