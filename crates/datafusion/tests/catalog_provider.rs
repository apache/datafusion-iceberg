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

//! The catalog and schema providers, and the schema of what they scan.

use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::execution::context::SessionContext;
use datafusion::parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::{NamespaceIdent, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use iceberg_datafusion_test_tools::FixtureCatalog;

/// The schema of the tables here: `foo1 INT NOT NULL, foo2 STRING NOT NULL`.
fn foo_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String))
                .into(),
        ])
        .build()
        .unwrap()
}

#[tokio::test]
async fn test_provider_plan_stream_schema() -> Result<(), Box<dyn Error>> {
    let fixtures = FixtureCatalog::try_new().await?;
    fixtures
        .table(
            TableIdent::from_strs(["test_provider_get_table_schema", "my_table"])?,
            foo_schema(),
        )
        .create()
        .await?;
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(fixtures.catalog()).await?),
    );

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
    let fixtures = FixtureCatalog::try_new().await?;
    fixtures
        .table(
            TableIdent::from_strs(["test_provider_list_table_names", "my_table"])?,
            foo_schema(),
        )
        .create()
        .await?;
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(fixtures.catalog()).await?),
    );

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
    let fixtures = FixtureCatalog::try_new().await?;
    fixtures
        .catalog()
        .create_namespace(
            &NamespaceIdent::new("test_provider_list_schema_names".to_string()),
            HashMap::new(),
        )
        .await?;
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(fixtures.catalog()).await?),
    );

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
