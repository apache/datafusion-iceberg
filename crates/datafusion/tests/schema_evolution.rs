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

//! Scans of a table whose schema changed after a provider was built.

use std::error::Error;
use std::sync::Arc;

use datafusion::arrow::compute::{concat_batches, sort_to_indices, take_record_batch};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::TableProvider;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::common::collect;
use iceberg::TableIdent;
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::transaction::{AddColumn, ApplyTransactionAction, Transaction};
use iceberg_datafusion::IcebergCatalogTableProvider;
use iceberg_datafusion_test_tools::{FixtureCatalog, display_schema};

/// A catalog-backed provider keeps the schema it was built with, but scans the
/// table as it is now. After a column is added to the table and written to, a
/// scan with no projection still returns only the provider's columns, matching
/// its schema.
#[tokio::test]
async fn test_scan_after_schema_evolution_reads_provider_columns()
-> Result<(), Box<dyn Error>> {
    let fixtures = FixtureCatalog::try_new().await?;
    let ident = TableIdent::from_strs(["test_schema_evolution", "my_table"])?;
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String))
                .into(),
        ])
        .build()?;
    fixtures.table(ident.clone(), schema).create().await?;
    let client = fixtures.catalog();

    let provider = Arc::new(
        IcebergCatalogTableProvider::try_new(
            client.clone(),
            ident.namespace().clone(),
            ident.name(),
        )
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
    assert_eq!(plan.properties().partitioning.partition_count(), 1);
    let batches = collect(plan.execute(0, ctx.task_ctx())?).await?;
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
