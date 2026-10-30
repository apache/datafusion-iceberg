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

//! The plan nodes and providers can be inspected from outside this crate and
//! rebuilt from their parts, as a codec that serializes them does.

use std::collections::BTreeSet;
use std::error::Error;
use std::sync::Arc;

use datafusion::arrow::array::{Array, AsArray, RecordBatch};
use datafusion::arrow::compute::{
    cast, concat_batches, sort_to_indices, take_record_batch,
};
use datafusion::arrow::datatypes::{DataType, Int64Type};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::TableProvider;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::common::collect;
use iceberg::TableIdent;
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg_datafusion::metadata::{
    IcebergMetadataDataSource, IcebergMetadataTableProvider,
};
use iceberg_datafusion::read::IcebergDataSource;
use iceberg_datafusion::write::{IcebergCommitExec, IcebergWriteExec};
use iceberg_datafusion::{
    IcebergCatalogProvider, IcebergCatalogTableProvider, IcebergStaticTableProvider,
};
use iceberg_datafusion_test_tools::FixtureCatalog;

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
    let fixtures = FixtureCatalog::try_new().await?;
    let ident = TableIdent::from_strs(["test_plan_nodes", "my_table"])?;
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String))
                .into(),
        ])
        .build()?;
    fixtures.table(ident.clone(), schema).create().await?;
    let client = fixtures.catalog();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(fixtures.catalog()).await?),
    );
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
