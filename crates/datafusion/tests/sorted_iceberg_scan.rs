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

//! Integration tests for sorted scans: with
//! `iceberg.execution.preserve_data_ordering`, a scan of files that all record
//! the same sort order reports that order to DataFusion and merges the files.

use std::collections::HashMap;
use std::error::Error;
use std::str::FromStr;
use std::sync::Arc;

use datafusion::arrow::array::{
    AsArray, Int32Array, Int64Array, ListArray, RecordBatch, StringArray, StructArray,
};
use datafusion::arrow::buffer::OffsetBuffer;
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{DataType, Field, Int32Type, Schema as ArrowSchema};
use datafusion::datasource::TableProvider;
use datafusion::datasource::source::{DataSource, DataSourceExec};
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use datafusion::physical_plan::common::collect;
use datafusion::physical_plan::joins::SortMergeJoinExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{ExecutionPlan, displayable};
use futures::{StreamExt, TryStreamExt};
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::scan::{FileScanTask, FileScanTaskDeleteFile};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, ListType, NestedField, NullOrder,
    PrimitiveType, Schema, SortDirection, SortField, SortOrder, SortOrderRef, Struct,
    StructType, Transform, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{
    Catalog, CatalogBuilder, MemoryCatalog, MetadataLocation, NamespaceIdent,
    Result as IcebergResult, TableCreation,
};
use iceberg_datafusion::read::IcebergDataSource;
use iceberg_datafusion::{
    IcebergCatalogTableProvider, IcebergDataFusionConfig, IcebergStaticTableProvider,
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

async fn set_test_namespace(
    catalog: &MemoryCatalog,
    namespace: &NamespaceIdent,
) -> IcebergResult<()> {
    let properties = HashMap::new();

    catalog.create_namespace(namespace, properties).await?;

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

/// Returns the first node of type `T` in `plan`, depth first.
fn find_node<T: ExecutionPlan + 'static>(plan: &Arc<dyn ExecutionPlan>) -> Option<&T> {
    plan.downcast_ref::<T>()
        .or_else(|| plan.children().into_iter().find_map(find_node::<T>))
}

/// Returns the Iceberg source of the first `DataSourceExec` in `plan`.
fn iceberg_source(plan: &Arc<dyn ExecutionPlan>) -> Option<&IcebergDataSource> {
    find_node::<DataSourceExec>(plan)?
        .data_source()
        .downcast_ref::<IcebergDataSource>()
}

/// Rebuilds `scan` from its accessors alone, as a codec would.
fn rebuild_scan(scan: &IcebergDataSource) -> IcebergDataSource {
    rebuild_scan_with_tasks(scan, scan.sorted_tasks().map(<[_]>::to_vec)).unwrap()
}

/// Rebuilds `scan` from its accessors, with `sorted_tasks` instead of its own.
fn rebuild_scan_with_tasks(
    scan: &IcebergDataSource,
    sorted_tasks: Option<Vec<FileScanTask>>,
) -> datafusion::error::Result<IcebergDataSource> {
    IcebergDataSource::builder(scan.table().clone(), scan.schema())
        .with_snapshot_id(scan.snapshot_id())
        .with_predicate(scan.predicate().cloned())
        .with_fetch(scan.fetch())
        .with_sorted_tasks(sorted_tasks)
        .build()
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
    let files: Vec<_> = files
        .iter()
        .map(|(sort_order_id, ids)| {
            (*sort_order_id, ids.iter().copied().map(Some).collect())
        })
        .collect();
    append_sorted_files(catalog, &table, &files).await
}

/// Commits one data file of `table` per entry of `files`, with the given ids,
/// in the given order, recording the given sort order id. The other columns'
/// values are derived from `id`.
async fn append_sorted_files(
    catalog: &Arc<dyn Catalog>,
    table: &Table,
    files: &[(Option<i32>, Vec<Option<i32>>)],
) -> Result<Table, Box<dyn Error>> {
    if files.is_empty() {
        return Ok(table.clone());
    }
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
    let snapshots = table.metadata().snapshots().count();
    let mut data_files = Vec::new();
    for (i, (sort_order_id, ids)) in files.iter().enumerate() {
        let path = format!("{data_dir}/{snapshots}-{i}.parquet");
        let label = |prefix: &str, id: &Option<i32>| match id {
            Some(id) => format!("{prefix} {id}"),
            None => format!("{prefix} null"),
        };
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids.clone())),
                Arc::new(StringArray::from_iter_values(
                    ids.iter().map(|id| label("row", id)),
                )),
                Arc::new(StructArray::new(
                    info_fields.clone(),
                    vec![Arc::new(StringArray::from_iter_values(
                        ids.iter().map(|id| label("tag", id)),
                    ))],
                    None,
                )),
                Arc::new(ListArray::new(
                    tags_field.clone(),
                    OffsetBuffer::from_lengths(ids.iter().map(|_| 2)),
                    Arc::new(Int32Array::from_iter_values(ids.iter().flat_map(|id| {
                        let id = id.unwrap_or_default();
                        [id, -id]
                    }))),
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
    let tx = Transaction::new(table);
    let tx = tx.fast_append().add_data_files(data_files).apply(tx)?;
    Ok(tx.commit(catalog.as_ref()).await?)
}

/// Stands in for another engine committing, without writing data, the current
/// schema of `table` with the field of the same id as `field` replaced by it.
/// iceberg-rust has no action that makes a column optional or promotes its
/// type, so this rewrites the table's current metadata file in place.
async fn replace_field_externally(
    table: &Table,
    field: NestedField,
) -> Result<(), Box<dyn Error>> {
    let current = table.metadata().current_schema();
    let schema = Schema::builder()
        .with_fields(current.as_struct().fields().iter().map(|current| {
            if current.id == field.id {
                Arc::new(field.clone())
            } else {
                current.clone()
            }
        }))
        .build()?;
    let metadata = table
        .metadata()
        .clone()
        .into_builder(None)
        .add_current_schema(schema)?
        .build()?
        .metadata;
    let location = MetadataLocation::from_str(table.metadata_location_result()?)?;
    metadata.write_to(table.file_io(), &location).await?;
    Ok(())
}

/// The values, or nulls, of the `id` column, which must be the first, of
/// `batches`.
fn nullable_ids(batches: &[RecordBatch]) -> Vec<Option<i32>> {
    batches
        .iter()
        .flat_map(|batch| batch.column(0).as_primitive::<Int32Type>().iter())
        .collect()
}

/// A session that registers the Iceberg options, with
/// `iceberg.execution.preserve_data_ordering` set to `preserve`.
async fn ordering_session(
    preserve: bool,
    target_partitions: usize,
) -> Result<SessionContext, Box<dyn Error>> {
    let config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_option_extension(IcebergDataFusionConfig::default());
    let ctx = SessionContext::new_with_config(config);
    ctx.sql(&format!(
        "SET iceberg.execution.preserve_data_ordering = {preserve}"
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
    let scan = iceberg_source(&plan).unwrap();
    assert_eq!(scan.sorted_tasks().map(<[_]>::len), Some(3));
    assert_eq!(
        plan.properties()
            .output_ordering()
            .map(|ordering| ordering.to_string()),
        Some("id@0 ASC".to_string())
    );
    insta::assert_snapshot!(
        displayable(plan.as_ref()).one_line().to_string().trim_end(),
        @"DataSourceExec: format=iceberg, projection=[id, data, info, tags], output_ordering=[id@0 ASC], files=3"
    );

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

/// Each file holds more rows than a reader batch, so the merge advances every
/// per-file stream across batch boundaries, where the order within a stream
/// comes from the reader rather than the merge.
#[tokio::test]
async fn test_sorted_scan_merges_files_longer_than_a_batch() -> Result<(), Box<dyn Error>>
{
    let (catalog, namespace) = sorted_namespace().await?;
    let evens: Vec<i32> = (0..6000).step_by(2).collect();
    let odds: Vec<i32> = (1..6000).step_by(2).collect();
    let table = create_sorted_table(
        &catalog,
        &namespace,
        "t",
        &[(SORTED, &evens), (SORTED, &odds)],
    )
    .await?;
    let ctx = ordering_session(true, 4).await?;
    let plan = plan_sorted_scan(&table, &ctx).await?;
    let scan = iceberg_source(&plan).unwrap();

    for task in scan.sorted_tasks().unwrap() {
        let batches: Vec<RecordBatch> = table
            .reader_builder()
            .build()
            .read(futures::stream::iter([Ok(task.clone())]).boxed())?
            .stream()
            .try_collect()
            .await?;
        assert!(batches.len() > 1, "{}", task.data_file_path());
    }
    assert_eq!(
        ids(&run_batches(plan.as_ref(), &ctx).await?),
        (0..6000).collect::<Vec<_>>()
    );
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
        let scan = iceberg_source(&plan).unwrap();
        assert!(scan.sorted_tasks().is_none(), "{name}");
        assert!(plan.properties().output_ordering().is_none(), "{name}");
        assert_eq!(plan.metrics().map(|m| m.iter().count()), Some(0), "{name}");
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
        .sql("SET iceberg.execution.max_merge_files_per_partition = 2")
        .await?
        .collect()
        .await?;
    let plan = plan_sorted_scan(&table, &capped).await?;
    assert!(plan.properties().output_ordering().is_none());

    capped
        .sql("SET iceberg.execution.max_merge_files_per_partition = 3")
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

/// A scan rebuilt from the accessors of a sorted scan reports and keeps the
/// same order.
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
    let scan = iceberg_source(&plan).unwrap();
    assert_eq!(scan.sorted_tasks().map(<[_]>::len), Some(2));

    let rebuilt = rebuild_scan(scan);
    assert_eq!(
        rebuilt.eq_properties().output_ordering().as_ref(),
        plan.properties().output_ordering()
    );
    let rebuilt = DataSourceExec::from_data_source(rebuilt);
    assert_eq!(
        ids(&run_batches(rebuilt.as_ref(), &ctx).await?),
        vec![1, 2, 3, 4]
    );

    // Without the tasks a rebuilt source lists its files when it runs, and
    // reports no order.
    let unsorted = rebuild_scan_with_tasks(scan, None)?;
    assert!(unsorted.sorted_tasks().is_none());
    assert!(unsorted.eq_properties().output_ordering().is_none());

    assert!(rebuild_scan_with_tasks(scan, Some(vec![])).is_err());
    Ok(())
}

/// Returns a copy of `task` with `deletes` and `sort_order` in place of its own.
fn copy_task(
    task: &FileScanTask,
    deletes: Vec<FileScanTaskDeleteFile>,
    sort_order: Option<SortOrderRef>,
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
        .with_sort_order(sort_order)
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
    let scan = iceberg_source(&plan).unwrap();
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
        .build()?;
    let tasks = tasks
        .iter()
        .map(|task| copy_task(task, vec![delete.clone()], task.sort_order().cloned()))
        .collect::<Result<Vec<_>, _>>()?;

    let rebuilt =
        DataSourceExec::from_data_source(rebuild_scan_with_tasks(scan, Some(tasks))?);
    assert_eq!(
        ids(&run_batches(rebuilt.as_ref(), &ctx).await?),
        vec![1, 2, 7, 8]
    );
    Ok(())
}

/// A provider created while `id` is required declares it non-nullable, and
/// DataFusion ignores the null order of a non-nullable column. Once another
/// engine makes `id` optional and writes a NULL, the scan must report no order,
/// or `ORDER BY id`, which puts NULLs last, would keep the file's NULL first.
#[tokio::test]
async fn test_sorted_scan_reports_no_order_for_stale_nullability()
-> Result<(), Box<dyn Error>> {
    let (catalog, namespace) = sorted_namespace().await?;
    let table = create_sorted_table(&catalog, &namespace, "t", &[]).await?;
    let provider = Arc::new(
        IcebergCatalogTableProvider::try_new(catalog.clone(), namespace.clone(), "t")
            .await?,
    );
    let id = NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int));
    replace_field_externally(&table, id).await?;
    let table = catalog.load_table(table.identifier()).await?;
    append_sorted_files(&catalog, &table, &[(SORTED, vec![None, Some(0), Some(7)])])
        .await?;

    for preserve in [false, true] {
        let ctx = ordering_session(preserve, 4).await?;
        ctx.register_table("t", provider.clone())?;
        let plan = ctx
            .sql("SELECT id FROM t ORDER BY id")
            .await?
            .create_physical_plan()
            .await?;
        let scan = iceberg_source(&plan).unwrap();
        assert!(scan.sorted_tasks().is_none(), "preserve = {preserve}");
        let batches =
            datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx()).await?;
        assert_eq!(
            nullable_ids(&batches),
            vec![Some(0), Some(7), None],
            "preserve = {preserve}"
        );
    }
    Ok(())
}

/// Another engine promotes `id` from int to long without writing data. A
/// provider created afterwards declares a long `id`, while the scan reads the
/// snapshot's int, which the merge could not compare as declared.
#[tokio::test]
async fn test_sorted_scan_reports_no_order_for_promoted_type()
-> Result<(), Box<dyn Error>> {
    let (catalog, namespace) = sorted_namespace().await?;
    let table = create_sorted_table(
        &catalog,
        &namespace,
        "t",
        &[(SORTED, &[1, 3]), (SORTED, &[2, 4])],
    )
    .await?;
    let id = NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long));
    replace_field_externally(&table, id).await?;
    let provider = Arc::new(
        IcebergCatalogTableProvider::try_new(catalog.clone(), namespace, "t").await?,
    );

    for preserve in [false, true] {
        let ctx = ordering_session(preserve, 4).await?;
        ctx.register_table("t", provider.clone())?;
        let plan = ctx
            .sql("SELECT * FROM t")
            .await?
            .create_physical_plan()
            .await?;
        let scan = iceberg_source(&plan).unwrap();
        assert!(scan.sorted_tasks().is_none(), "preserve = {preserve}");
        let batches =
            datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx()).await?;
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(rows, 4, "preserve = {preserve}");
    }
    Ok(())
}

/// Iceberg sorts ascending fields with NULLs first by default, while
/// DataFusion's `ORDER BY id` puts them last, so on a nullable sort column only
/// `ORDER BY id NULLS FIRST` can use the order the scan reports.
#[tokio::test]
async fn test_sorted_scan_on_nullable_key() -> Result<(), Box<dyn Error>> {
    let (catalog, namespace) = sorted_namespace().await?;
    let table = create_sorted_table(&catalog, &namespace, "t", &[]).await?;
    let id = NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int));
    replace_field_externally(&table, id).await?;
    let table = catalog.load_table(table.identifier()).await?;
    let table = append_sorted_files(
        &catalog,
        &table,
        &[
            (SORTED, vec![None, Some(1), Some(4)]),
            (SORTED, vec![None, Some(2), Some(3)]),
        ],
    )
    .await?;
    let ctx = ordering_session(true, 4).await?;
    ctx.register_table(
        "t",
        Arc::new(IcebergStaticTableProvider::try_new_from_table(table).await?),
    )?;

    for (sql, sorts, expected) in [
        (
            "SELECT id FROM t ORDER BY id NULLS FIRST",
            false,
            vec![None, None, Some(1), Some(2), Some(3), Some(4)],
        ),
        (
            "SELECT id FROM t ORDER BY id",
            true,
            vec![Some(1), Some(2), Some(3), Some(4), None, None],
        ),
    ] {
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let scan = iceberg_source(&plan).unwrap();
        assert_eq!(scan.sorted_tasks().map(<[_]>::len), Some(2), "{sql}");
        assert_eq!(find_node::<SortExec>(&plan).is_some(), sorts, "{sql}");
        let batches =
            datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx()).await?;
        assert_eq!(nullable_ids(&batches), expected, "{sql}");
    }
    Ok(())
}

/// A sorted scan of one file reports its order and returns its rows without a
/// merge, and a scan of a sorted table with no files reports no order.
#[tokio::test]
async fn test_sorted_scan_of_one_file_and_of_no_files() -> Result<(), Box<dyn Error>> {
    let (catalog, namespace) = sorted_namespace().await?;
    let ctx = ordering_session(true, 4).await?;

    let one =
        create_sorted_table(&catalog, &namespace, "one", &[(SORTED, &[1, 2, 3])]).await?;
    let plan = plan_sorted_scan(&one, &ctx).await?;
    assert_eq!(
        iceberg_source(&plan)
            .unwrap()
            .sorted_tasks()
            .map(<[_]>::len),
        Some(1)
    );
    assert_eq!(
        plan.properties()
            .output_ordering()
            .map(|ordering| ordering.to_string()),
        Some("id@0 ASC".to_string())
    );
    assert_eq!(ids(&run_batches(plan.as_ref(), &ctx).await?), vec![1, 2, 3]);

    let empty = create_sorted_table(&catalog, &namespace, "empty", &[]).await?;
    let plan = plan_sorted_scan(&empty, &ctx).await?;
    assert!(iceberg_source(&plan).unwrap().sorted_tasks().is_none());
    assert!(plan.properties().output_ordering().is_none());
    assert!(ids(&run_batches(plan.as_ref(), &ctx).await?).is_empty());
    Ok(())
}

/// Tasks given to the builder must all record the same sort order, not only
/// the same sort order id, because the merge uses one order for all of them.
#[tokio::test]
async fn test_rebuilt_sorted_scan_rejects_tasks_with_different_orders()
-> Result<(), Box<dyn Error>> {
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
    let scan = iceberg_source(&plan).unwrap();
    let mut tasks = scan.sorted_tasks().unwrap().to_vec();

    let descending = SortOrder::builder()
        .with_order_id(1)
        .with_sort_field(
            SortField::builder()
                .source_id(1)
                .transform(Transform::Identity)
                .direction(SortDirection::Descending)
                .null_order(NullOrder::Last)
                .build(),
        )
        .build_unbound()?;
    tasks[1] = copy_task(&tasks[1], vec![], Some(Arc::new(descending)))?;
    assert_eq!(tasks[1].sort_order_id(), tasks[0].sort_order_id());
    assert_ne!(tasks[1].sort_order(), tasks[0].sort_order());

    assert!(rebuild_scan_with_tasks(scan, Some(tasks)).is_err());
    Ok(())
}
