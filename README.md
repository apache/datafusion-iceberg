<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Apache Iceberg Table Provider for Apache DataFusion

This repository connects [Apache Iceberg](https://iceberg.apache.org/) tables to
[Apache DataFusion](https://datafusion.apache.org/). The `iceberg-datafusion`
crate provides table providers for querying Iceberg data and for inserting rows
through an Iceberg catalog.

## Workspace

- [`crates/datafusion`](crates/datafusion): the DataFusion integration library.
- [`crates/sqllogictest`](crates/sqllogictest): SQL logic tests for the integration.
- [`crates/playground`](crates/playground): a command-line SQL playground backed
  by DataFusion and Iceberg catalogs.

## Query an existing table

Register `IcebergTableProviderFactory` in a DataFusion session, then point an
external table at an existing Iceberg metadata file:

```rust
use std::sync::Arc;

use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use iceberg_datafusion::IcebergTableProviderFactory;

async fn query_table() -> datafusion::error::Result<()> {
    let mut state = SessionStateBuilder::new().with_default_features().build();
    state.table_factories_mut().insert(
        "ICEBERG".to_string(),
        Arc::new(IcebergTableProviderFactory::new()),
    );
    let ctx = SessionContext::new_with_state(state);

    ctx.sql(
        "CREATE EXTERNAL TABLE trips STORED AS ICEBERG \
         LOCATION '/absolute/path/to/table/metadata/v1.metadata.json'",
    )
    .await?
    .collect()
    .await?;

    let batches = ctx
        .sql("SELECT * FROM trips LIMIT 10")
        .await?
        .collect()
        .await?;
    println!("{batches:?}");
    Ok(())
}
```

Replace the metadata path with one from an existing table whose data files are
accessible to the process. External-table registration reads an existing table;
it does not create one. For catalog-backed access and inserts, register an
`IcebergCatalogProvider` with a configured Iceberg `Catalog`. See the
[integration tests](crates/datafusion/tests/integration_datafusion_test.rs) for
examples.

The workspace uses a pinned `iceberg-rust` Git revision. Applications that
also depend on Iceberg crates should use the same revision shown in
[`Cargo.toml`](Cargo.toml) so Cargo uses one Iceberg crate source.

## Development

The repository's [`rust-toolchain.toml`](rust-toolchain.toml) selects the Rust
toolchain used by CI. From the repository root, run:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked
```

This project is licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

### Eager file planning

Register `IcebergPlanningConfig` and enable `iceberg.eager_file_planning` to discover
files during physical planning. The default remains lazy discovery. Both modes use
Iceberg's native file tasks, including projection, predicates, and delete references.

`IcebergDataSource::builder(table, schema).build()` constructs a lazy source without
file-discovery I/O. `build_eager(task_context).await` constructs a separate
`EagerIcebergDataSource`: its state contains assigned files and a shared reader,
without retaining the table or the request used to discover files. Eager planning
requires an I/O-capable caller runtime and charges retained file descriptors to the
query memory pool. Local partition count is bounded by selected files and the session
target; an empty scan has one empty partition. Statistics remain unknown.

Consumers can inspect `planned_partition_tasks(partition)`, derive assignments with
`with_file_groups(groups)`, and reconstruct an eager source with `from_file_groups`.
Regrouping uses positions in the source's `planned_files()` iterator, permits subsets,
and shares the original descriptors and memory reservation. Groups preserve task
order but do not declare sorted output. Explicit groups may include empty partitions.

Reconstruction accepts a projected schema, native file tasks grouped by partition,
a process-local `ArrowReader`, fetch, and a task context. The caller configures the
reader's FileIO, runtime, and concurrency before construction. Reader clones share
Iceberg's delete cache across local partitions; regrouping preserves reader settings.
Transport and worker resource configuration remain the consumer's responsibility.
The pinned Iceberg API does not yet expose the portable reader options proposed in
[iceberg-rust #3514](https://github.com/apache/iceberg-rust/pull/3514).
