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
[Apache DataFusion](https://datafusion.apache.org/). The `datafusion-iceberg`
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
use datafusion_iceberg::IcebergTableProviderFactory;

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
