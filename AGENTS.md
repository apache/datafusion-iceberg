<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Agent Guidelines for Apache DataFusion Iceberg

Apache DataFusion integration for Apache Iceberg: catalog/table providers and physical operators that let DataFusion read and append to Iceberg tables. The code and its git history were ported from `apache/iceberg-rust`, so older commits and issue links refer to that repo.

| Directory | Package | Purpose |
|---|---|---|
| `crates/datafusion` | `datafusion-iceberg` | The published library |
| `crates/sqllogictest` | `iceberg-sqllogictest` | SQL-level test harness (unpublished) |
| `crates/playground` | `iceberg-playground` | REPL CLI built on `datafusion-cli` (unpublished) |

## Before Committing

CI runs these checks. Run them and fix any errors before committing; `cargo fmt --all` fixes formatting.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked
```

- The Rust toolchain is pinned to 1.98.1 in `rust-toolchain.toml`. The same version is repeated in `.github/actions/setup-rust/action.yml`, so bump both together.
- `--locked`: any dependency change must include the updated `Cargo.lock`.
- The workspace lint `unused_qualifications = "deny"` fails compilation (not just clippy) when a path is spelled out but already imported, e.g. `std::sync::Arc::new` after `use std::sync::Arc`.
- rustfmt is configured with `max_width = 90`.

## Testing

```sh
cargo test -p datafusion-iceberg --lib <name_substring>                          # unit tests
cargo test -p datafusion-iceberg --test integration_datafusion_test <name>      # integration tests
cargo test -p iceberg-sqllogictest --test sqllogictests -- <schedule_substring> # e.g. basic_queries
UPDATE_EXPECT=1 cargo test -p datafusion-iceberg --test integration_datafusion_test  # rewrite expect![[...]] snapshots
cargo run -p iceberg-playground -- --rc <catalogs.toml>                          # REPL; defaults to ~/.icebergrc
```

Tests are self-contained (in-memory catalogs, temp dirs, checked-in metadata JSON); no Docker or external services are needed. The playground config holds `[[catalogs]]` entries with `name`, `type` (`rest` or `memory`), and a `[catalogs.config]` table of string properties.

## Architecture (`crates/datafusion`)

### Catalog → schema → table

`IcebergCatalogProvider` wraps an `iceberg::Catalog` and builds one `IcebergSchemaProvider` per namespace, each holding an `IcebergTableProvider` per table. Namespaces and tables are listed once, at construction, and are never refreshed. Tables created by other clients after that point are invisible.

- Names like `<table>$snapshots`, `$manifests`, and `$history` resolve to `IcebergMetadataTableProvider`.
- SQL `CREATE TABLE` and `DROP TABLE` reach the *synchronous* `SchemaProvider::register_table` and `deregister_table` methods. On a multi-thread Tokio runtime they run the async catalog through `block_in_place` + `Handle::block_on`; without a caller runtime they create a current-thread runtime for the operation. Calls from a current-thread runtime return an error because blocking would stop that runtime from driving catalog I/O. `register_table` rejects input tables that contain rows, so CTAS isn't supported. It auto-assigns field IDs and uses format V2 unless the schema needs V3.

### Table providers (`table/`)

- **`IcebergTableProvider`**: catalog-backed. It caches the Arrow schema at construction but reloads table metadata from the catalog on every `scan` and `insert_into`. INSERT supports append only.
- **`IcebergStaticTableProvider`**: wraps a fixed `Table`, optionally pinned to a snapshot id for time travel. Read-only.
- **`IcebergTableProviderFactory`**: handles `CREATE EXTERNAL TABLE t STORED AS ICEBERG LOCATION '<metadata.json>'` and produces a static provider. Bare names get the `default` namespace. Schema, partition, and order clauses are rejected.

### Read path and filter pushdown

`IcebergTableScan` converts the projection to column names and the filters to a single Iceberg `Predicate` (`physical_plan/expr_to_predicate.rs`). It applies `limit` in-stream, has one output partition, and delegates the actual read to iceberg's `TableScan::to_arrow()`.

Both providers report every filter as `Inexact`, so DataFusion re-applies the original filters after the scan. The pushed-down predicate therefore only has to be *implied by* the original filter. It may match extra rows but must never drop a row that matches. Unconvertible parts are dropped: an `AND` keeps whichever side converted, while an `OR` needs both sides. Preserve this soundness rule when extending `expr_to_predicate.rs`; for example, it is why date casts and some NaN arithmetic are deliberately not pushed down.

### Write path (`IcebergTableProvider::insert_into`)

```
input
 → project_with_partition  partitioned tables only: appends a `_partition` struct column computed by `PartitionExpr`
 → repartition             hash on `_partition` for identity/bucket transforms, otherwise round-robin; uses session target_partitions
 → sort_by_partition       only when table property `write.datafusion.fanout.enabled` is false (default: true)
 → IcebergWriteExec        per partition: `TaskWriter` writes Parquet files and emits JSON-serialized DataFiles in a `data_files` column
 → CoalescePartitionsExec
 → IcebergCommitExec       single partition: one `fast_append` transaction; outputs a `count` row
```

- **`TaskWriter`** picks one of three writers: `UnpartitionedWriter`; `FanoutWriter` (accepts unsorted input, keeps many files open); or `ClusteredWriter` (requires sorted input, which is why the sort step exists).
- **Writer limits:** only Parquet is supported. The writer honors `write.parquet.*` table properties and table encryption. It matches columns by name, because DataFusion batches carry no field IDs.
- **`PartitionExpr`** is public and retains its partition spec and table schema so that distributed engines can serialize them and rebuild the expression on workers with `try_new`. Its equality and hash are by value; don't go back to pointer equality.

### Errors

Library code returns `datafusion::error::Result`. Convert iceberg errors with `to_datafusion_error`, which encodes them as `DataFusionError::Context("IcebergError(<Kind>)", Execution(msg))`. `from_datafusion_error` reverses the encoding and keeps the `ErrorKind`. The kind list in `error.rs` is hand-maintained, so a new iceberg `ErrorKind` must be added there or it will round-trip as `Unexpected`.

## sqllogictest harness (`crates/sqllogictest`)

- It's a custom `harness = false` runner using libtest-mimic. Each TOML file in `testdata/schedules/` is one test: it declares engines and ordered `[[steps]]` that point at `.slt` files under `testdata/slts/`. An `.slt` file that no schedule references never runs.
- Each schedule gets a fresh `SessionContext` (`target_partitions = 4`, information_schema enabled) and a fresh in-memory Iceberg catalog. The catalog is registered as `default` with namespace `default`, so SQL addresses tables as `default.default.<table>`. `datafusion` is the only engine type. The `catalog` setting in a schedule is parsed but ignored.
- `engine/datafusion.rs` pre-creates the tables SQL can't express yet:
  - `test_partitioned_table`: identity-partitioned on `category`
  - `test_binary_table`
  - `test_encrypted_round_trip`: format V3, encrypted with in-memory KMS key `test-master-key`

  All other tables are created by `CREATE TABLE` inside the `.slt` files.
- Several `.slt` files assert `EXPLAIN` output, such as `IcebergTableScan projection:[...] predicate:[...] limit:[...]`. Changes to the scan's `DisplayAs`, to pushdown, or to the DataFusion version require updating those expectations by hand; the harness has no auto-complete mode.

## Dependencies

- **iceberg-rust:** `iceberg` and `iceberg-catalog-rest` are git dependencies on `apache/iceberg-rust`, pinned to the same `rev` in the root `Cargo.toml`. The writers, partition splitting, table properties, encryption, and metadata tables all come from there. A missing API usually has to land upstream first; then bump both `rev`s together.
- **Arrow/Parquet version:** DataFusion, iceberg-rust, and `crates/datafusion`'s direct `parquet` dependency must all resolve to one Arrow/Parquet version (currently 59.x). For that reason Dependabot skips semver-major `datafusion*`/`parquet` bumps. Do those upgrades manually, together with a compatible iceberg-rust `rev`.

## Conventions

- New files need the ASF license header; copy it from a neighbouring file.
- Any GitHub Action added to a workflow must be on the ASF allowlist; the `asf-allowlist-check` workflow enforces this.
