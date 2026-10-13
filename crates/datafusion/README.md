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

# Apache Iceberg DataFusion Integration

This crate contains the integration of Apache DataFusion and Apache Iceberg.

## Optional file planning before execution

By default, the Iceberg reader discovers files when execution starts. To select
`FileScanTask`s during `TableProvider::scan()` instead, register the planning
options on the DataFusion session before planning a query:

```rust
use datafusion::prelude::SessionConfig;
use iceberg_datafusion::IcebergPlanningConfig;

let config = SessionConfig::new()
    .with_option_extension(IcebergPlanningConfig::default())
    .set_bool("iceberg_local.plan_files", true);
```

The option defaults to `false`. When enabled, each physical scan retains its
selected tasks within the query's memory pool, failing rather than truncating
if more than `iceberg_local.planning_max_files` (default: 100,000) are selected.
Tasks are assigned round-robin to at most the session's `target_partitions`;
an empty scan exposes one empty partition. This does **not** read Parquet data
during planning. The physical scan continues to report unknown statistics.
`iceberg_local` is distinct from DataFusion Distributed's `iceberg` options so
both can be registered in the same session.

The local reader is constructed with the existing public
`Table::reader_builder()` API, which uses the table's `FileIO` and runtime.
This matches the current integration's reader defaults because it does not
set reader options on `TableScan`: the snapshot, projection, and predicate are
carried by the tasks. For multiple output partitions, each partition reads one
file at a time instead of using the lazy scan's per-reader file concurrency.
**Reader settings are not transferred separately from the scan.**
If scan-specific batch size, data-file concurrency, row-group, row-selection,
or bloom-filter settings are introduced later, the lazy and planned readers
could diverge. A follow-up should expose portable scan-level reader options
and apply them to a locally constructed builder in iceberg-rust, with
`TableScan::to_arrow()` using the same mapping. Distributed consumers should
construct readers using worker-local IO and runtimes rather than transport a
builder or reader.

Planning the whole file list earlier adds metadata I/O and retained-task memory
even when a query stops early (for example, `LIMIT`), and equivalent provider
`scan()` calls currently repeat planning. No performance improvement is claimed
by this option alone. Delete-bearing eager scans and cross-partition delete-cache
reuse also need dedicated validation before enabling it by default.
