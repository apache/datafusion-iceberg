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
    .set_bool("iceberg.eager_file_planning", true);
```

### Behavior

- Lazy planning is the default. Eager planning selects files before execution;
  it reads metadata, not Parquet data.
- Selected tasks are assigned round-robin to at most `target_partitions`.
  An empty scan has one empty partition. Each eager partition reads one file
  at a time when there are multiple output partitions.
- Retained tasks have an estimated reservation in the query memory pool.
  Exceeding the pool fails planning; there is no file-count cap or truncation.
- Both modes use the same reader implementation. Reader clones share the
  delete loader across partitions. Scan statistics remain unknown.

### Trade-offs

Eager planning completes file discovery and retains tasks even for queries
that stop early. Repeated provider scans repeat planning. The planning caller
must support metadata I/O: Iceberg's initial manifest-list read still runs on
that runtime, even when the table has a separate I/O runtime.

Register only one configuration extension under the `iceberg` prefix per session.
DataFusion Distributed currently uses that same prefix for its own options.
