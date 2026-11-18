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

## File task providers

`IcebergDataSource` reads a `FileScanTaskProvider` using a shared, process-local
reader. It does not require a table.

`FileScanTaskProviderBuilder::new(context, table, schema)` receives configuration
at construction. Set `with_projection`, `with_snapshot_id` and `with_filters` as
needed, then await `build()` for the projected schema and task provider together.
The builder selects eager or lazy discovery from
`iceberg.execution.eager_file_planning`; callers do not branch on that setting.
Lazy construction does not read manifests. Eager construction consumes the same
discovery through `LazyFileScanTaskProvider::into_eager`, retaining only file tasks.

`IcebergDataSource::builder(config, schema, file_io, reader_builder)` applies session
reader settings at construction. Pass storage and the reader builder directly
from the table, not through its discovery provider. The retained reader uses the
session batch size and page-index setting, with one concurrent file per partition.
`with_reader` can override those defaults with an existing configured reader.

Already-planned groups can be supplied with `EagerFileScanTaskProvider::try_new`
and `with_file_scan_task_provider`. Source construction remains synchronous and
performs no discovery, even when eager planning is enabled. It requires no current
runtime: the supplied reader builder already holds the process-local runtime.
Its FileIO must match the source's FileIO, and tasks must produce the supplied
projected schema. See the builder's API documentation for a reconstruction example.
Custom task providers use the same source; no factory or transform is required.
