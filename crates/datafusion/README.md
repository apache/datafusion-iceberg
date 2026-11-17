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
reader. It does not require a table. `LazyFileScanTaskProvider` retains a table and
scan options and discovers files when its stream is polled. Configure it with
`LazyFileScanTaskProvider::builder(table, schema)`, followed by `with_projection`,
`with_snapshot_id` and `with_filters` as needed. `build()` validates the projection
and keeps the projected output schema and discovery columns together; it does not
read manifests. Use the provider's `schema()` when assembling the source.

`EagerFileScanTaskProvider::plan` collects that same discovery stream before
execution. Table providers choose between these using
`iceberg.execution.eager_file_planning`. Reader configuration and fetch remain
source settings, independent of the discovery builder.

Already-planned groups can be supplied with `EagerFileScanTaskProvider::try_new`
and `IcebergDataSourceBuilder::with_file_scan_task_provider`. Source construction
is synchronous, requires no current runtime, and performs no discovery. Supply an
already-projected schema and a reader configured for the process that will execute
it; the builder's FileIO must match that reader. See the builder's API documentation
for a reconstruction example. Custom task providers use the same source.
