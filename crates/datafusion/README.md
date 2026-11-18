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

## Lazy file-task delivery

`IcebergDataSource<T = TableScanTaskSource>` separates delivery of native Iceberg
`FileScanTask`s from reading them. Existing table providers and
`IcebergDataSource::builder(table, schema).build()?` keep single-partition lazy
discovery. The default reader still uses the execution session's batch size and
page-index setting, and the table's FileIO and runtime.

Implement `FileScanTaskSource` to supply a positive, stable number of local
partitions and a task stream for each partition. Tasks retain their schema,
projection, predicate, ranges and delete-file references. The implementation can
receive tasks from elsewhere; this crate does not choose workers, serialize tasks
or require a transport.

```rust,no_run
# use datafusion::common::Result;
# use datafusion::arrow::datatypes::SchemaRef;
# use iceberg::arrow::ArrowReader;
# use iceberg::io::FileIO;
use iceberg_datafusion::read::{FileScanTaskSource, IcebergDataSource};

fn delivered_scan<T: FileScanTaskSource>(
    schema: SchemaRef,
    tasks: T,
    file_io: FileIO,
    reader: ArrowReader,
) -> Result<IcebergDataSource<T>> {
    IcebergDataSource::from_task_source(schema, tasks, file_io, reader, Some(100))
}
```

`from_task_source` needs no table or catalog and performs no discovery or I/O.
The output schema must match the supplied tasks' projection, and the supplied
FileIO must match the reader's storage configuration. Explicit reader settings
are authoritative; clones share that reader's cached delete-file loader.

Alternatively, use `source.with_task_source(tasks)?` to replace a table-backed
source's discovery while preserving its schema, reader settings and fetch. A
replacement must supply compatible tasks and storage access; it may change the
number of local partitions. Task-source clones must preserve the assignment but
reset source-local execution state. Streams retain their resources independently
of the source and release them when dropped.

Fetch is a **per-local-partition** limit, not a global query limit. Reaching it
drops that partition's reader stream; fetch zero opens no task stream. These APIs
do not promise bounded buffering or strict backpressure. Statistics remain
unknown, as for the existing lazy source; this change adds no statistics
computation, runtime override, provider hook or SQL snapshot option.
