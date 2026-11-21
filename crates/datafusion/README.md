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

### Tracing file discovery

With an INFO-level `tracing` subscriber, a scan emits `iceberg.plan_scan` while
building the task provider. With `iceberg.execution.eager_file_planning` enabled,
`iceberg.plan_files` is a child of that span. Otherwise it appears during execution,
when the reader opens the lazy file-task stream. Its lifetime can include time
between polls; it is not a measure of S3 request latency. On exhaustion it records
`file_tasks`, the number of successfully discovered tasks.
