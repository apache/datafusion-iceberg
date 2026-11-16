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

//! Session-configured construction of table scans and already-planned scans.

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::Result;
use datafusion::datasource::source::DataSource;
use datafusion::execution::TaskContext;
use datafusion::prelude::Expr;
use iceberg::arrow::ArrowReader;
use iceberg::expr::Predicate;
use iceberg::io::FileIO;
use iceberg::scan::FileScanTask;
use iceberg::table::Table;

use super::eager::EagerIcebergDataSource;
use super::file_planning::{PlannedFileScan, flatten_groups};
use super::predicate::convert_filters_to_predicate;
use super::scan::{IcebergDataSource, build_table_scan};
use crate::IcebergDataFusionConfig;

/// Builds a scan using the session settings and memory pool in a task context.
///
/// Obtain the context from `SessionContext::task_ctx()` or the `Session` passed to
/// a table provider. Eager planning, target partitions, batch size and page-index
/// settings come from that context, not separate builder options.
///
/// Select either a table or already-planned files after supplying the schema.
/// Only table input exposes discovery projection, snapshot and filter settings.
/// Table scans finish asynchronously; reconstruction finishes synchronously and
/// never discovers files. Both return an `Arc<dyn DataSource>`.
///
/// ```
/// # use datafusion::arrow::datatypes::SchemaRef;
/// # use datafusion::common::Result;
/// # use datafusion::datasource::source::DataSource;
/// # use datafusion::execution::TaskContext;
/// # use iceberg::table::Table;
/// # use iceberg_datafusion::read::IcebergDataSourceBuilder;
/// # use std::sync::Arc;
/// async fn scan(table: Table, schema: SchemaRef, context: &TaskContext)
///     -> Result<Arc<dyn DataSource>>
/// {
///     IcebergDataSourceBuilder::new(context)
///         .with_schema(schema)
///         .with_table(table)
///         .build()
///         .await
/// }
/// ```
///
/// Reconstruction uses the same entry point but has no asynchronous discovery:
/// ```
/// # use datafusion::arrow::datatypes::SchemaRef;
/// # use datafusion::common::Result;
/// # use datafusion::datasource::source::DataSource;
/// # use datafusion::execution::TaskContext;
/// # use iceberg::arrow::ArrowReader;
/// # use iceberg::io::FileIO;
/// # use iceberg::scan::FileScanTask;
/// # use iceberg_datafusion::read::IcebergDataSourceBuilder;
/// # use std::sync::Arc;
/// fn restore(context: &TaskContext, schema: SchemaRef,
///     groups: Vec<Vec<FileScanTask>>, file_io: FileIO, reader: ArrowReader)
///     -> Result<Arc<dyn DataSource>>
/// {
///     IcebergDataSourceBuilder::new(context)
///         .with_schema(schema)
///         .with_planned_files(groups)
///         .with_file_io(file_io)
///         .with_reader(reader)
///         .build()
/// }
/// ```
///
/// Planned input cannot request snapshot discovery:
/// ```compile_fail,E0599
/// # use datafusion::arrow::datatypes::SchemaRef;
/// # use datafusion::execution::TaskContext;
/// # use iceberg::arrow::ArrowReader;
/// # use iceberg::io::FileIO;
/// # use iceberg::scan::FileScanTask;
/// # use iceberg_datafusion::read::IcebergDataSourceBuilder;
/// fn invalid(context: &TaskContext, schema: SchemaRef,
///     groups: Vec<Vec<FileScanTask>>, file_io: FileIO, reader: ArrowReader)
/// {
///     IcebergDataSourceBuilder::new(context)
///         .with_schema(schema)
///         .with_planned_files(groups)
///         .with_file_io(file_io)
///         .with_reader(reader)
///         .with_snapshot_id(Some(1));
/// }
/// ```
///
/// Reconstruction cannot finish without a process-local reader:
/// ```compile_fail,E0599
/// # use datafusion::arrow::datatypes::SchemaRef;
/// # use datafusion::execution::TaskContext;
/// # use iceberg::io::FileIO;
/// # use iceberg::scan::FileScanTask;
/// # use iceberg_datafusion::read::IcebergDataSourceBuilder;
/// fn invalid(context: &TaskContext, schema: SchemaRef,
///     groups: Vec<Vec<FileScanTask>>, file_io: FileIO)
/// {
///     IcebergDataSourceBuilder::new(context)
///         .with_schema(schema)
///         .with_planned_files(groups)
///         .with_file_io(file_io)
///         .build();
/// }
/// ```
#[derive(Clone)]
#[must_use]
pub struct IcebergDataSourceBuilder<'a, Input = NeedsSchema> {
    context: &'a TaskContext,
    input: Input,
    fetch: Option<usize>,
}

impl<Input> Debug for IcebergDataSourceBuilder<'_, Input> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergDataSourceBuilder")
            .field("fetch", &self.fetch)
            .finish_non_exhaustive()
    }
}

/// Initial builder state, before supplying the output schema.
#[derive(Clone, Debug)]
pub struct NeedsSchema;

/// Builder state before selecting table discovery or already-planned files.
#[derive(Clone, Debug)]
pub struct NeedsInput {
    schema: SchemaRef,
}

/// A table discovery request. Its settings cannot be applied to planned files.
#[derive(Clone)]
pub struct TableScanInput {
    schema: SchemaRef,
    table: Table,
    projection: Option<Vec<usize>>,
    snapshot_id: Option<i64>,
    predicate: Option<Predicate>,
    reader: Option<ArrowReader>,
}

/// Planned input whose FileIO and reader are supplied in subsequent steps.
#[derive(Clone)]
pub struct PlannedFilesInput<Reader> {
    schema: SchemaRef,
    groups: Vec<Vec<FileScanTask>>,
    reader: Reader,
}

/// Planned files still require process-local file access.
#[derive(Clone, Debug)]
pub struct NeedsFileIo;

/// Planned files have file access and still require a configured reader.
#[derive(Clone, Debug)]
pub struct NeedsReader {
    file_io: FileIO,
}

/// Complete process-local file access for planned input.
#[derive(Clone)]
pub struct PlannedReader {
    file_io: FileIO,
    reader: ArrowReader,
}

impl<'a> IcebergDataSourceBuilder<'a> {
    /// Starts construction with the session's execution context.
    pub fn new(context: &'a TaskContext) -> Self {
        Self {
            context,
            input: NeedsSchema,
            fetch: None,
        }
    }

    /// Supplies the provider's schema for discovery, or the already-projected
    /// output schema when reconstructing planned files.
    pub fn with_schema(
        self,
        schema: SchemaRef,
    ) -> IcebergDataSourceBuilder<'a, NeedsInput> {
        self.map_input(|_| NeedsInput { schema })
    }
}

impl<'a, Input> IcebergDataSourceBuilder<'a, Input> {
    fn map_input<Next>(
        self,
        f: impl FnOnce(Input) -> Next,
    ) -> IcebergDataSourceBuilder<'a, Next> {
        IcebergDataSourceBuilder {
            context: self.context,
            input: f(self.input),
            fetch: self.fetch,
        }
    }

    /// Replaces the execution context, including session settings and query memory
    /// pool. Construction has not discovered files or reserved descriptor memory yet.
    pub fn with_context<'b>(
        self,
        context: &'b TaskContext,
    ) -> IcebergDataSourceBuilder<'b, Input> {
        IcebergDataSourceBuilder {
            context,
            input: self.input,
            fetch: self.fetch,
        }
    }

    /// Stops each output partition after `fetch` rows, or reads all rows for `None`.
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        self
    }
}

impl<'a> IcebergDataSourceBuilder<'a, NeedsInput> {
    /// Selects discovery from a table. The session chooses eager or lazy planning.
    pub fn with_table(
        self,
        table: Table,
    ) -> IcebergDataSourceBuilder<'a, TableScanInput> {
        self.map_input(|input| TableScanInput {
            schema: input.schema,
            table,
            projection: None,
            snapshot_id: None,
            predicate: None,
            reader: None,
        })
    }

    /// Selects already-planned groups, each representing one output partition.
    /// Supply FileIO and a process-local reader next. No table or catalog is needed.
    /// Supplied group order and empty individual groups are preserved.
    pub fn with_planned_files(
        self,
        groups: Vec<Vec<FileScanTask>>,
    ) -> IcebergDataSourceBuilder<'a, PlannedFilesInput<NeedsFileIo>> {
        self.map_input(|input| PlannedFilesInput {
            schema: input.schema,
            groups,
            reader: NeedsFileIo,
        })
    }
}

impl<'a> IcebergDataSourceBuilder<'a, PlannedFilesInput<NeedsFileIo>> {
    /// Supplies file access matching the planned descriptors and local reader.
    pub fn with_file_io(
        self,
        file_io: FileIO,
    ) -> IcebergDataSourceBuilder<'a, PlannedFilesInput<NeedsReader>> {
        self.map_input(|input| PlannedFilesInput {
            schema: input.schema,
            groups: input.groups,
            reader: NeedsReader { file_io },
        })
    }
}

impl<'a> IcebergDataSourceBuilder<'a, PlannedFilesInput<NeedsReader>> {
    /// Supplies the process-local reader. Its settings and shared delete loader
    /// are preserved; it must use the FileIO supplied in the preceding step.
    pub fn with_reader(
        self,
        reader: ArrowReader,
    ) -> IcebergDataSourceBuilder<'a, PlannedFilesInput<PlannedReader>> {
        self.map_input(|input| PlannedFilesInput {
            schema: input.schema,
            groups: input.groups,
            reader: PlannedReader {
                file_io: input.reader.file_io,
                reader,
            },
        })
    }
}

impl IcebergDataSourceBuilder<'_, PlannedFilesInput<PlannedReader>> {
    /// Reconstructs a source synchronously, without discovery or file I/O.
    ///
    /// # Errors
    /// Returns an error for zero groups or insufficient descriptor memory. Empty
    /// individual groups are valid. Tasks must match the supplied projected schema.
    pub fn build(self) -> Result<Arc<dyn DataSource>> {
        let input = self.input;
        let (tasks, partitions) = flatten_groups(input.groups)?;
        let files = PlannedFileScan::from_tasks(tasks, self.context)?;
        Ok(Arc::new(EagerIcebergDataSource::from_planned_files(
            input.schema,
            files,
            partitions,
            input.reader.file_io,
            input.reader.reader,
            self.fetch,
        )))
    }
}

impl IcebergDataSourceBuilder<'_, TableScanInput> {
    /// Selects indices into the provider schema, or all columns for `None`.
    pub fn with_projection(mut self, projection: Option<&Vec<usize>>) -> Self {
        self.input.projection = projection.cloned();
        self
    }

    /// Selects a snapshot, or the table's current snapshot for `None`.
    pub fn with_snapshot_id(mut self, snapshot_id: Option<i64>) -> Self {
        self.input.snapshot_id = snapshot_id;
        self
    }

    /// Supplies a predicate for file discovery and reading.
    pub fn with_predicate(mut self, predicate: Option<Predicate>) -> Self {
        self.input.predicate = predicate;
        self
    }

    /// Pushes down convertible parts of the filters. DataFusion must reapply
    /// the original filters because pushdown can admit additional rows.
    pub fn with_filters(self, filters: &[Expr]) -> Self {
        self.with_predicate(convert_filters_to_predicate(filters))
    }

    /// Overrides the default reader, preserving all of its settings in either
    /// mode. Its file access must match the discovery table's FileIO.
    pub fn with_reader(mut self, reader: ArrowReader) -> Self {
        self.input.reader = Some(reader);
        self
    }

    /// Builds the source selected by the session's Iceberg configuration.
    ///
    /// Eager mode completes file discovery and reserves descriptor memory before
    /// returning. Lazy mode defers discovery to execution. Neither reads data files
    /// here. Eager collection uses the caller's I/O-capable runtime; dropping the
    /// future releases its retained descriptors and reservation.
    ///
    /// # Errors
    /// Returns an error for invalid projection, discovery failure or insufficient
    /// query memory. Lazy mode does not perform discovery I/O during this call.
    pub async fn build(self) -> Result<Arc<dyn DataSource>> {
        let input = self.input;
        let schema = match input.projection {
            Some(indices) => Arc::new(input.schema.project(&indices)?),
            None => input.schema,
        };
        let projection = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();
        let eager = self
            .context
            .session_config()
            .options()
            .extensions
            .get::<IcebergDataFusionConfig>()
            .is_some_and(|config| config.execution.eager_file_planning);
        if !eager {
            let reader = configured_reader(input.reader, &input.table, self.context, 1);
            return Ok(Arc::new(IcebergDataSource {
                table: input.table,
                schema,
                projection,
                snapshot_id: input.snapshot_id,
                predicate: input.predicate,
                fetch: self.fetch,
                reader,
                metrics: Default::default(),
            }));
        }
        let scan = build_table_scan(
            &input.table,
            input.snapshot_id,
            Some(&projection),
            input.predicate.as_ref(),
        )?;
        let files = PlannedFileScan::collect(scan, self.context).await?;
        let partitions = files
            .tasks()
            .len()
            .max(1)
            .min(self.context.session_config().target_partitions().max(1));
        let reader =
            configured_reader(input.reader, &input.table, self.context, partitions);
        let mut groups = vec![Vec::new(); partitions];
        for index in 0..files.tasks().len() {
            groups[index % partitions].push(index);
        }
        Ok(Arc::new(EagerIcebergDataSource::from_planned_files(
            schema,
            files,
            groups,
            input.table.file_io().clone(),
            reader,
            self.fetch,
        )))
    }
}

fn configured_reader(
    reader: Option<ArrowReader>,
    table: &Table,
    context: &TaskContext,
    partitions: usize,
) -> ArrowReader {
    reader.unwrap_or_else(|| {
        let reader = table
            .reader_builder()
            .with_batch_size(context.session_config().batch_size())
            .with_row_selection_enabled(
                context
                    .session_config()
                    .options()
                    .execution
                    .parquet
                    .enable_page_index,
            );
        let reader = if partitions > 1 {
            reader.with_data_file_concurrency_limit(1)
        } else {
            reader
        };
        reader.build()
    })
}
