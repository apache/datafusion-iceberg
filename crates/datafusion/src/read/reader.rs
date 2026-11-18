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

use std::fmt::{Debug, Formatter};

use datafusion::execution::TaskContext;
use iceberg::arrow::ArrowReader;
use iceberg::io::FileIO;
use iceberg::table::Table;

/// Table-backed defaults remain execution-context dependent, as with
/// TableScan::to_arrow. An explicitly supplied reader is never reconfigured.
#[derive(Clone)]
pub(super) enum ScanReader {
    Table(Table),
    Provided {
        file_io: FileIO,
        reader: ArrowReader,
    },
}

impl ScanReader {
    pub(super) fn file_io(&self) -> &FileIO {
        match self {
            Self::Table(table) => table.file_io(),
            Self::Provided { file_io, .. } => file_io,
        }
    }

    pub(super) fn reader(&self, context: &TaskContext) -> ArrowReader {
        match self {
            Self::Table(table) => table
                .reader_builder()
                .with_batch_size(context.session_config().batch_size())
                .with_row_selection_enabled(
                    context
                        .session_config()
                        .options()
                        .execution
                        .parquet
                        .enable_page_index,
                )
                .build(),
            // ArrowReader::clone shares the cached delete-file loader.
            Self::Provided { reader, .. } => reader.clone(),
        }
    }
}

impl Debug for ScanReader {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Table(table) => f.debug_tuple("Table").field(table).finish(),
            Self::Provided { file_io, .. } => f
                .debug_struct("Provided")
                .field("file_io", file_io)
                .finish_non_exhaustive(),
        }
    }
}
