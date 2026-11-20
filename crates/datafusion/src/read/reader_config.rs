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

use datafusion::prelude::SessionConfig;
use iceberg::arrow::{ArrowReader, ArrowReaderBuilder};

use crate::IcebergDataFusionConfig;

/// Applies session settings using the reader API in the pinned iceberg-rust.
/// Switch this mapping to ArrowReaderOptions when the dependency includes #3514.
pub(super) fn build_reader(
    reader: ArrowReaderBuilder,
    config: &SessionConfig,
) -> ArrowReader {
    let iceberg = IcebergDataFusionConfig::from_config_or_default(config);
    let parquet = &config.options().execution.parquet;
    reader
        .with_batch_size(config.batch_size())
        .with_data_file_concurrency_limit(
            iceberg.execution.data_file_concurrency_limit.max(1),
        )
        .with_row_group_filtering_enabled(parquet.pruning)
        .with_row_selection_enabled(parquet.enable_page_index)
        .build()
}
