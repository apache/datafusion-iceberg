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

use std::sync::{Arc, LazyLock};

use datafusion::common::{config_namespace, extensions_options};
use datafusion::config::ConfigExtension;
use datafusion::prelude::SessionConfig;
use iceberg::arrow::{ArrowReaderOptions, ParquetReadOptions};

extensions_options! {
    /// Session options for Iceberg tables, under `iceberg.<group>.<option>`.
    ///
    /// Register this extension before setting Iceberg options:
    /// ```
    /// use datafusion::prelude::SessionConfig;
    /// use iceberg_datafusion::IcebergDataFusionConfig;
    ///
    /// let config = SessionConfig::new()
    ///     .with_option_extension(IcebergDataFusionConfig::default())
    ///     .set_bool("iceberg.execution.eager_file_planning", true)
    ///     .set_usize("iceberg.execution.data_file_concurrency_limit", 2);
    /// ```
    ///
    /// Scans honor DataFusion's execution batch size and its Parquet `pruning`,
    /// `enable_page_index`, `bloom_filter_on_read`, and `metadata_size_hint` options.
    /// Bloom-filter reads follow DataFusion's default of enabled.
    ///
    /// Rust callers can supply the complete reader options for scans planned with
    /// this config. These take precedence over individual SQL reader settings,
    /// including DataFusion's batch size and Parquet options. Omitted fields use
    /// Iceberg's defaults. File planning still uses the execution config.
    /// ```
    /// use std::sync::Arc;
    /// use datafusion::prelude::SessionConfig;
    /// use iceberg::arrow::ArrowReaderOptions;
    ///
    /// let session = SessionConfig::new();
    /// let reader = ArrowReaderOptions::builder().with_batch_size(2048).build();
    /// let query_config = session.clone().with_extension(Arc::new(reader));
    /// ```
    pub struct IcebergDataFusionConfig {
        /// Options for executing table scans.
        pub execution: IcebergExecutionConfig, default = IcebergExecutionConfig::default()
    }
}

config_namespace! {
    /// Options for executing Iceberg scans, under `iceberg.execution`.
    pub struct IcebergExecutionConfig {
        /// Discover file tasks in TableProvider::scan instead of at execution, and
        /// split them across up to the session's target partitions.
        pub eager_file_planning: bool, default = false
        /// Maximum concurrent data files per output partition. Zero is treated as one.
        pub data_file_concurrency_limit: usize, default = 1
        /// Iceberg-specific Parquet range-read and index settings.
        pub parquet: IcebergParquetConfig, default = IcebergParquetConfig::default()
    }
}

config_namespace! {
    /// Iceberg Parquet options under `iceberg.execution.parquet`.
    pub struct IcebergParquetConfig {
        /// Maximum gap in bytes between ranges to combine into one read.
        pub range_coalesce_bytes: u64, default = ParquetReadOptions::default().range_coalesce_bytes()
        /// Maximum concurrent range requests per file.
        pub range_fetch_concurrency: usize, default = ParquetReadOptions::default().range_fetch_concurrency()
        /// Preload the Parquet column index.
        pub preload_column_index: bool, default = ParquetReadOptions::default().preload_column_index()
        /// Preload the Parquet offset index.
        pub preload_offset_index: bool, default = ParquetReadOptions::default().preload_offset_index()
        /// Stored page-index preference. Iceberg overrides it per file based on predicates and deletes.
        pub preload_page_index: bool, default = ParquetReadOptions::default().preload_page_index()
    }
}

impl ConfigExtension for IcebergDataFusionConfig {
    const PREFIX: &'static str = "iceberg";
}

static DEFAULT: LazyLock<IcebergDataFusionConfig> =
    LazyLock::new(IcebergDataFusionConfig::default);

impl IcebergDataFusionConfig {
    pub fn from_config_or_default(cfg: &SessionConfig) -> &Self {
        match cfg.options().extensions.get::<Self>() {
            Some(v) => v,
            None => &DEFAULT,
        }
    }
}

/// Resolves the reader options once, when planning the scan.
pub(crate) fn arrow_reader_options(config: &SessionConfig) -> ArrowReaderOptions {
    if let Some(options) = config.get_extension::<ArrowReaderOptions>() {
        return Arc::unwrap_or_clone(options);
    }

    let execution = &IcebergDataFusionConfig::from_config_or_default(config).execution;
    let parquet = &config.options().execution.parquet;
    let io = &execution.parquet;
    let parquet_options = ParquetReadOptions::builder()
        .with_metadata_size_hint(
            parquet
                .metadata_size_hint
                .unwrap_or_else(|| ParquetReadOptions::default().metadata_size_hint()),
        )
        .with_range_coalesce_bytes(io.range_coalesce_bytes)
        .with_range_fetch_concurrency(io.range_fetch_concurrency)
        .with_preload_column_index(io.preload_column_index)
        .with_preload_offset_index(io.preload_offset_index)
        .with_preload_page_index(io.preload_page_index)
        .build();
    ArrowReaderOptions::builder()
        .with_batch_size(config.batch_size())
        .with_row_group_filtering_enabled(parquet.pruning)
        .with_row_selection_enabled(parquet.enable_page_index)
        .with_bloom_filter_enabled(parquet.bloom_filter_on_read)
        .with_data_file_concurrency_limit(execution.data_file_concurrency_limit.max(1))
        .with_parquet_options(parquet_options)
        .build()
}
