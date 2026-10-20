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

//! Session options for Iceberg tables, set with `SET iceberg.<group>.<option>`.

use datafusion::common::config::ConfigExtension;
use datafusion::common::extensions_options;

extensions_options! {
    /// Session options for reading Iceberg tables through DataFusion.
    ///
    /// `SET iceberg.<group>.<option> = <value>` only works once this extension
    /// is registered on the session, with
    /// [`SessionConfig::with_option_extension`](datafusion::prelude::SessionConfig::with_option_extension).
    /// Without it, every option keeps its default. DataFusion lists these
    /// options in `information_schema.df_settings` by their last name alone,
    /// such as `max_merge_files_per_partition`, and `SHOW` and `RESET` do not
    /// support them.
    ///
    /// ```
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> datafusion::error::Result<()> {
    /// use datafusion::prelude::{SessionConfig, SessionContext};
    /// use iceberg_datafusion::IcebergDataFusionConfig;
    ///
    /// let config =
    ///     SessionConfig::new().with_option_extension(IcebergDataFusionConfig::default());
    /// let ctx = SessionContext::new_with_config(config);
    /// ctx.sql("SET iceberg.execution.preserve_data_ordering = true")
    ///     .await?
    ///     .collect()
    ///     .await?;
    ///
    /// let state = ctx.state();
    /// let options = state.config().options();
    /// let iceberg = options.extensions.get::<IcebergDataFusionConfig>().unwrap();
    /// assert!(iceberg.execution.preserve_data_ordering);
    /// # Ok(())
    /// # }
    /// ```
    pub struct IcebergDataFusionConfig {
        /// Options for executing table scans.
        pub execution: IcebergExecutionConfig, default = IcebergExecutionConfig::default()
    }
}

extensions_options! {
    /// Options for executing Iceberg table scans, under `iceberg.execution`.
    pub struct IcebergExecutionConfig {
        /// When true, a scan lists its data files while it is planned and, if
        /// every file records the same sort order, reports that order to
        /// DataFusion so that it can skip sorting the scan's output. The scan
        /// then merges the files in that order, with every file open at once.
        ///
        /// Only the leading sort fields that are identity transforms of
        /// projected top-level columns are reported, stopping at the first
        /// floating point or UUID field, whose orders in Iceberg writers and in
        /// DataFusion can differ.
        ///
        /// The scan reports each sort field's null order as the table declares
        /// it. Writers commonly sort ascending fields with nulls first, the
        /// default of Iceberg Java's `SortOrderBuilder.asc`, while DataFusion's
        /// `ORDER BY c` puts them last. On a nullable column sorted that way,
        /// only `ORDER BY c NULLS FIRST`, or a session with
        /// `datafusion.sql_parser.default_null_ordering = 'nulls_min'`, can
        /// skip its sort. Other queries sort the merged output again.
        pub preserve_data_ordering: bool, default = false
        /// The most data files a scan merges in one output partition to
        /// preserve their sort order. A scan with more files in a partition
        /// reports no order, and DataFusion sorts its output instead, which
        /// can spill to disk where the merge cannot.
        pub max_merge_files_per_partition: usize, default = 64
    }
}

impl ConfigExtension for IcebergDataFusionConfig {
    const PREFIX: &'static str = "iceberg";
}
