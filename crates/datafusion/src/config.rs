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
    /// Without it, every option keeps its default.
    pub struct IcebergDataFusionConfig {
        /// Options for planning table scans.
        pub planning: IcebergPlanningConfig, default = IcebergPlanningConfig::default()
    }
}

extensions_options! {
    /// Options for planning Iceberg table scans, under `iceberg.planning`.
    pub struct IcebergPlanningConfig {
        /// When true, a scan lists its data files while it is planned and, if
        /// every file records the same sort order, reports that order to
        /// DataFusion so that it can skip sorting the scan's output. The scan
        /// then merges the files in that order, with every file open at once.
        ///
        /// Only the leading sort fields that are identity transforms of
        /// projected top-level columns are reported, stopping at the first
        /// floating point or UUID field, whose orders in Iceberg writers and in
        /// DataFusion can differ.
        pub preserve_data_ordering: bool, default = false
        /// The most data files a scan merges to preserve their sort order.
        /// A scan of more files reports no order, and DataFusion sorts its
        /// output instead, which can spill to disk where the merge cannot.
        pub max_merge_files: usize, default = 64
    }
}

impl ConfigExtension for IcebergDataFusionConfig {
    const PREFIX: &'static str = "iceberg";
}
