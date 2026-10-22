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

use datafusion::common::extensions_options;
use datafusion::config::ConfigExtension;

extensions_options! {
    /// Opt-in file planning during physical planning for local Iceberg scans.
    pub struct IcebergPlanningConfig {
        /// Discover and retain file tasks in TableProvider::scan instead of at execution.
        pub eager_file_planning: bool, default = false
    }
}

impl ConfigExtension for IcebergPlanningConfig {
    const PREFIX: &'static str = "iceberg";
}
