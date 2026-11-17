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

use std::sync::LazyLock;

use datafusion::common::{config_namespace, extensions_options};
use datafusion::config::ConfigExtension;
use datafusion::prelude::SessionConfig;

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
    ///     .set_bool("iceberg.execution.eager_file_planning", true);
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
