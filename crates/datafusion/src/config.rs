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

use std::any::Any;

use datafusion::common::{Result, config_err, extensions_options};
use datafusion::config::{ConfigEntry, ConfigExtension, ExtensionOptions};

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
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct IcebergDataFusionConfig {
    /// Options for executing table scans.
    pub execution: IcebergExecutionConfig,
}

// The nested extensions_options! macro drops parent names from entries(). Keep
// the group here so every listed key can be passed back to set(), including when
// consumers propagate options to another session or process.
impl ExtensionOptions for IcebergDataFusionConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let Some(key) = key.strip_prefix("execution.") else {
            return config_err!("Unknown Iceberg configuration key: {key}");
        };
        self.execution.set(key, value)
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        self.execution
            .entries()
            .into_iter()
            .map(|mut entry| {
                entry.key = format!("execution.{}", entry.key);
                entry
            })
            .collect()
    }
}

extensions_options! {
    /// Options for executing Iceberg scans, under `iceberg.execution`.
    pub struct IcebergExecutionConfig {
        /// Discover and retain file tasks in TableProvider::scan instead of at execution.
        /// Selected files are grouped into at most the session's target partitions.
        /// Planning needs an I/O-capable caller runtime and reserves query memory
        /// for the descriptors. The default remains lazy, single-partition discovery.
        pub eager_file_planning: bool, default = false
    }
}

impl ConfigExtension for IcebergDataFusionConfig {
    const PREFIX: &'static str = "iceberg";
}

#[cfg(test)]
mod tests {
    use super::IcebergDataFusionConfig;
    use datafusion::common::Result;
    use datafusion::config::ExtensionOptions;
    use datafusion::prelude::{SessionConfig, SessionContext};

    #[test]
    fn listed_options_round_trip_with_their_group_names() -> Result<()> {
        let mut configured = IcebergDataFusionConfig::default();
        configured.execution.eager_file_planning = true;
        let mut restored = IcebergDataFusionConfig::default();
        let entries = configured.entries();
        assert_eq!(entries[0].key, "execution.eager_file_planning");
        for entry in entries {
            if let Some(value) = entry.value {
                restored.set(&entry.key, &value)?;
            }
        }
        assert!(restored.execution.eager_file_planning);
        Ok(())
    }

    #[tokio::test]
    async fn execution_options_can_be_set_through_sql() -> Result<()> {
        let ctx = SessionContext::new_with_config(
            SessionConfig::new()
                .with_option_extension(IcebergDataFusionConfig::default()),
        );
        assert!(
            !ctx.state()
                .config_options()
                .extensions
                .get::<IcebergDataFusionConfig>()
                .unwrap()
                .execution
                .eager_file_planning
        );
        ctx.sql("SET iceberg.execution.eager_file_planning = true")
            .await?
            .collect()
            .await?;
        assert!(
            ctx.state()
                .config_options()
                .extensions
                .get::<IcebergDataFusionConfig>()
                .unwrap()
                .execution
                .eager_file_planning
        );
        Ok(())
    }
}
