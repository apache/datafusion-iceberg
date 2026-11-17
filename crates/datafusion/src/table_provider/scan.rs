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

use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::common::Result;

use crate::IcebergDataFusionConfig;
use crate::read::{
    EagerFileScanTaskProvider, FileScanTaskProvider, IcebergDataSource,
    IcebergDataSourceBuilder, LazyFileScanTaskProvider,
};

/// Assembles a source from configured discovery and session reader settings.
/// Projection, snapshot and filters have already been set by the caller.
pub(super) async fn build_source(
    discovery: LazyFileScanTaskProvider,
    state: &dyn Session,
    fetch: Option<usize>,
) -> Result<IcebergDataSource> {
    let schema = discovery.schema();
    let file_io = discovery.table().file_io().clone();
    let config = state.config();
    let reader = discovery
        .table()
        .reader_builder()
        .with_batch_size(config.batch_size())
        .with_row_selection_enabled(config.options().execution.parquet.enable_page_index)
        // DataFusion already runs output partitions concurrently.
        .with_data_file_concurrency_limit(1)
        .build();
    let discovery = Arc::new(discovery);
    let provider: Arc<dyn FileScanTaskProvider> =
        if IcebergDataFusionConfig::from_config_or_default(config)
            .execution
            .eager_file_planning
        {
            Arc::new(EagerFileScanTaskProvider::plan(discovery, state.task_ctx()).await?)
        } else {
            discovery
        };
    IcebergDataSourceBuilder::new(schema, file_io, reader)
        .with_file_scan_task_provider(provider)
        .with_fetch(fetch)
        .build()
}
