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

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::Result;
use datafusion::prelude::Expr;
use iceberg::table::Table;

use crate::IcebergDataFusionConfig;
use crate::read::{
    EagerFileScanTaskProvider, FileScanTaskProvider, IcebergDataSource,
    IcebergDataSourceBuilder, LazyFileScanTaskProvider,
};

/// Table planning is the only path that needs metadata. Both table providers
/// select their snapshot before entering here; reconstruction skips this helper.
pub(super) async fn scan_table(
    table: &Table,
    schema: SchemaRef,
    state: &dyn Session,
    projection: Option<&Vec<usize>>,
    filters: &[Expr],
    fetch: Option<usize>,
    snapshot_id: Option<i64>,
) -> Result<IcebergDataSource> {
    let schema = match projection {
        Some(projection) => Arc::new(schema.project(projection)?),
        None => schema,
    };
    let columns = schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let discovery = Arc::new(
        LazyFileScanTaskProvider::new(table.clone(), columns)
            .with_snapshot_id(snapshot_id)
            .with_filters(filters),
    );
    let config = state.config();
    let reader = table
        .reader_builder()
        .with_batch_size(config.batch_size())
        .with_row_selection_enabled(config.options().execution.parquet.enable_page_index)
        // DataFusion already runs output partitions concurrently.
        .with_data_file_concurrency_limit(1)
        .build();
    let provider: Arc<dyn FileScanTaskProvider> =
        if IcebergDataFusionConfig::from_config_or_default(config)
            .execution
            .eager_file_planning
        {
            Arc::new(EagerFileScanTaskProvider::plan(discovery, state.task_ctx()).await?)
        } else {
            discovery
        };
    IcebergDataSourceBuilder::new(schema, table.file_io().clone(), reader)
        .with_file_scan_task_provider(provider)
        .with_fetch(fetch)
        .build()
}
