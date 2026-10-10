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

use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::common::Statistics;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::datasource::TableProvider;
use datafusion::datasource::source::DataSource;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayFormatType, Partitioning};
use futures::{StreamExt, TryStreamExt};

use super::table::IcebergMetadataTableProvider;

/// Reads rows from an Iceberg metadata table through DataFusion's `DataSource` API.
#[derive(Debug, Clone)]
pub struct IcebergMetadataDataSource {
    provider: IcebergMetadataTableProvider,
    schema: ArrowSchemaRef,
    projection: Option<Vec<usize>>,
}

impl IcebergMetadataDataSource {
    /// Creates a source with the requested output columns.
    pub fn try_new(
        provider: IcebergMetadataTableProvider,
        projection: Option<&Vec<usize>>,
    ) -> Result<Self> {
        let schema = match projection {
            Some(indices) => Arc::new(provider.schema().project(indices)?),
            None => provider.schema(),
        };
        Ok(Self {
            provider,
            schema,
            projection: projection.cloned(),
        })
    }

    /// The provider this source scans.
    pub fn provider(&self) -> &IcebergMetadataTableProvider {
        &self.provider
    }
}

impl DataSource for IcebergMetadataDataSource {
    fn open(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let fut = self.provider.clone().scan();
        let projection = self.projection.clone();

        // TODO: Push these projections down into the scan layer instead of manually iterating over the result set
        // and applying them.
        // This will be possible once this issue is addressed in iceberg-rust: https://github.com/apache/iceberg-rust/issues/3391
        let stream =
            futures::stream::once(fut)
                .try_flatten()
                .map(move |result| -> Result<_> {
                    let batch = result?;
                    match &projection {
                        Some(indices) => Ok(batch.project(indices)?),
                        None => Ok(batch),
                    }
                });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }

    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "format=iceberg_metadata")
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(self.schema.clone())
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        Ok(Arc::new(Statistics::new_unknown(&self.schema)))
    }

    fn with_fetch(&self, _fetch: Option<usize>) -> Option<Arc<dyn DataSource>> {
        None
    }

    fn fetch(&self) -> Option<usize> {
        None
    }

    fn try_swapping_with_projection(
        &self,
        _projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn DataSource>>> {
        Ok(None)
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
}
