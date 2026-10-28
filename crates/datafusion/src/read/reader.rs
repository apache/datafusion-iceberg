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

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::Result;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::limit::LimitStream;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::TryStreamExt;
use iceberg::arrow::ArrowReader;
use iceberg::scan::FileScanTaskStream;

use crate::to_datafusion_error;

/// One reader per source; clones share the Iceberg delete loader across partitions.
#[derive(Clone)]
pub(crate) struct ScanReader {
    reader: ArrowReader,
    pub(crate) metrics: ExecutionPlanMetricsSet,
}

impl ScanReader {
    pub(crate) fn new(reader: ArrowReader) -> Self {
        Self {
            reader,
            metrics: Default::default(),
        }
    }

    pub(crate) fn open(
        &self,
        tasks: FileScanTaskStream,
        schema: SchemaRef,
        fetch: Option<usize>,
        partition: usize,
    ) -> Result<SendableRecordBatchStream> {
        let stream = self
            .reader
            .clone()
            .read(tasks)
            .map_err(to_datafusion_error)?
            .stream()
            .map_err(to_datafusion_error);
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));
        Ok(Box::pin(LimitStream::new(
            stream,
            0,
            fetch,
            BaselineMetrics::new(&self.metrics, partition),
        )))
    }
}
