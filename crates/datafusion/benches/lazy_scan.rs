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

//! Time a local Iceberg scan from execution to result, including lazy file discovery.
//! This is a warm, repeated scan of the same table, not a cold object-store benchmark.
//! Both queries project the same data column; only the stopping condition differs.
//! Run this same target on each revision, with a shared `CARGO_TARGET_DIR`:
//!
//! ```sh
//! cargo bench -p iceberg-datafusion --bench lazy_scan -- --save-baseline before
//! cargo bench -p iceberg-datafusion --bench lazy_scan -- --baseline before
//! ```

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use datafusion::arrow::array::RecordBatch;
use datafusion::error::Result;
use datafusion::physical_plan::collect;
use iceberg::TableIdent;
use iceberg_datafusion::IcebergTableProviderFactory;
use iceberg_datafusion_test_tools::{
    FixtureCatalog, FixtureTable, IcebergTestHarness, TAXI_TRIPS_PER_DAY, taxi_schema,
    taxi_trips,
};
use tokio::runtime::Builder;

const DAYS: usize = 160;
const ROWS_PER_FILE: usize = 1_000;

/// One append per day: 160 manifests, each listing 25 small Parquet files.
async fn many_file_taxi_table() -> Result<FixtureTable> {
    let catalog = FixtureCatalog::try_new().await?;
    let name = TableIdent::from_strs(["default", "taxi"]).unwrap();
    let mut table = catalog.table(name, taxi_schema()).create().await?;
    for day in 0..DAYS {
        let trips = taxi_trips(day)?;
        let mut append = table.append();
        for offset in (0..trips.num_rows()).step_by(ROWS_PER_FILE) {
            let len = ROWS_PER_FILE.min(trips.num_rows() - offset);
            append = append.file(trips.slice(offset, len));
        }
        append.commit().await?;
    }
    Ok(table)
}

fn lazy_scan(c: &mut Criterion) {
    let rt = Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let harness = rt.block_on(async {
        IcebergTestHarness::builder()
            .with_table(many_file_taxi_table().await.unwrap())
            .with_iceberg_table_factory(IcebergTableProviderFactory::new())
            .with_config_option("datafusion.execution.target_partitions", 4)
            .build()
            .await
            .unwrap()
    });
    let context = harness.state().task_ctx();
    let mut group = c.benchmark_group("lazy_scan");
    for (name, sql, rows) in [
        (
            "all_files",
            "SELECT fare_amount FROM taxi",
            DAYS * TAXI_TRIPS_PER_DAY,
        ),
        ("limit_one", "SELECT fare_amount FROM taxi LIMIT 1", 1),
    ] {
        let prepare = || rt.block_on(harness.physical_plan(sql)).unwrap();
        let execute = |plan| rt.block_on(collect(plan, context.clone())).unwrap();
        let output = execute(prepare());
        assert_eq!(
            output.iter().map(RecordBatch::num_rows).sum::<usize>(),
            rows
        );
        drop(output);

        group.bench_function(name, |b| {
            b.iter_batched(prepare, execute, BatchSize::PerIteration)
        });
    }
    group.finish();
}

criterion_group!(benches, lazy_scan);
criterion_main!(benches);
