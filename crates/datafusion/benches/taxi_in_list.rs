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

//! End-to-end SQL lookup on a taxi table with a high-cardinality string key.
//! Set TAXI_DATA_DIR to use the seven daily NYC taxi Parquet files in place of
//! the built-in taxi fixture.

use std::env;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main};
use datafusion::arrow::array::{ArrayRef, RecordBatch, RecordBatchReader, StringArray};
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::util::display::array_value_to_string;
use datafusion::prelude::{SessionConfig, SessionContext};
use iceberg::TableIdent;
use iceberg::spec::{NestedField, PrimitiveType, Schema, Transform, Type};
use iceberg_datafusion::IcebergStaticTableProvider;
use iceberg_datafusion_test_tools::{
    FixtureCatalog, FixtureTable, TAXI_DAYS, TAXI_TRIPS_PER_DAY, taxi_schema, taxi_trips,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tokio::runtime::Builder;

fn resource_key(row: usize) -> String {
    let region = match row % 369 {
        0..14 => "ap-southeast-2",
        14..16 => "eu-central-1",
        16..18 => "eu-south-1",
        _ => "us-east-1",
    };
    // An odd multiplier scatters adjacent row IDs while keeping the keys unique.
    let volume = (row as u64).wrapping_mul(0x9e3779b97f4a7c15);
    format!("arn:aws:ec2:{region}:000000000000:volume/vol-{volume:017x}")
}

async fn taxi_table(data_dir: Option<&Path>) -> Result<FixtureTable, Box<dyn Error>> {
    let catalog = FixtureCatalog::try_new().await?;
    let mut fields = taxi_schema().as_struct().fields().to_vec();
    fields.push(
        NestedField::required(14, "resource_key", Type::Primitive(PrimitiveType::String))
            .into(),
    );
    let mut table = catalog
        .table(
            TableIdent::from_strs(["default", "taxi"])?,
            Schema::builder().with_fields(fields).build()?,
        )
        .partition_by("pickup_date", Transform::Identity)
        .create()
        .await?;
    let mut append = table.append().writer_properties(
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build(),
    );
    for day in 0..TAXI_DAYS {
        let trips = if let Some(dir) = data_dir {
            let path =
                dir.join(format!("pickup_date=2024-01-{:02}/data_0.parquet", 8 + day));
            let reader =
                ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?;
            concat_batches(&reader.schema(), &reader.collect::<Result<Vec<_>, _>>()?)?
        } else {
            taxi_trips(day)?
        };
        assert_eq!(trips.num_rows(), TAXI_TRIPS_PER_DAY);
        let first = day * TAXI_TRIPS_PER_DAY;
        let keys = StringArray::from_iter_values(
            (first..first + trips.num_rows()).map(resource_key),
        );
        let columns = trips
            .schema_ref()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .zip(trips.columns().iter().cloned())
            .chain([("resource_key", Arc::new(keys) as ArrayRef)]);
        append = append.file(RecordBatch::try_from_iter(columns)?);
    }
    append.commit().await?;
    Ok(table)
}

fn bench_taxi_in_list(c: &mut Criterion) {
    let runtime = Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let data_dir = env::var_os("TAXI_DATA_DIR");
    let table = runtime
        .block_on(taxi_table(data_dir.as_deref().map(Path::new)))
        .unwrap();
    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    let provider =
        IcebergStaticTableProvider::try_new_from_table(table.table().clone()).unwrap();
    ctx.register_table("taxi", Arc::new(provider)).unwrap();

    let mut group = c.benchmark_group("taxi_string_in_list");
    group.sample_size(20).sampling_mode(SamplingMode::Flat);
    let rows = TAXI_DAYS * TAXI_TRIPS_PER_DAY;
    for size in [1, 16, 31, 32, 100, 369, 400, 1000] {
        // This stride is coprime to the row count, so the keys span the daily files.
        let mut expected: Vec<_> = (0..size)
            .map(|i| resource_key((37_417 + i * 104_729) % rows))
            .collect();
        let literals = expected
            .iter()
            .map(|key| format!("'{key}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql =
            format!("SELECT resource_key FROM taxi WHERE resource_key IN ({literals})");
        let query = || async { ctx.sql(&sql).await.unwrap().collect().await.unwrap() };

        let mut actual = Vec::new();
        for batch in runtime.block_on(query()) {
            for row in 0..batch.num_rows() {
                actual.push(array_value_to_string(batch.column(0), row).unwrap());
            }
        }
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected);

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.to_async(&runtime).iter(query);
        });
    }
    group.finish();
}

criterion_group!(benches, bench_taxi_in_list);
criterion_main!(benches);
