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

//! Compare two release builds on one immutable, taxi-derived Iceberg snapshot.

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use datafusion::arrow::array::{ArrayRef, AsArray, RecordBatch, StringArray};
use datafusion::arrow::compute::{cast, concat_batches};
use datafusion::arrow::datatypes::DataType;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::{collect, display::DisplayableExecutionPlan};
use datafusion::prelude::{SessionConfig, SessionContext};
use iceberg::TableIdent;
use iceberg::spec::{NestedField, PrimitiveType, Schema, Transform, Type};
use iceberg_datafusion::IcebergTableProviderFactory;
use iceberg_datafusion_test_tools::{FixtureCatalog, FixtureTable, taxi_schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rustix::time::{ClockId, clock_gettime};
use tempfile::tempdir;

const KEY: &str = "dd_resource_key";
const SIZES: [usize; 8] = [1, 16, 31, 32, 100, 369, 400, 1000];
const RUNS_PER_BLOCK: usize = 5;

fn resource_key(row: usize) -> String {
    // Mirror the lengths and predominantly shared prefix in the reported query.
    let region = match row % 369 {
        0..14 => "ap-southeast-2",
        14..16 => "eu-central-1",
        16..18 => "eu-south-1",
        _ => "us-east-1",
    };
    // An odd multiplier permutes the row IDs uniquely over u64.
    let volume = (row as u64).wrapping_mul(0x9e3779b97f4a7c15);
    format!("arn:aws:ec2:{region}:000000000000:volume/vol-{volume:017x}")
}

async fn make_fixture(data_dir: &Path) -> Result<(FixtureTable, usize)> {
    let fixtures = FixtureCatalog::try_new().await?;
    let mut fields = taxi_schema().as_struct().fields().to_vec();
    fields.push(
        NestedField::required(14, KEY, Type::Primitive(PrimitiveType::String)).into(),
    );
    let schema = Schema::builder().with_fields(fields).build()?;
    let mut table = fixtures
        .table(TableIdent::from_strs(["bench", "taxi_keys"])?, schema)
        .partition_by("pickup_date", Transform::Identity)
        .create()
        .await?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_row_count(Some(25_000))
        .build();
    let mut append = table.append().writer_properties(props);
    let mut rows = 0;
    for day in 8..=14 {
        let path = data_dir.join(format!("pickup_date=2024-01-{day:02}/data_0.parquet"));
        let reader = ParquetRecordBatchReaderBuilder::try_new(
            File::open(&path).with_context(|| format!("opening {}", path.display()))?,
        )?
        .build()?;
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        let source = concat_batches(
            &batches.first().context("empty taxi file")?.schema(),
            &batches,
        )?;
        ensure!(
            source.num_rows() == 25_000,
            "unexpected row count in {}",
            path.display()
        );
        let keys = StringArray::from_iter_values(
            (rows..rows + source.num_rows()).map(resource_key),
        );
        rows += source.num_rows();
        let original = source.schema();
        let mut columns = original
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .zip(source.columns().iter().cloned())
            .collect::<Vec<_>>();
        if source.column_by_name("pickup_date").is_none() {
            columns.push((
                "pickup_date",
                cast(
                    source
                        .column_by_name("pickup_at")
                        .context("missing pickup_at")?,
                    &DataType::Date32,
                )?,
            ));
        }
        columns.push((KEY, Arc::new(keys) as ArrayRef));
        append = append.file(RecordBatch::try_from_iter(columns)?);
    }
    ensure!(
        append.commit().await?.len() == 7,
        "expected seven daily files"
    );
    Ok((table, rows))
}

async fn compare(taxi: &str, before: &str, after: &str, out: &str) -> Result<()> {
    let (table, rows) = make_fixture(Path::new(taxi)).await?;
    let dir = tempdir()?;
    let key_path = dir.path().join("keys.txt");
    // A coprime stride samples unique trips across all seven files. Smaller lists
    // are prefixes of the same pool, so a list of length N must yield exactly N keys.
    let pool = (0..*SIZES.last().unwrap())
        .map(|n| resource_key((37_417 + n * 104_729) % rows))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&key_path, pool)?;
    fs::create_dir_all(out)?;
    let mut csv = BufWriter::new(File::create(Path::new(out).join("samples.csv"))?);
    writeln!(
        csv,
        "variant,block,sample,keys,rows,wall_ms,cpu_ms,plan_wall_ms,plan_cpu_ms"
    )?;
    for (variant, binary, block) in [
        ("before", before, "1"),
        ("after", after, "1"),
        ("after", after, "2"),
        ("before", before, "2"),
    ] {
        eprintln!("Benchmarking {variant}, block {block}");
        let output = Command::new(binary)
            .arg("measure")
            .arg(table.metadata_location()?)
            .arg(&key_path)
            .args([out, variant, block])
            .output()?;
        ensure!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        csv.write_all(&output.stdout)?;
        csv.flush()?;
    }
    Ok(())
}

fn cpu() -> Duration {
    Duration::try_from(clock_gettime(ClockId::ProcessCPUTime))
        .expect("positive process CPU time")
}

fn ms(time: Duration) -> f64 {
    time.as_secs_f64() * 1000.0
}

async fn measure(
    metadata: &str,
    keys: &str,
    out: &str,
    variant: &str,
    block: &str,
) -> Result<()> {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .with_table_factory(
            "ICEBERG".to_owned(),
            Arc::new(IcebergTableProviderFactory::new()),
        )
        .build();
    let ctx = SessionContext::new_with_state(state);
    let metadata = metadata.replace('\'', "''");
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE taxi_keys STORED AS ICEBERG LOCATION '{metadata}'"
    ))
    .await?
    .collect()
    .await?;
    let pool = fs::read_to_string(keys)?;
    let pool = pool.lines().collect::<Vec<_>>();
    let mut sizes = SIZES;
    if block == "2" {
        sizes.reverse();
    }
    for size in sizes {
        let input = &pool[..size];
        let literals = input
            .iter()
            .map(|key| format!("'{key}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!("SELECT {KEY} FROM taxi_keys WHERE {KEY} IN ({literals})");
        let mut expected = input.to_vec();
        expected.sort_unstable();
        for run in 0..=RUNS_PER_BLOCK {
            let start = Instant::now();
            let start_cpu = cpu();
            let plan = ctx.sql(&sql).await?.create_physical_plan().await?;
            let plan_wall = start.elapsed();
            let plan_cpu = cpu() - start_cpu;
            let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
            let elapsed = start.elapsed();
            let cpu_time = cpu() - start_cpu;

            let strings = batches
                .iter()
                .map(|batch| cast(batch.column(0), &DataType::Utf8))
                .collect::<Result<Vec<_>, _>>()?;
            let mut actual = strings
                .iter()
                .flat_map(|a| a.as_string::<i32>().iter())
                .collect::<Option<Vec<_>>>()
                .context("unexpected null key")?;
            actual.sort_unstable();
            ensure!(
                actual == expected,
                "wrong result for {size} keys on {variant}, run {run}"
            );
            if run == 0 {
                if size == 400 && block == "1" {
                    let explain = DisplayableExecutionPlan::with_metrics(plan.as_ref())
                        .indent(false)
                        .to_string();
                    ensure!(
                        explain.contains("FilterExec")
                            && explain.contains("provider=lazy")
                            && explain.contains("predicate=dd_resource_key IN"),
                        "unexpected 400-key plan: {explain:.500}"
                    );
                    fs::write(
                        Path::new(out).join(format!("{variant}-400-plan.txt")),
                        explain,
                    )?;
                }
            } else {
                println!(
                    "{variant},{block},{run},{size},{},{:.3},{:.3},{:.3},{:.3}",
                    actual.len(),
                    ms(elapsed),
                    ms(cpu_time),
                    ms(plan_wall),
                    ms(plan_cpu)
                );
            }
        }
    }
    Ok(())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, taxi, before, after, out] if cmd == "compare" => {
            compare(taxi, before, after, out).await
        }
        [cmd, metadata, keys, out, variant, block] if cmd == "measure" => {
            measure(metadata, keys, out, variant, block).await
        }
        _ => bail!(
            "usage: taxi_in_list compare TAXI_DATA_DIR BASE_BINARY PATCHED_BINARY OUTPUT_DIR"
        ),
    }
}
