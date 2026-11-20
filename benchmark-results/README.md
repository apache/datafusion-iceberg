# Large string IN lists on the DataFusion Iceberg taxi scan

The hash-set reader used **90.3% less CPU** for a 400-key lookup across 175,000 trips. Median end-to-end wall time fell from **149.0 ms to 14.3 ms**. All queries in both builds returned exactly the expected keys.

| Distinct keys | Baseline CPU (ms) | Hash reader CPU (ms) | Baseline wall (ms) | Hash reader wall (ms) | Wall speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.32 | 5.49 | 5.22 | 5.36 | 0.97× |
| 16 | 11.20 | 10.93 | 11.06 | 10.81 | 1.02× |
| 31 | 16.62 | 16.31 | 16.49 | 16.19 | 1.02× |
| 32 | 17.05 | 10.76 | 16.89 | 10.64 | 1.59× |
| 100 | 40.90 | 13.73 | 40.76 | 13.58 | 3.00× |
| 369 | 139.83 | 13.00 | 139.67 | 12.87 | 10.85× |
| 400 | 149.09 | 14.40 | 148.97 | 14.27 | 10.44× |
| 1,000 | 375.52 | 13.77 | 375.42 | 13.62 | 27.55× |

The hash path starts at 32 distinct literals. At 400 keys, median planning CPU was 1.48 ms before and 1.31 ms after. Median execution CPU fell from 147.61 ms to 13.10 ms. The original reader evaluates a vectorized equality for each literal in each candidate Arrow batch. Those comparisons reuse the decoded column in memory.

Both executed plans projected only `dd_resource_key` and pushed the same 400 keys into the lazy Iceberg scan. Iceberg emitted 400 rows in seven batches in each build. DataFusion then applied its own `IN (SET)` filter, which passed 400 of 400 rows. [Baseline plan](before-400-plan.txt) · [Patched plan](after-400-plan.txt)

## Dataset and timing

- The input was the seven Git LFS files from the connector’s January 8–14, 2024 NYC yellow taxi fixture at `82cea1462bff959021b881311c9ef9f4f22328af` (25,000 trips per day). The harness retained the trip columns and wrote one additional UTF-8 key per trip into a date-partitioned Iceberg snapshot with seven Zstandard-compressed Parquet files. Both binaries read exactly the same snapshot.
- The supplied query projects and filters the same string column. Its truncated text contains 369 complete distinct keys. The benchmark generated 63–68 byte strings with common prefixes and a unique volume ID, then queried for increasing prefixes of a deterministic sample spread across the files. Every requested key matches exactly one trip. At 400 keys this selects about 0.23% of the table.
- One query ran at a time on an Apple M5 Pro (arm64, macOS), with four Tokio workers, four DataFusion target partitions and the default lazy scan. Each cell reports the median of ten release-build queries. The run order was baseline, patched, patched, baseline. Each block warmed each list size once, then recorded five queries. The second block for each binary reversed the list-size order.
- Wall and process CPU measurements cover SQL parsing, optimization, physical planning, execution and collection. CPU includes user and system time across threads. Fixture creation, table registration, result comparison and plan formatting were outside the timer. The harness compared the full result multiset against the expected keys after each query. [All 160 measurements](samples.csv)
- This is a warm, local Parquet scan with literal SQL predicates. Production concurrency, remote storage, dynamic join filters and load shedding were outside the measurement.

## Reproduce

The connector baseline is `865c0813864fdf2d416eda4bd77d09ea1918f374` with DataFusion 55.2.0, Arrow 59.3.0 and Parquet 59.3.2. The reader baseline is DataDog's `9354b42ad8c28b9f3df2ae4c3e4931beab0be12b`. The patched reader is `4220eb57769582471838905767eab786cf591eb2`, a direct cherry-pick of `a16ffe709c258595895f5d3c9f00d34b3abedc9f` onto that baseline.

The DataDog pin renders a binary literal as `Binary` in logical EXPLAIN output. The local benchmark branch updates the two corresponding snapshot lines. The physical plan and binary query result assertions are identical. The baseline passed workspace tests and Clippy. All 41 connector integration tests also passed with the patched reader.

From this connector worktree, with the taxi LFS files materialized, run the comparison below.

```sh
python3 dev/bench_taxi_in_list.py \
  ../main/crates/datafusion/testdata/iceberg/taxi/data \
  ../../iceberg-rust/arrow59-taxi-in-list \
  target/taxi-results
```

The runner builds the baseline from the lockfile and applies a Cargo override for the patched reader in a temporary copy of the same connector source. It checks that the versions of every resolved package match and only the three Iceberg package sources differ. It leaves the two binaries and the optional Cargo override in `target/taxi-in-list/`. [Runner](../dev/bench_taxi_in_list.py) · [Benchmark](../crates/datafusion/benches/taxi_in_list.rs)

| Binary | SHA-256 for this run |
| --- | --- |
| Baseline | `f98ef250717f7035118db9066d919409c1d8938efbfb341122eb93088a019dd5` |
| Patched | `fa3360add9c471e720800312a458523ef7a08b9642531015f16c52ea2a57a590` |
