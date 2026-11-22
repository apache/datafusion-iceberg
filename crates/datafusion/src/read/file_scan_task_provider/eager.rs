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

use datafusion::arrow::datatypes::Schema;
use datafusion::common::stats::Precision;
use datafusion::common::{Result, Statistics, exec_err};
use datafusion::physical_plan::Partitioning;
use futures::StreamExt;
use iceberg::scan::FileScanTask;
use std::collections::BTreeSet;
use std::fmt::Formatter;
use std::sync::Arc;

use super::provider::{FileScanTaskProvider, FileScanTaskProviderContext};

/// Files discovered before execution, in groups that each become one output
/// partition. Group order and empty groups are retained.
/// Row counts sum the retained file record counts. The sum is exact when every
/// task reads a whole file with no row predicate and no deletes, and an upper
/// bound otherwise.
#[derive(Debug)]
pub struct EagerFileScanTaskProvider {
    file_scan_tasks: Vec<Vec<FileScanTask>>,
}

impl EagerFileScanTaskProvider {
    /// Reads `file_scan_tasks`, one output partition per group.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no groups. Empty groups are allowed.
    pub fn try_new(file_scan_tasks: Vec<Vec<FileScanTask>>) -> Result<Self> {
        if file_scan_tasks.is_empty() {
            return exec_err!(
                "An Iceberg task provider requires at least one file group"
            );
        }
        Ok(Self { file_scan_tasks })
    }

    /// All file tasks, in partition order and then group order.
    pub fn tasks(&self) -> impl Iterator<Item = &FileScanTask> {
        self.file_scan_tasks.iter().flatten()
    }

    /// The file tasks of each partition.
    pub fn groups(&self) -> &[Vec<FileScanTask>] {
        &self.file_scan_tasks
    }
}

impl FileScanTaskProvider for EagerFileScanTaskProvider {
    fn partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.groups().len())
    }

    fn open(
        self: Arc<Self>,
        partition: usize,
        _context: FileScanTaskProviderContext,
    ) -> Result<iceberg::scan::FileScanTaskStream> {
        let Some(count) = self.groups().get(partition).map(Vec::len) else {
            return exec_err!("Invalid Iceberg scan partition {partition}");
        };
        // Clone each task as it's read; holding `self` keeps the groups alive
        // after the data source is dropped.
        Ok(futures::stream::iter(0..count)
            .map(move |offset| Ok(self.groups()[partition][offset].clone()))
            .boxed())
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
        schema: &Schema,
    ) -> Result<Statistics> {
        if partition.is_some_and(|partition| partition >= self.groups().len()) {
            return exec_err!("Invalid Iceberg scan partition {partition:?}");
        }
        let groups = match partition {
            Some(partition) => &self.groups()[partition..=partition],
            None => self.groups(),
        };
        let rows = groups.iter().flatten().try_fold(0_usize, |rows, task| {
            rows.checked_add(usize::try_from(task.record_count()?).ok()?)
        });
        // Predicates and deletes may reduce a positive count, but cannot add rows.
        let exact = groups.iter().flatten().all(returns_every_record);
        let num_rows = match rows {
            Some(0) => Precision::Exact(0),
            Some(rows) if exact => Precision::Exact(rows),
            Some(rows) => Precision::Inexact(rows),
            None => Precision::Absent,
        };
        Ok(Statistics::new_unknown(schema).with_num_rows(num_rows))
    }

    fn fmt_as(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            ", provider=eager, planned_files={}",
            self.tasks().count()
        )?;
        // Assigned tasks may have different predicates. Report the distinct task
        // predicates rather than assuming a single filter applies to every file.
        let predicates = self
            .tasks()
            .filter_map(FileScanTask::predicate)
            .map(ToString::to_string)
            .collect::<BTreeSet<_>>();
        if !predicates.is_empty() {
            write!(
                f,
                ", predicates=[{}]",
                predicates.into_iter().collect::<Vec<_>>().join(", ")
            )?;
        }
        Ok(())
    }
}

/// Whether `task` reads a whole file with no row predicate and no deletes, so it
/// returns exactly its record count.
fn returns_every_record(task: &FileScanTask) -> bool {
    task.predicate().is_none()
        && task.deletes().is_empty()
        && task.start() == 0
        && task.length() == task.file_size_in_bytes()
}

#[cfg(test)]
mod tests {
    use iceberg::expr::BoundPredicate;
    use iceberg::scan::FileScanTaskDeleteFile;
    use iceberg::spec::{DataContentType, DataFileFormat, Schema as IcebergSchema};

    use super::*;

    #[test]
    fn a_file_with_zero_records_has_exactly_zero_rows() -> Result<()> {
        let provider =
            EagerFileScanTaskProvider::try_new(vec![vec![file_task(Some(0))]])?;
        let statistics = provider.partition_statistics(None, &Schema::empty())?;
        assert_eq!(statistics.num_rows, Precision::Exact(0));
        Ok(())
    }

    #[test]
    fn leaves_rows_unknown_when_a_task_has_no_record_count() -> Result<()> {
        let provider = EagerFileScanTaskProvider::try_new(vec![vec![
            file_task(Some(10)),
            file_task(None),
        ]])?;
        let statistics = provider.partition_statistics(None, &Schema::empty())?;
        assert_eq!(statistics.num_rows, Precision::Absent);
        Ok(())
    }

    #[test]
    fn whole_files_without_predicates_or_deletes_have_an_exact_row_count() -> Result<()> {
        let provider = EagerFileScanTaskProvider::try_new(vec![
            vec![file_task(Some(10)), file_task(Some(5))],
            vec![file_task(Some(7))],
        ])?;
        let schema = Schema::empty();
        assert_eq!(
            provider.partition_statistics(None, &schema)?.num_rows,
            Precision::Exact(22)
        );
        assert_eq!(
            provider.partition_statistics(Some(1), &schema)?.num_rows,
            Precision::Exact(7)
        );
        Ok(())
    }

    #[test]
    fn a_task_that_may_drop_rows_makes_only_its_partition_inexact() -> Result<()> {
        let with_predicate = task(Some(10), 2, Some(BoundPredicate::AlwaysTrue), vec![]);
        let with_delete = task(Some(10), 2, None, vec![position_delete_file()]);
        let half_the_file = task(Some(10), 1, None, vec![]);
        for inexact in [with_predicate, with_delete, half_the_file] {
            let provider = EagerFileScanTaskProvider::try_new(vec![
                vec![inexact],
                vec![file_task(Some(7))],
            ])?;
            let schema = Schema::empty();
            assert_eq!(
                provider.partition_statistics(None, &schema)?.num_rows,
                Precision::Inexact(17)
            );
            assert_eq!(
                provider.partition_statistics(Some(1), &schema)?.num_rows,
                Precision::Exact(7)
            );
        }
        Ok(())
    }

    // Only metadata is inspected; these files are never opened.
    fn file_task(record_count: Option<u64>) -> FileScanTask {
        task(record_count, 2, None, vec![])
    }

    /// A task for `length` bytes of a two-byte file.
    fn task(
        record_count: Option<u64>,
        length: u64,
        predicate: Option<BoundPredicate>,
        deletes: Vec<FileScanTaskDeleteFile>,
    ) -> FileScanTask {
        FileScanTask::builder()
            .with_file_size_in_bytes(2)
            .with_start(0)
            .with_length(length)
            .with_record_count(record_count)
            .with_data_file_path("data.parquet".to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(Arc::new(IcebergSchema::builder().build().unwrap()))
            .with_project_field_ids(vec![])
            .with_predicate(predicate)
            .with_deletes(deletes)
            .with_case_sensitive(true)
            .build()
            .unwrap()
    }

    fn position_delete_file() -> FileScanTaskDeleteFile {
        FileScanTaskDeleteFile::builder()
            .with_file_path("deletes.parquet".to_string())
            .with_file_size_in_bytes(1)
            .with_file_type(DataContentType::PositionDeletes)
            .with_file_format(DataFileFormat::Parquet)
            .with_partition_spec_id(0)
            .build()
            .unwrap()
    }
}
