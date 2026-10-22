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

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use iceberg::io::FileIO;
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_DELETE_FILE_PATH, RESERVED_COL_NAME_DELETE_FILE_POS,
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, ManifestListWriter,
    ManifestWriterBuilder, NestedField, Operation, PrimitiveType, Schema, Snapshot,
    SortOrder, Struct, Summary, TableMetadataBuilder, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::{Runtime, TableIdent};
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use tempfile::TempDir;

/// Two data files containing IDs [0, 1] and [2, 3], with ID 0 position-deleted.
/// Keeps the generated files alive while the table is scanned.
pub struct PositionDeleteFixture {
    pub table: Table,
    _directory: TempDir,
}

pub async fn table_with_position_delete() -> Result<PositionDeleteFixture, Box<dyn Error>>
{
    let directory = TempDir::new()?;
    let root = directory.path().to_str().unwrap();
    let io = FileIO::new_with_fs();
    let schema = Schema::builder()
        .with_fields([NestedField::required(
            1,
            "id",
            Type::Primitive(PrimitiveType::Long),
        )
        .into()])
        .build()?;
    let metadata = TableMetadataBuilder::new(
        schema,
        UnboundPartitionSpec::default(),
        SortOrder::unsorted_order(),
        root.to_string(),
        FormatVersion::V2,
        HashMap::new(),
    )?
    .build()?
    .metadata;
    let schema = metadata.current_schema().clone();
    let spec = metadata.default_partition_spec().as_ref().clone();
    let mut data_manifest = ManifestWriterBuilder::new(
        io.new_output(format!("{root}/data.avro"))?,
        Some(2),
        schema.clone(),
        spec.clone(),
    )
    .build_v2_data();
    let data_schema = Arc::new(ArrowSchema::new(vec![field("id", DataType::Int64, 1)]));
    for file in 0..2 {
        let path = format!("{root}/data-{file}.parquet");
        let batch = RecordBatch::try_new(
            data_schema.clone(),
            vec![Arc::new(Int64Array::from(vec![file * 2, file * 2 + 1]))],
        )?;
        let size = write_parquet(&path, batch)?;
        data_manifest.add_file(
            DataFileBuilder::default()
                .partition_spec_id(0)
                .content(DataContentType::Data)
                .file_path(path)
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(size)
                .record_count(2)
                .partition(Struct::empty())
                .build()?,
            1,
        )?;
    }
    let data_manifest = data_manifest.write_manifest_file().await?;

    let delete_path = format!("{root}/delete.parquet");
    let delete_schema = Arc::new(ArrowSchema::new(vec![
        field(
            RESERVED_COL_NAME_DELETE_FILE_PATH,
            DataType::Utf8,
            RESERVED_FIELD_ID_DELETE_FILE_PATH,
        ),
        field(
            RESERVED_COL_NAME_DELETE_FILE_POS,
            DataType::Int64,
            RESERVED_FIELD_ID_DELETE_FILE_POS,
        ),
    ]));
    let batch = RecordBatch::try_new(
        delete_schema,
        vec![
            Arc::new(StringArray::from(vec![format!("{root}/data-0.parquet")])),
            Arc::new(Int64Array::from(vec![0])),
        ],
    )?;
    let size = write_parquet(&delete_path, batch)?;
    let mut delete_manifest = ManifestWriterBuilder::new(
        io.new_output(format!("{root}/deletes.avro"))?,
        Some(2),
        schema.clone(),
        spec,
    )
    .build_v2_deletes();
    delete_manifest.add_file(
        DataFileBuilder::default()
            .partition_spec_id(0)
            .content(DataContentType::PositionDeletes)
            .file_path(delete_path)
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(size)
            .record_count(1)
            .partition(Struct::empty())
            .build()?,
        2,
    )?;
    let delete_manifest = delete_manifest.write_manifest_file().await?;
    let manifest_list_path = format!("{root}/snapshot.avro");
    let mut list = ManifestListWriter::v2(
        io.new_output(&manifest_list_path)?.writer().await?,
        2,
        None,
        2,
    );
    list.add_manifests([data_manifest, delete_manifest].into_iter())?;
    list.close().await?;
    let snapshot = Snapshot::builder()
        .with_snapshot_id(2)
        .with_sequence_number(2)
        .with_timestamp_ms(metadata.last_updated_ms() + 1)
        .with_manifest_list(manifest_list_path)
        .with_schema_id(schema.schema_id())
        .with_summary(Summary {
            operation: Operation::Overwrite,
            additional_properties: HashMap::new(),
        })
        .build();
    let metadata = metadata
        .into_builder(None)
        .set_branch_snapshot(snapshot, "main")?
        .build()?
        .metadata;
    let table = Table::builder()
        .file_io(io)
        .identifier(TableIdent::from_strs(["test", "deletes"])?)
        .metadata(metadata)
        .runtime(Runtime::try_current()?)
        .build()?;
    Ok(PositionDeleteFixture {
        table,
        _directory: directory,
    })
}

fn field(name: &str, data_type: DataType, id: i32) -> Field {
    Field::new(name, data_type, false).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

fn write_parquet(path: &str, batch: RecordBatch) -> Result<u64, Box<dyn Error>> {
    let mut writer = ArrowWriter::try_new(File::create(path)?, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(std::fs::metadata(path)?.len())
}
