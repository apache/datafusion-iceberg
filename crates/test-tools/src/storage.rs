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

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use iceberg::Result;
use iceberg::io::{
    FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorage, OutputFile, Storage,
    StorageConfig, StorageFactory,
};
use serde::{Deserialize, Serialize};

/// Local file storage that counts the byte ranges read from files, e.g. to
/// pass to `IcebergTableProviderFactory::new_with_storage_factory` in a test
/// that checks how much of a table a scan reads.
///
/// Iceberg reads metadata and manifest files whole, so only data file reads
/// go through [`Storage::reader`] and are counted. Clones share the count.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadCountingStorageFactory {
    #[serde(skip)]
    reads: Arc<AtomicUsize>,
}

impl ReadCountingStorageFactory {
    /// Returns how many byte ranges have been read from data files since the
    /// last call, and resets the count.
    pub fn take_reads(&self) -> usize {
        self.reads.swap(0, Ordering::Relaxed)
    }
}

#[typetag::serde]
impl StorageFactory for ReadCountingStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(ReadCountingStorage {
            reads: self.reads.clone(),
        }))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReadCountingStorage {
    #[serde(skip)]
    reads: Arc<AtomicUsize>,
}

#[async_trait]
#[typetag::serde]
impl Storage for ReadCountingStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        LocalFsStorage::new().exists(path).await
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        LocalFsStorage::new().metadata(path).await
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        LocalFsStorage::new().read(path).await
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        Ok(Box::new(CountingFileRead {
            inner: LocalFsStorage::new().reader(path).await?,
            reads: self.reads.clone(),
        }))
    }

    async fn write(&self, path: &str, bytes: Bytes) -> Result<()> {
        LocalFsStorage::new().write(path, bytes).await
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        LocalFsStorage::new().writer(path).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        LocalFsStorage::new().delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        LocalFsStorage::new().delete_prefix(path).await
    }

    async fn delete_stream(&self, paths: BoxStream<'static, String>) -> Result<()> {
        LocalFsStorage::new().delete_stream(paths).await
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

/// Counts the byte ranges read through `inner`.
struct CountingFileRead {
    inner: Box<dyn FileRead>,
    reads: Arc<AtomicUsize>,
}

#[async_trait]
impl FileRead for CountingFileRead {
    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.read(range).await
    }
}
