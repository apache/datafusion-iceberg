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

//! Builders for the Iceberg tables used in tests.
//!
//! A [`FixtureCatalog`] is an in-memory catalog over a temporary directory.
//! Tables are created in it with a [`TableBuilder`], and each
//! [`FixtureTable::append`] commits one snapshot whose data files are exactly
//! the batches it was given.
//!
//! ```no_run
//! # async fn example() -> datafusion::error::Result<()> {
//! use iceberg::TableIdent;
//! use iceberg::spec::Transform;
//! use iceberg_datafusion_test_tools::{FixtureCatalog, taxi_schema, taxi_trips};
//!
//! let fixtures = FixtureCatalog::try_new().await?;
//! let ident = TableIdent::from_strs(["public", "trips"]).expect("valid identifier");
//! let mut table = fixtures
//!     .table(ident, taxi_schema())
//!     .partition_by("pickup_date", Transform::Identity)
//!     .create()
//!     .await?;
//! table.append().file(taxi_trips(0)?).commit().await?;
//! # Ok(())
//! # }
//! ```

mod append;
mod catalog;
mod error;
mod harness;
mod table;
mod taxi;

pub use append::Append;
pub use catalog::FixtureCatalog;
pub use harness::{
    IcebergTestHarness, IcebergTestHarnessBuilder, data_file_name, without_snapshots,
};
pub use table::{FixtureTable, TableBuilder};
pub use taxi::*;
