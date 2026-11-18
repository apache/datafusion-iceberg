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

//! Each query looks its tables up in the catalog afresh, so it sees changes
//! other clients made to a table's schema after the catalog was registered.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::error::Error;
    use std::sync::Arc;

    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::prelude::SessionContext;
    use iceberg::spec::{PrimitiveType, Type};
    use iceberg::transaction::{
        AddColumn, ApplyTransactionAction, Transaction, UpdateSchemaAction,
    };
    use iceberg::{NamespaceIdent, TableIdent};
    use iceberg_datafusion::IcebergCatalogProvider;
    use iceberg_datafusion_test_tools::FixtureCatalog;

    type Result<T> = std::result::Result<T, Box<dyn Error>>;

    /// Creates a catalog with the empty namespace `ns`. Returns a session with
    /// the catalog registered as `c`, and another client of the same catalog.
    async fn setup() -> Result<(Session, OtherClient)> {
        let fixtures = FixtureCatalog::try_new().await?;
        fixtures
            .catalog()
            .create_namespace(&NamespaceIdent::new("ns".to_string()), HashMap::new())
            .await?;
        Ok((Session::new(&fixtures).await?, OtherClient { fixtures }))
    }

    /// A DataFusion session with the catalog registered as `c`.
    struct Session {
        ctx: SessionContext,
    }

    impl Session {
        async fn new(fixtures: &FixtureCatalog) -> Result<Self> {
            let ctx = SessionContext::new();
            let provider = IcebergCatalogProvider::try_new(fixtures.catalog()).await?;
            ctx.register_catalog("c", Arc::new(provider));
            Ok(Self { ctx })
        }

        async fn sql(&self, sql: &str) -> Result<String> {
            let batches = self.ctx.sql(sql).await?.collect().await?;
            Ok(pretty_format_batches(&batches)?.to_string())
        }
    }

    /// Another client of the catalog, changing table `ns.t` as another engine
    /// would, without the session knowing.
    struct OtherClient {
        fixtures: FixtureCatalog,
    }

    impl OtherClient {
        async fn update_schema(
            &self,
            change: impl FnOnce(UpdateSchemaAction) -> UpdateSchemaAction,
        ) -> Result<()> {
            let ident = TableIdent::from_strs(["ns", "t"])?;
            let catalog = self.fixtures.catalog();
            let table = catalog.load_table(&ident).await?;
            let tx = Transaction::new(&table);
            change(tx.update_schema())
                .apply(tx)?
                .commit(catalog.as_ref())
                .await?;
            Ok(())
        }

        /// Runs `sql` in a session of its own, started now.
        async fn sql(&self, sql: &str) -> Result<String> {
            Session::new(&self.fixtures).await?.sql(sql).await
        }
    }

    #[tokio::test]
    async fn inserts_into_a_column_another_client_added() -> Result<()> {
        let (session, other_client) = setup().await?;
        session
            .sql("CREATE TABLE c.ns.t (id INT NOT NULL, name VARCHAR NOT NULL)")
            .await?;
        session.sql("INSERT INTO c.ns.t VALUES (1, 'a')").await?;

        other_client
            .update_schema(|schema| {
                schema.add_column(AddColumn::optional(
                    "email",
                    Type::Primitive(PrimitiveType::String),
                ))
            })
            .await?;

        session
            .sql("INSERT INTO c.ns.t (id, name) VALUES (2, 'b')")
            .await?;
        session
            .sql("INSERT INTO c.ns.t VALUES (3, 'c', 'c@example.com')")
            .await?;
        let rows = session.sql("SELECT * FROM c.ns.t ORDER BY id").await?;
        insta::assert_snapshot!(rows, @r"
    +----+------+---------------+
    | id | name | email         |
    +----+------+---------------+
    | 1  | a    |               |
    | 2  | b    |               |
    | 3  | c    | c@example.com |
    +----+------+---------------+
    ");

        Ok(())
    }

    /// Dropping a column and adding it back is how a column's type is changed
    /// when the change isn't an allowed promotion.
    #[tokio::test]
    async fn reads_a_column_another_client_dropped_and_added_back() -> Result<()> {
        let (session, other_client) = setup().await?;
        session
            .sql("CREATE TABLE c.ns.t (id INT NOT NULL, price DOUBLE NOT NULL)")
            .await?;
        session.sql("INSERT INTO c.ns.t VALUES (1, 1.5)").await?;

        other_client
            .update_schema(|schema| schema.delete_column("price"))
            .await?;
        other_client
            .update_schema(|schema| {
                schema.add_column(AddColumn::optional(
                    "price",
                    Type::Primitive(PrimitiveType::Decimal {
                        precision: 10,
                        scale: 2,
                    }),
                ))
            })
            .await?;
        other_client
            .sql("INSERT INTO c.ns.t VALUES (2, 2.25)")
            .await?;

        let rows = session
            .sql("SELECT *, price * 2 AS doubled FROM c.ns.t ORDER BY id")
            .await?;
        insta::assert_snapshot!(rows, @r"
    +----+-------+---------+
    | id | price | doubled |
    +----+-------+---------+
    | 1  |       |         |
    | 2  | 2.25  | 4.50    |
    +----+-------+---------+
    ");
        let nulls = session
            .sql("SELECT count(*) AS nulls FROM c.ns.t WHERE price IS NULL")
            .await?;
        insta::assert_snapshot!(nulls, @r"
    +-------+
    | nulls |
    +-------+
    | 1     |
    +-------+
    ");

        Ok(())
    }
}
