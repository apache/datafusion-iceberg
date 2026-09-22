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

use datafusion::error::DataFusionError;
use iceberg::{Error, ErrorKind};

/// Converts a datafusion error into an iceberg error.
pub fn from_datafusion_error(error: DataFusionError) -> Error {
    let fallback_message = error.to_string();
    let DataFusionError::Context(context, error) = error else {
        return unexpected_datafusion_error(fallback_message);
    };

    let Some(kind) = parse_iceberg_error_kind(&context) else {
        return unexpected_datafusion_error(fallback_message);
    };
    let DataFusionError::Execution(message) = error.as_ref() else {
        return unexpected_datafusion_error(fallback_message);
    };

    Error::new(kind, strip_error_kind(kind, message))
}

/// Converts an iceberg error into a datafusion error.
pub fn to_datafusion_error(error: Error) -> DataFusionError {
    DataFusionError::Context(
        format!("IcebergError({})", error.kind()),
        Box::new(DataFusionError::Execution(error.to_string())),
    )
}

fn unexpected_datafusion_error(message: String) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        format!("DataFusion execution failed: {message}"),
    )
}

fn parse_iceberg_error_kind(context: &str) -> Option<ErrorKind> {
    let kind = context.strip_prefix("IcebergError(")?.strip_suffix(')')?;

    match kind {
        "PreconditionFailed" => Some(ErrorKind::PreconditionFailed),
        "Unexpected" => Some(ErrorKind::Unexpected),
        "DataInvalid" => Some(ErrorKind::DataInvalid),
        "NamespaceAlreadyExists" => Some(ErrorKind::NamespaceAlreadyExists),
        "TableAlreadyExists" => Some(ErrorKind::TableAlreadyExists),
        "NamespaceNotFound" => Some(ErrorKind::NamespaceNotFound),
        "TableNotFound" => Some(ErrorKind::TableNotFound),
        "FeatureUnsupported" => Some(ErrorKind::FeatureUnsupported),
        "CatalogCommitConflicts" => Some(ErrorKind::CatalogCommitConflicts),
        _ => None,
    }
}

fn strip_error_kind(kind: ErrorKind, message: &str) -> String {
    let kind = kind.into_static();
    if message == kind {
        String::new()
    } else {
        message
            .strip_prefix(&format!("{kind} => "))
            .unwrap_or(message)
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use datafusion::error::DataFusionError;
    use iceberg::{Error, ErrorKind};

    use super::{from_datafusion_error, to_datafusion_error};

    #[test]
    fn roundtrips_iceberg_error_kind_and_message() {
        let error = Error::new(ErrorKind::DataInvalid, "invalid manifest");
        let roundtripped = from_datafusion_error(to_datafusion_error(error));

        assert_eq!(roundtripped.kind(), ErrorKind::DataInvalid);
        assert_eq!(roundtripped.to_string(), "DataInvalid => invalid manifest");
    }

    #[test]
    fn encodes_iceberg_errors_with_native_datafusion_variants() {
        let error =
            to_datafusion_error(Error::new(ErrorKind::DataInvalid, "invalid manifest"));

        assert!(matches!(
            error,
            DataFusionError::Context(context, inner)
                if context == "IcebergError(DataInvalid)"
                    && matches!(inner.as_ref(), DataFusionError::Execution(message) if message == "DataInvalid => invalid manifest")
        ));
    }

    #[test]
    fn maps_non_iceberg_datafusion_errors_to_unexpected() {
        let error = from_datafusion_error(DataFusionError::Execution(
            "worker failed".to_string(),
        ));

        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert_eq!(
            error.to_string(),
            "Unexpected => DataFusion execution failed: Execution error: worker failed"
        );
    }
}
