#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# Adapted from https://github.com/apache/datafusion/tree/main/dev/release/verify-release-candidate.sh

# Downloads a release candidate, verifies its signature and checksums, then
# builds and tests it in a sandbox with the toolchain pinned in
# rust-toolchain.toml. Set VERIFY_TMPDIR to keep the sandbox afterwards.

# Check that required dependencies are installed
check_dependencies() {
  local missing_deps=0
  local required_deps=("curl" "git" "gpg" "cc")

  for dep in "${required_deps[@]}"; do
    if ! command -v "$dep" &> /dev/null; then
      echo "Error: $dep is not installed or not in PATH"
      missing_deps=1
    fi
  done

  # Either shasum or sha256sum/sha512sum are required
  if ! command -v shasum &> /dev/null \
    && ! { command -v sha256sum &> /dev/null && command -v sha512sum &> /dev/null; }; then
    echo "Error: Neither shasum nor sha256sum/sha512sum are installed or in PATH"
    missing_deps=1
  fi

  if [ $missing_deps -ne 0 ]; then
    echo "Please install missing dependencies and try again"
    exit 1
  fi
}

case $# in
  2) VERSION="$1"
     RC_NUMBER="$2"
     ;;
  *) echo "Usage: $0 X.Y.Z RC_NUMBER"
     exit 1
     ;;
esac

set -e
set -x
set -o pipefail

check_dependencies

DIST_URL='https://dist.apache.org/repos/dist/dev/datafusion'

download_dist_file() {
  curl \
    --silent \
    --show-error \
    --fail \
    --location \
    --remote-name "${DIST_URL}/$1"
}

download_rc_file() {
  download_dist_file "apache-datafusion-iceberg-${VERSION}-rc${RC_NUMBER}/$1"
}

import_gpg_keys() {
  download_dist_file KEYS
  gpg --import KEYS
}

if type shasum >/dev/null 2>&1; then
  sha256_verify="shasum -a 256 -c"
  sha512_verify="shasum -a 512 -c"
else
  sha256_verify="sha256sum -c"
  sha512_verify="sha512sum -c"
fi

fetch_archive() {
  local dist_name=$1
  download_rc_file "${dist_name}.tar.gz"
  download_rc_file "${dist_name}.tar.gz.asc"
  download_rc_file "${dist_name}.tar.gz.sha256"
  download_rc_file "${dist_name}.tar.gz.sha512"
  verify_dir_artifact_signatures
}

verify_dir_artifact_signatures() {
  # verify the signature and the checksums of each artifact
  find . -name '*.asc' | while read -r sigfile; do
    artifact=${sigfile/.asc/}
    gpg --verify "$sigfile" "$artifact" || exit 1

    # go into the directory because the checksum files contain only the
    # basename of the artifact
    pushd "$(dirname "$artifact")"
    base_artifact=$(basename "$artifact")
    ${sha256_verify} "$base_artifact.sha256" || exit 1
    ${sha512_verify} "$base_artifact.sha512" || exit 1
    popd
  done
}

setup_tempdir() {
  # shellcheck disable=SC2329 # invoked by the EXIT trap below
  cleanup() {
    if [ "${TEST_SUCCESS}" = "yes" ]; then
      rm -fr "${VERIFY_TMPDIR}"
    else
      echo "Failed to verify release candidate. See ${VERIFY_TMPDIR} for details."
    fi
  }

  if [ -z "${VERIFY_TMPDIR}" ]; then
    # clean up automatically if VERIFY_TMPDIR is not defined
    VERIFY_TMPDIR=$(mktemp -d -t "$1.XXXXX")
    trap cleanup EXIT
  else
    # don't clean up automatically
    mkdir -p "${VERIFY_TMPDIR}"
  fi
}

test_source_distribution() {
  # install a sandboxed rust toolchain that doesn't touch the user's own
  export RUSTUP_HOME=$PWD/test-rustup
  export CARGO_HOME=$PWD/test-rustup

  curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path

  export PATH=$RUSTUP_HOME/bin:$PATH
  # shellcheck disable=SC1091
  source "$RUSTUP_HOME/env"

  # install the version of rust pinned in rust-toolchain.toml
  rustup toolchain install

  # raises on any formatting errors
  cargo fmt --all -- --check

  cargo test --workspace --locked

  # the published crate must build from its packaged contents, and every
  # dependency must come from crates.io
  cargo publish --dry-run --locked -p iceberg-datafusion
}

TEST_SUCCESS=no

setup_tempdir "datafusion-iceberg-${VERSION}"
echo "Working in sandbox ${VERIFY_TMPDIR}"
cd "${VERIFY_TMPDIR}"

dist_name="apache-datafusion-iceberg-${VERSION}"
import_gpg_keys
fetch_archive "${dist_name}"
tar xf "${dist_name}.tar.gz"
pushd "${dist_name}"
    test_source_distribution
popd

TEST_SUCCESS=yes
echo 'Release candidate looks good!'
exit 0
