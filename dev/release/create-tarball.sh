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

# Adapted from https://github.com/apache/datafusion/tree/main/dev/release/create-tarball.sh

# This script creates a signed tarball in
# dev/dist/apache-datafusion-iceberg-<version>-rc<rc>/apache-datafusion-iceberg-<version>.tar.gz,
# uploads it to the "dev" area of the dist.apache.org datafusion repository and
# prints an email for the dev@datafusion.apache.org list to start the vote.
#
# See dev/release/README.md for full release instructions.
#
# Requirements:
#
# 1. gpg set up for signing, with your public key in the DataFusion KEYS file
# 2. svn, logged into the Apache SVN server with your ASF credentials
# 3. java and python3 (for the RAT license check)

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_TOP_DIR="$(cd "${SOURCE_DIR}/../../" && pwd)"

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <version> <rc>"
  echo "ex. $0 0.11.0 1"
  exit 1
fi

version=$1
rc=$2
tag="${version}-rc${rc}"

release=apache-datafusion-iceberg-${version}
distdir=${SOURCE_TOP_DIR}/dev/dist/${release}-rc${rc}
tarname=${release}.tar.gz
tarball=${distdir}/${tarname}
url="https://dist.apache.org/repos/dist/dev/datafusion/${release}-rc${rc}"

echo "Attempting to create ${tarball} from tag ${tag}"
if ! release_hash=$(cd "${SOURCE_TOP_DIR}" && git rev-list --max-count=1 "${tag}" 2> /dev/null); then
  echo "Cannot continue: unknown git tag: ${tag}"
  exit 1
fi

if ! (cd "${SOURCE_TOP_DIR}" && git show "${release_hash}:Cargo.toml") \
  | grep "^version = \"${version}\"$" > /dev/null; then
  echo "Cannot continue: Cargo.toml at ${tag} does not set version = \"${version}\""
  exit 1
fi

echo "Draft email for dev@datafusion.apache.org mailing list"
echo ""
echo "---------------------------------------------------------"
cat <<MAIL
To: dev@datafusion.apache.org
Subject: [VOTE] Release Apache DataFusion Iceberg ${version} RC${rc}
Hi,

I would like to propose a release of Apache DataFusion Iceberg version ${version}.

This release candidate is based on commit: ${release_hash} [1]
The proposed release tarball and signatures are hosted at [2].
The changelog is located at [3].

Please download, verify checksums and signatures, run the unit tests, and vote
on the release. The vote will be open for at least 72 hours.

Only votes from PMC members are binding, but all members of the community are
encouraged to test the release and vote with "(non-binding)".

The standard verification procedure is documented at https://github.com/apache/datafusion-iceberg/blob/main/dev/release/README.md#verifying-release-candidates.

[ ] +1 Release this as Apache DataFusion Iceberg ${version}
[ ] +0
[ ] -1 Do not release this as Apache DataFusion Iceberg ${version} because...

Here is my vote:

+1

[1]: https://github.com/apache/datafusion-iceberg/tree/${release_hash}
[2]: ${url}
[3]: https://github.com/apache/datafusion-iceberg/blob/${release_hash}/dev/changelog/${version}.md
MAIL
echo "---------------------------------------------------------"

# create <tarball> containing the files in git at $release_hash
# the files in the tarball are prefixed with ${release} (e.g. apache-datafusion-iceberg-0.11.0)
mkdir -p "${distdir}"
(cd "${SOURCE_TOP_DIR}" && git archive "${release_hash}" --prefix "${release}/" | gzip > "${tarball}")

echo "Running rat license checker on ${tarball}"
"${SOURCE_DIR}/run-rat.sh" "${tarball}"

echo "Signing tarball and creating checksums"
gpg --armor --output "${tarball}.asc" --detach-sig "${tarball}"
# create signing with relative path of tarball
# so that they can be verified with a command such as
#  shasum --check apache-datafusion-iceberg-0.11.0.tar.gz.sha512
(cd "${distdir}" && shasum -a 256 "${tarname}") > "${tarball}.sha256"
(cd "${distdir}" && shasum -a 512 "${tarname}") > "${tarball}.sha512"

echo "Uploading to datafusion dist/dev to ${url}"
svn co --depth=empty https://dist.apache.org/repos/dist/dev/datafusion "${SOURCE_TOP_DIR}/dev/dist"
svn add "${distdir}"
svn ci -m "Apache DataFusion Iceberg ${version} ${rc}" "${distdir}"
