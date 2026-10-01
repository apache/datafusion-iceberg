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

# Runs the Apache Release Audit Tool (RAT) on a directory or source tarball
# and fails if any file lacks an approved license header. Files that cannot
# carry a header are listed in rat_exclude_files.txt.
#
# Requires curl, java and python3. The RAT jar and the reports (rat.txt,
# filtered_rat.txt) are written to the current directory.

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <directory or tarball>"
  exit 1
fi

RAT_VERSION=0.16.1
# Maven Central only publishes a SHA-1 for this jar; this SHA-512 was computed
# from a download that matched it.
RAT_SHA512=23047a236abcc182a0b3ecb42941f2c4d6307bf3b15c2b765863e33ec9040db6dd0466df0563d039622f503b87931948b5d6b8f729cbefb0f062c5bc67fcaa35
RAT_JAR=apache-rat-${RAT_VERSION}.jar

RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "${RAT_JAR}" ]; then
  curl -sSfL -o "${RAT_JAR}" \
    "https://repo1.maven.org/maven2/org/apache/rat/apache-rat/${RAT_VERSION}/${RAT_JAR}"
fi

if command -v sha512sum > /dev/null; then
  echo "${RAT_SHA512}  ${RAT_JAR}" | sha512sum -c --quiet -
else
  echo "${RAT_SHA512}  ${RAT_JAR}" | shasum -a 512 -c --quiet -
fi

java -jar "${RAT_JAR}" -x "$1" > rat.txt

if python3 "${RELEASE_DIR}/check-rat-report.py" \
  "${RELEASE_DIR}/rat_exclude_files.txt" rat.txt > filtered_rat.txt; then
  echo "No unapproved licenses"
else
  cat filtered_rat.txt
  echo "Unapproved licenses found. See the full report in rat.txt"
  exit 1
fi
