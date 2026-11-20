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

"""Build the two Iceberg readers and compare them on the same taxi snapshot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib


def build(workspace, target, binary, *options):
    command = [
        "cargo", "build", "--release", "-p", "iceberg-datafusion",
        "--bench", "taxi_in_list", "--message-format=json-render-diagnostics", *options,
    ]
    result = subprocess.run(
        command, cwd=workspace, env={**os.environ, "CARGO_TARGET_DIR": str(target)},
        stdout=subprocess.PIPE, text=True, check=True,
    )
    artifacts = [json.loads(line) for line in result.stdout.splitlines()]
    executable, = [
        a["executable"] for a in artifacts
        if a.get("target", {}).get("name") == "taxi_in_list" and a.get("executable")
    ]
    shutil.copy2(executable, binary)


def packages(workspace):
    with (workspace / "Cargo.lock").open("rb") as lock:
        return {
            (p["name"], p["version"], p.get("source"))
            for p in tomllib.load(lock)["package"]
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("taxi_data", type=Path, help="directory with the seven pickup_date partitions")
    parser.add_argument("patched_iceberg", type=Path, help="Iceberg worktree with the hash-set change")
    parser.add_argument("output", type=Path, help="directory for samples and executed plans")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / "target"
    binaries = target / "taxi-in-list"
    binaries.mkdir(parents=True, exist_ok=True)
    before, after = binaries / "before", binaries / "after"
    patched = args.patched_iceberg.resolve()
    baseline_lock = packages(root)
    build(root, target, before, "--locked")

    # Keep the lockfile for a regular build pinned to the baseline.
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        for filename in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml"):
            shutil.copy2(root / filename, workspace / filename)
        shutil.copytree(root / "crates", workspace / "crates")
        overrides = ['[patch."https://github.com/DataDog/iceberg-rust"]']
        for name, subdir in [
            ("iceberg", "crates/iceberg"),
            ("iceberg-catalog-rest", "crates/catalog/rest"),
        ]:
            overrides.append(f"{name} = {{ path = {json.dumps(str(patched / subdir))} }}")
        config = binaries / "hash.toml"
        config.write_text("\n".join(overrides) + "\n")
        build(workspace, target, after, "--config", str(config))
        after_lock = packages(workspace)

    expected = {"iceberg", "iceberg-property-macro", "iceberg-catalog-rest"}
    changed = {name for name, _, _ in baseline_lock ^ after_lock}
    if changed != expected or any(s is not None for n, _, s in after_lock if n in expected):
        raise ValueError(f"Expected only local Iceberg source overrides, saw {changed}")
    if {(n, v) for n, v, _ in baseline_lock} != {(n, v) for n, v, _ in after_lock}:
        raise ValueError("The two builds resolved different package versions")
    for binary in (before, after):
        with binary.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        print(f"{binary.name} SHA-256 {digest}", flush=True)
    subprocess.run(
        [before, "compare", args.taxi_data.resolve(), before, after, args.output.resolve()],
        check=True,
    )


if __name__ == "__main__":
    main()
