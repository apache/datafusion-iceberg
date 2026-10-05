<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Release Process

This guide is for maintainers creating a release of Apache DataFusion Iceberg.

As part of the Apache governance model, official releases consist of signed source tarballs approved by the DataFusion PMC. The `iceberg-datafusion` crate is then published to crates.io from the approved source.

## Release Prerequisites

### Add git remote for `apache` repo

The instructions below assume the upstream git repo `git@github.com:apache/datafusion-iceberg.git` in remote `apache`.

```shell
git remote add apache git@github.com:apache/datafusion-iceberg.git
```

### Install tools

- `gpg` and `svn` to sign and upload release candidates.
- `java` and `python3` for the license check that `create-tarball.sh` runs.
- [`uv`](https://docs.astral.sh/uv/), or the `PyGithub` Python package, for the changelog script.

### Create GitHub Personal Access Token (PAT)

A personal access token (PAT) is needed for the changelog script. If you do not already have one, create a token with `repo` access by navigating to the [GitHub Developer Settings] page, and [follow these steps].

[github developer settings]: https://github.com/settings/developers
[follow these steps]: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

### Add GPG Public Key to SVN `KEYS` file

DataFusion subprojects share one `KEYS` file. If you will be releasing the final tarball, your GPG public key must be present in:

- https://dist.apache.org/repos/dist/dev/datafusion/KEYS
- https://dist.apache.org/repos/dist/release/datafusion/KEYS

See https://infra.apache.org/release-signing.html#generate for instructions on generating keys.

Committers can add signing keys using the Subversion client and their ASF account:

```shell
$ svn co https://dist.apache.org/repos/dist/dev/datafusion
$ cd datafusion
$ editor KEYS # add your key here
$ svn ci KEYS # commit changes
```

Follow the instructions in the header of the KEYS file to append your key. Here is an example:

```shell
(gpg --list-sigs "John Doe" && gpg --armor --export "John Doe") >> KEYS
svn commit KEYS -m "Add key for John Doe"
```

## Release Process: Step by Step

The examples below release version `0.11.0`.

### 1. Check that the crate can be published

crates.io rejects crates with git dependencies, so every dependency of `iceberg-datafusion` must come from crates.io. In particular, `iceberg` and `iceberg-catalog-rest` in the root `Cargo.toml` must point to a released iceberg-rust version, not a git revision. Check with:

```shell
cargo publish --dry-run -p iceberg-datafusion
```

### 2. Update the version

Update `version` under `[workspace.package]` in the root `Cargo.toml`. It must be higher than the latest version of [`iceberg-datafusion` on crates.io](https://crates.io/crates/iceberg-datafusion), which includes releases made from apache/iceberg-rust. Then update `Cargo.lock`:

```shell
cargo check --workspace
```

Commit the changes and create a PR against `main`.

### 3. Update the changelog

Each release has its own changelog file in `dev/changelog/`, such as `dev/changelog/0.11.0.md`, listing all changes since the previous release.

Generate it with `generate-changelog.py`. Pass the previous release tag, the commit to release, and the new version. For the first release from this repository, which has no earlier tag, pass the commit to start from instead of a tag. Run the script from the repository root with the `GITHUB_TOKEN` environment variable set; without a token, GitHub's API rate limit causes `403` errors.

```shell
export GITHUB_TOKEN=<your-token-here>
uv run dev/release/generate-changelog.py <previous-release-tag> apache/main 0.11.0 > dev/changelog/0.11.0.md
```

Without `uv`, install `PyGithub` and run the script with `python3` instead.

The script groups PRs using their labels and [Conventional Commits](https://www.conventionalcommits.org/) title prefixes (`feat:`, `fix:`, `docs:`, `perf:`, and `!` for breaking changes).

Commit the changelog and create a PR against `main`.

### 4. Create the release candidate

You must be a committer to run these steps because they upload to the Apache SVN distribution servers.

#### Pick a release candidate (RC) number

Pick numbers in sequential order, with `1` for `rc1`, `2` for `rc2`, etc.

#### Create a git tag for the release candidate

The official release artifacts are signed tarballs, but we also tag the commit they were created from. Release tags look like `0.11.0`, and release candidate tags look like `0.11.0-rc1`.

```shell
git fetch apache
git tag 0.11.0-rc1 apache/main
git push apache 0.11.0-rc1
```

#### Create, sign, and upload artifacts

```shell
./dev/release/create-tarball.sh 0.11.0 1
```

The script:

1. Checks that the tag exists and that its `Cargo.toml` has the release version.
2. Prints an email template for the vote.
3. Creates the source tarball from the tag with `git archive` and runs the RAT license check on it.
4. Signs the tarball, writes SHA-256 and SHA-512 checksums, and uploads everything to https://dist.apache.org/repos/dist/dev/datafusion/.

### 5. Vote on the release candidate

Send the email printed by `create-tarball.sh` to `dev@datafusion.apache.org`.

To become official, the release needs at least three +1 votes from PMC members and no -1 votes. The vote must stay open for at least 72 hours so everyone has a chance to review the release candidate.

#### Verifying release candidates

`verify-release-candidate.sh` downloads a release candidate and checks its signature and checksums. It then builds and tests it in a sandbox, using the toolchain pinned in `rust-toolchain.toml`, and runs `cargo publish --dry-run`:

```shell
./dev/release/verify-release-candidate.sh 0.11.0 1
```

To check license headers as well, run `./dev/release/run-rat.sh` on the downloaded tarball.

#### If changes are requested

Merge the fixes to `main`, update the changelog, and start again from step 4 with the next RC number.

#### If the vote passes: announce the result

Reply to the vote thread with the `[RESULT]` prefix added to the subject line. For example:

```
The vote has passed with <NUMBER> +1 votes. Thank you to all who helped
with the release verification.
```

### 6. Finalize the release

Only PMC members can do this step, after the release is approved.

Move the artifacts to the release location in SVN, e.g. https://dist.apache.org/repos/dist/release/datafusion/datafusion-iceberg-0.11.0/, using `release-tarball.sh`:

```shell
./dev/release/release-tarball.sh 0.11.0 1
```

Congratulations! The release is now official!

### 7. Create the release git tag

Tag the same commit as the approved release candidate:

```shell
git checkout 0.11.0-rc1
git tag 0.11.0
git push apache 0.11.0
```

### 8. Publish on crates.io

Only approved releases of the tarball may be published to crates.io, in order to conform to Apache Software Foundation governance standards.

Follow [these instructions](https://doc.rust-lang.org/cargo/reference/publishing.html) to create an account and log in to crates.io. Then ask an existing owner of `iceberg-datafusion` to add you; `cargo owner --list iceberg-datafusion` shows the current owners.

Download and unpack the official release tarball, check that its `Cargo.toml` has the release version, and publish from the unpacked directory:

```shell
cargo publish -p iceberg-datafusion
```

`iceberg-datafusion` is the only published crate. `iceberg-sqllogictest` and `iceberg-playground` set `publish = false`.

### 9. Add the release to Apache Reporter

Add the release to [Apache Reporter](https://reporter.apache.org/addrelease.html?datafusion), following the examples from previous releases. The reporter system should send you a reminder email. The release information is used to generate a template for the DataFusion board report.

### 10. Delete old RCs and releases

See the ASF documentation on [when to archive](https://www.apache.org/legal/release-policy.html#when-to-archive) for more information.

Release candidates should be deleted once the release is published.

To list DataFusion Iceberg release candidates:

```shell
svn ls https://dist.apache.org/repos/dist/dev/datafusion | grep datafusion-iceberg
```

To delete a release candidate:

```shell
svn delete -m "delete old DataFusion Iceberg RC" https://dist.apache.org/repos/dist/dev/datafusion/apache-datafusion-iceberg-0.11.0-rc1/
```

Only the latest release should be available from the `release` area. Delete old releases after publishing a new one.

To list DataFusion Iceberg releases:

```shell
svn ls https://dist.apache.org/repos/dist/release/datafusion | grep datafusion-iceberg
```

To delete a release:

```shell
svn delete -m "delete old DataFusion Iceberg release" https://dist.apache.org/repos/dist/release/datafusion/datafusion-iceberg-0.11.0
```

## Checking License Headers

CI runs the Apache Release Audit Tool (RAT) on every pull request. To run it locally, which requires `java` and `python3`:

```shell
./dev/release/run-rat.sh .
```

Files that cannot carry a license header are listed in `dev/release/rat_exclude_files.txt`.
