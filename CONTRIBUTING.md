# Contributing to MpAnafis

Contributions are welcome: bug reports, questions, small documentation fixes,
benchmarks, and code changes all help improve the project.

You can start with an issue or a draft pull request, including a reproduction,
a partial fix, or an idea you would like to discuss. I can help with
the internal invariants, test coverage, and checks needed to get a change ready.

## Getting started

The [repository guide](docs/organization.md) explains where the code lives.
[AGENTS.md](AGENTS.md) contains the detailed architecture and safety rules and
serves as a reference while working on a change.

Cargo declares Rust 1.91.0 as its minimum version,
and the Python source audits require Python 3.11 or later. The default library
build needs no external arithmetic library. Benchmark and tuning work can need
additional dependencies; the [benchmark guide](tools/benchmark/README.md) and
[CI setup](.github/workflows/ci.yml) describe those configurations.

## Sharing a change

A useful pull request explains the problem, what the change does, and how you
checked it. For a bug report, include the input, expected result, actual result,
and a small reproduction where possible. Mention the Rust version, enabled
features, and target when they affect the issue.

Keep patches focused so they are easy to review. A regression test is helpful
for a bug fix; property tests can extend coverage around arithmetic boundaries.
If a test or implementation is still incomplete, a draft is a good place to
share progress and ask for help.

Correctness comes first. Changes to unsafe code need a `// SAFETY:` comment
explaining why the operation is valid. The detailed representation, assignment,
lint, and module rules are in [AGENTS.md](AGENTS.md). Review can help bring a
contribution into alignment with those rules before merging.

Documentation describes current behavior. The design specifications also contain
explicitly marked planned requirements. When public behavior changes, update the
relevant API documentation and specification alongside it.

## Checking your work

For a library patch, start with:

```sh
cargo check --lib
cargo test --lib
```

Before a code change is ready to merge, formatting, lint, and policy checks also
need to pass:

```sh
cargo fmt --all --check
cargo clippy --lib
python3 tools/structure_audit.py
python3 tools/import_audit.py
```

Both policy audits require zero findings. Additional checks depend on the change:
feature and architecture changes need their relevant configurations tested;
Python tools have regression tests in their package's `tests/` directory.
[AGENTS.md](AGENTS.md) lists the verification commands by scope.

For documentation edits, check links and run `git diff --check`. Run
`cargo test --doc` when changing executable examples, including the README example.

Please say which checks you ran and flag anything you could not verify locally.
Maintainers can help with cross-target and Miri checks that need extra setup.

## Benchmarks and performance changes

For a performance change, include correctness checks and measurements of the
affected operations. Record the CPU, features, operand sizes, worker count,
and sampling settings so reviewers can understand the comparison.

The [benchmark driver](tools/benchmark/README.md) handles selected runs, reports,
and time-versus-size graphs. The [case guide](benches/public_api/README.md)
explains the shared benchmark templates and Rug/GMP comparisons, with FLINT used
where no equivalent is available. Full size sweeps are separate experiments;
selected cases are enough to check that a benchmark works.

Keep raw runs and working reports under ignored `target/bench-results/`.
Reviewed curves, a short description, and measurement details go under
`docs/int/benchmarks/`. Single-size comparisons stay in local tables. Keep the
complete run separately when sharing reproducible results.

## Continuous integration

The [CI guide](.github/workflows/README.md) describes formatting, lint and policy
checks, Python regressions, MSRV and platform compilation, native and selected
backend tests, cross-target checks, Miri, and short fuzz runs.
The [fuzz workflow](.github/workflows/fuzz.yml) runs both integer targets in a
matrix, with longer campaigns on its weekly schedule or through manual dispatch.
CI checks benchmark compilation and comparison registration; performance
measurements are run locally when relevant.
The [fuzz guide](fuzz/README.md) explains how to run the targets locally and which
API families their reference checks cover.

## Licensing and credit

Contributions must be compatible with the project's [Apache-2.0 license](LICENSE).
Please identify external implementations, papers, references, and machine-assisted
contributions used in a patch. Git history and `Co-authored-by` trailers preserve
authorship credit.
