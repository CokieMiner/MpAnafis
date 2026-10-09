# Continuous integration

[ci.yml](ci.yml) runs on pushes and pull requests to `main` and `master`, and on
manual dispatch. It validates the crate, tools, benchmarks, and fuzz harnesses.

| Job | Scope |
| --- | --- |
| `quality` | Rust formatting, import and structure audits, nightly Clippy |
| `tools` | Workflow validation and Python tool regressions |
| `msrv` | Library compilation with Rust 1.93.0 |
| `host_check` | Library compilation on Windows and macOS |
| `test` | Native tests, doctests, fuzz harness checks, and benchmark compilation |
| `fuzz_smoke` | Both fuzz targets for 60 seconds each |
| `test_x86_backends` | Tests with each available x86 backend |
| `test_power9` | POWER9 compilation and QEMU execution |
| `check_archs` | Target and feature matrix in [tools/check_all_archs.sh](../../tools/check_all_archs.sh) |
| `test_cross` | Tests on twelve Linux targets through `cross` or QEMU |
| `miri` | Interpreter tests with no features and `std,num-traits` |

Linux jobs use Ubuntu 24.04. Cargo uses the committed lockfiles. Downloaded
Actionlint, FLINT, and cross-compiler archives have fixed versions and SHA-256 checks.

The native job caches FLINT 3.6.0 by compiler CPU settings, installs it in the system
library directory, and runs `ldconfig`. Cargo's `--config` selects cross-linkers,
runners, and compiler flags. [Cross.toml](../../Cross.toml) forwards POWER9's
`QEMU_CPU` to its container. Unavailable x86 backends are reported and skipped.

Each Miri feature set has four disjoint test groups: interfaces, division,
multiplication, and other arithmetic. The default property budget is eight cases.
[.cargo/config.toml](../../.cargo/config.toml) configures Miri environment forwarding
and disables property-test file persistence. Global precision tests run in the
separate [global_precision executable](../../src/int/logic/tests/precision.rs).

[fuzz.yml](fuzz.yml) runs `int_ops` and `math_ops` every Sunday at 01:17 UTC, on
manual dispatch, and when called by CI. Runs use all features and the GNU x86-64
target. Scheduled runs last five hours per target; manual runs accept 1 to 18,000
seconds. Logs and crash inputs are retained for 30 days. Corpus saving is optional.
See [fuzz/README.md](../../fuzz/README.md) for test coverage.

Local workflow and tool checks:

```sh
actionlint
python3 tools/audit.py --tests
cargo +1.93.0 check --locked --lib --all-features
```

Tool checks use ShellCheck, Matplotlib, binutils, and LLVM tools.
