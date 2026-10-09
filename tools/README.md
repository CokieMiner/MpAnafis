# Development tools

| Entry point | Responsibility | Guide |
| --- | --- | --- |
| `audit.py` | Source checks, function graphs and Python regression suites. | [Audits](audit/README.md) |
| `bench.py` | Benchmark plans, execution, reports, and retained records. | [Benchmark driver](benchmark/README.md) |
| `asm_analyzer.py` | Emitted assembly extraction, analysis, and measured schedule comparisons. | [Assembly analysis](asm_analyzer/README.md) |
| `check_all_archs.sh` | Cross-target Clippy and assembly code generation. | [CI](../.github/workflows/README.md) |
| `tune/` | Multiplication tuning binary. | [Tuner](tune/README.md) |

Entry scripts delegate to focused package modules. Package `__init__.py` files
declare exports; implementations and tests reside separately.

```sh
python3 tools/audit.py --tests
```

This runs source/tooling checks and the four Python suites outside the tuner.
Cargo, Miri, emulator, and hardware checks remain separate execution steps.
The [audit guide](audit/README.md) states which project rules are checked and
which require manual proof or compiler-assisted review.
