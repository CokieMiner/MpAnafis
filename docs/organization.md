# MpAnafis Repository Organization

This guide maps the source, tests, benchmarks, tools, and documentation.
[AGENTS.md](../AGENTS.md) defines the maintenance rules.

## Repository layout

```text
src/                       library source
build.rs                   Cargo build script
build_support/             shared build support
examples/                  runnable public API examples
benches/                   executable benchmarks
tools/                     development tools
docs/                      specifications, references, and reviewed records
fuzz/                      fuzzing source and harness tests
.github/workflows/         CI configuration
target/                    ignored build and working artifacts
```

## Library source

```text
src/
  lib.rs                         external API exports and feature gates
  error/                         public error types
  int/
    types.rs                     native limb types and shared constants
    api/
      precision.rs               public precision types and context
      types.rs                   integer type definitions
      types/                     methods, conversions, and trait implementations
        int/                     signed integer methods
        uint/                    unsigned integer methods
        ops/                     operator implementations
    logic/
      precision.rs               internal precision context
      signed/                    signed representation and arithmetic
      unsigned/                  limb storage and arithmetic
        math/arch/               CPU selection and architecture kernels
    tests/                       public integer API tests
    tune_api/                    feature-gated interface for tuning and benchmarks
  parallel/                      execution and worker selection
```

`src/lib.rs` defines the external API through explicit exports. Implementation
modules remain private.

The integer API handles public precision, domain, and error policies. `logic`
provides representations, arithmetic algorithms, and kernels.

`api/types.rs` defines `MpInt` and `MpUint` with private fields. Implementations
under `api/types/` are descendant modules and can access those fields.

## Tests

Rust tests reside in `tests.rs` or `tests/` beside the module they exercise.
Python tests reside in each tool package's `tests/` directory.
Fuzzing entry points are in `fuzz/fuzz_targets/`; shared checks and harness
tests are in `fuzz/src/`.

## Build support and tools

| Location | Responsibility |
| --- | --- |
| `build_support/` | Build configuration, tuning profiles, and generated tables. |
| `tools/bench.py`, `tools/benchmark/` | Benchmark planning, execution, reporting, and export. |
| `tools/tune/` | The `mp-tune` binary and tuning implementation. |
| `tools/audit.py`, `tools/audit/` | Source and tooling audits, function graphs and review evidence; combined regression runner. |
| `tools/asm_analyzer.py`, `tools/asm_analyzer/` | Assembly analysis. |

## Benchmarks and records

| Location | Contents |
| --- | --- |
| `benches/public_api/` | Public integer API comparisons. |
| `benches/internal_improvement/` | Arithmetic kernel, algorithm, and crossover studies. |
| `target/bench-results/` | Local working measurements and reports. |
| `docs/int/benchmarks/` | Reviewed benchmark records and plots. |

## Documentation

| Subject | Guide |
| --- | --- |
| Specifications, API inventories, algorithms, and architecture references | [Documentation index](README.md) |
| Public benchmark source | [Public benchmark guide](../benches/public_api/README.md) |
| Internal benchmark source | [Internal benchmark guide](../benches/internal_improvement/README.md) |
| Benchmark execution and reporting | [Benchmark driver guide](../tools/benchmark/README.md) |
| Reviewed performance records | [Benchmark records index](int/benchmarks/README.md) |
| Tuning | [Tuner guide](../tools/tune/README.md) |
| Source audits | [Audit guide](../tools/audit/README.md) |
| Assembly analysis | [Assembly analyzer guide](../tools/asm_analyzer/README.md) |
| Fuzzing | [Fuzz guide](../fuzz/README.md) |
| Continuous integration | [CI guide](../.github/workflows/README.md) |
