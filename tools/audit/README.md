# Source and tool audits

Run the combined checks from the repository root:

```sh
python3 tools/audit.py
python3 tools/audit.py --tests
python3 tools/audit.py --json
python3 tools/structure_audit.py --function-report target/structure-review/functions
```

`--tests` runs the audit, benchmark, and assembly-analyzer Python
suites in separate interpreters. It supplies their import roots and excludes
the tuner. CI uses this entry point. Findings and failed suites produce a nonzero
exit status; `--json` keeps one structured result on stdout and test output on
stderr.

## Checks and source coverage

| Check | Rules |
| --- | --- |
| Rust structure | Structural registries, declaration order, separate tests, no placeholders, visibility gates, and architecture selection confined to `math/arch` in integer logic. Function relationships produce separate review candidates. |
| Rust imports | Facade boundaries, explicit names, grouping, ordering, duplicate bindings, and unused aliases. Tests may bypass production boundaries but retain grouping. |
| Rust lint attributes | Both `allow` and `expect` require a nonempty literal reason, including nested `cfg_attr` branches. Neither may suppress `dead_code`. |
| Python tools | Valid syntax, structural package facades, explicit imports, and test implementations in `tests/`. Tuner sources and ignored assembly captures are excluded. |
| Public benchmarks | Matching engine declarations, argument ladders, sampling, counters, and reset strategy. |

Rust discovery includes `src/`, both benchmark suites, examples, integration
tests, fuzz source and entry points, build support, `build.rs`, and tuner source.
Generated fuzz artifacts are excluded. Library facade rules apply to library and
tuner production modules; executable consumers retain their own import roots.

`pub(crate)` is accepted only on inherent functions of types reachable from
`src/lib.rs`, including aliases and feature-gated public modules. Private module
declarations alone do not expose their types. Restricted fields, free functions,
constants, trait implementations, and sealed types remain findings.

Direct sibling dependencies must pass through their parent facade. A module may
import its own declared children. The architecture selection DSL and generated
threshold reexports retain their documented, narrowly scoped exceptions.

Production Rust files over 500 lines are reported for cohesion review;
architecture backends are exempt. `--deny-oversized` makes this review a failure.

## Function relationships

Both audit entry points build a source function graph by default. It records
calls, function values, possible macro references, and unresolved call sites.
Explicit imports and reexports, `Self`, simple typed parameters, and direct
fields of `self` supply name-resolution evidence. All cfg alternatives remain
visible; mutually recursive functions form one ordering component.

The review checks:

- `pub` functions whose resolved users are all in their module or descendants.
- Helpers placed above a resolved caller in the same source file.
- Functions with a single resolved production consumer in another file of the
  same subsystem, or non-leaves with external consumers and no local consumer.
- Associated functions without a receiver or explicit dependency on their type.
- Functions with no observed call or value use.

Exported APIs, trait methods, conditional declarations, type constructors,
private-field ownership, namespace types, architecture dispatch, and subsystem
boundaries constrain the applicable checks. Tests and executable consumers count
as users; test functions are not relocation candidates. Tuner sources supply
consumer evidence but receive no function-review candidates.

These are advisory source observations. A single *resolved* consumer can coexist
with unresolved uses; placement candidates include their possible-use count.
A function with no resolved project calls is only an observed leaf. Neither this
classification nor a textual call order proves execution order or dead code.
Use `--json` for all candidates and evidence, or `--deny-function-reviews` to
make candidates fail a deliberate review run. They do not fail CI by default.

`--function-report DIR` writes `functions.json` and `functions.dot`. Both retain
the full graph. DOT uses solid edges for resolved calls, dashed edges for
resolved function values, and dotted edges for possible references. JSON also
records review evidence, capture time, and source hashes. Reports inside the
repository must stay under ignored `target/`; explicit external output
directories are also accepted.

Individual entry points remain available:

```sh
python3 tools/structure_audit.py --json
python3 tools/import_audit.py --json
python3 tools/check_allows.py --check --all-lints
python3 tools/bench.py check --source-only
```

## Implementation and limits

The shared Rust scanner masks comments and literals while preserving positions,
then reads balanced attributes, items, and use trees. Source branches are
inspected regardless of the host target. Source discovery, lint parsing,
visibility resolution, import rules, and Python checks have separate modules;
tests are grouped by those responsibilities.

These checks do not expand macros or replace Rust name resolution, type checking,
cfg compilation, or Clippy. Visibility resolution follows ordinary module files
and explicit export paths; custom `#[path]` modules and types exposed indirectly
through signatures require compiler-assisted review. A literal reason can be
checked structurally; its adequacy and scope require review.

Mathematical correctness, unsafe proofs, transaction boundaries, the necessity
of reviewed function boundaries, academic references, generated code, and performance evidence still
require the corresponding manual review and execution checks in
[AGENTS.md](../../AGENTS.md).
