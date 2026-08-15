# MpAnafis Source Organization

This guide describes the repository's current module boundaries. It is a
maintenance contract: when the code and this document disagree, fix both in the
same change.

## Principles

- Public users enter through an `api` module and never through `logic` paths.
- Raw limb algorithms, scratch management, target selection, and assembly live
  under `logic`.
- API code owns only public type construction, precision/domain policy,
  ergonomic forwarding, and trait implementations.
- Visibility is the narrowest absolute subtree that contains every caller.
- `pub(super)` is not used. Its meaning changes when a file is moved and makes
  boundary review unnecessarily positional.
- Source and test files stay focused and normally below 500 lines.

## Integer Layout

```text
src/int/
├── mod.rs                 # public re-export boundary (`pub use api::*`)
├── types.rs               # platform-adaptive limb definitions (`Limb = usize`, `DoubleLimb`, `LIMB_BITS`, `INLINE_LIMBS`)
├── api/                   # public types, methods, and trait implementations
│   ├── mod.rs             # private logic imports and public API registry
│   ├── int/               # MpInt API
│   ├── uint/              # MpUint API
│   └── ops/               # operator trait implementations
├── logic/                 # private implementation namespace
│   ├── signed/            # signed integer arithmetic, bitwise, and representation logic
│   └── unsigned/          # unsigned arbitrary-precision core
│       ├── bitwise/       # bitwise logic, scanning, and shifts
│       ├── cmp/           # magnitude and equality comparisons
│       ├── convert/       # native primitive, string/radix, and byte serialization conversions
│       ├── memory/        # in-place slice transforms and scratch buffers
│       ├── math/          # core mathematical operations
│       │   ├── arch/      # all target and CPU-feature selection (x86_64, AArch64, ARM, RISC-V, etc.)
│       │   └── mul/       # multiplication tower (Basecase -> Karatsuba -> Toom-Cook -> SSA/NTT)
│       ├── storage.rs     # inline storage buffer (`INLINE_LIMBS = 4`) vs. heap vector management
│       └── properties.rs  # structural queries, bit lengths, and trailing zero checks
├── tests/                 # categorized integer API properties
└── tune_api/              # feature-gated facade for external bench/bin crates (`_internal-tune`)
```

`src/int/mod.rs` keeps both `api` and `logic` private and exports the supported
surface with `pub use api::*`. The `_internal-tune` exception is documented
below.

## API and Logic Boundary

The API layer may:

- define public types and validated newtypes;
- enforce precision, sign, overflow, and domain contracts;
- translate internal return values into `Option`, `Result`, or public wrappers;
- implement standard and ecosystem traits;
- delegate to private logic functions.

The API layer must not contain:

- raw limb loops;
- multiplication, division, transform, or number-theory kernels;
- scratch allocation strategies used by those kernels;
- inline assembly, target-feature detection, or backend selection.

Small policy expressions in an API method are intentional. Moving a two-line
overflow or precision decision into `logic` would hide the public contract
rather than improve separation.

Bare private items are preferred when every caller is in the same module.
`pub(crate)` is reserved for the few values that genuinely need the whole
crate, such as private fields of public integer wrappers. External API items are
plain `pub` and must originate from the API boundary.

Plain `pub` may appear inside a private leaf module when Rust requires that
visibility for a restricted parent re-export. The private ancestors still seal
the path.

## Multiplication and Architecture Boundaries

`logic/unsigned/math/mul` contains architecture-neutral algorithms. A tier may
call an architecture-neutral kernel function, but it must not contain target
`cfg`, feature detection, or assembly.

`logic/unsigned/math/arch` owns:

- target and pointer-width `cfg` selection;
- runtime CPU-feature selection;
- inline assembly and its portable fallback;
- cross-backend agreement properties.

Each operation has one canonical public-within-logic entry point. Backend names
remain private to `arch`, except for test-only cross-backend hooks. Do not copy
an assembly body merely to give it an ADX/BMI2 label; share the canonical kernel
when the instruction stream is identical.

## Internal Tuning Boundary

The tuner binary and benchmark targets are separate Cargo crates, so they
cannot call `pub(crate)` library items. The `_internal-tune` feature therefore
enables `int::tune_api`, a `#[doc(hidden)]` public facade.

This facade is the only allowed exception to the normal public API boundary:

- expose only operations actually consumed by `tools/tune.rs` or a named
  benchmark;
- keep scratch allocation outside timed loops;
- validate raw slice sizes before entering unchecked kernels;
- never re-export the private `logic` module itself;
- keep the feature disabled in normal builds.

## Imports

- Use the supported module surface for cross-component imports.
- Inside one component, prefer shallow relative imports routed through the
  nearest module registry.
- Do not use deep `crate::...::logic::...` paths from API or benchmark leaves;
  centralize the boundary import in their parent facade.
- Avoid inline-qualified calls when a clear import is possible.
- Avoid aliases unless they disambiguate genuine domain names or backend roles.

`python3 tools/import_audit.py` enforces the repository-specific boundary and
alias rules.

## Test Placement

Tests are primarily generalized properties or fuzz coverage. Fixed
input/output unit tests are used only where a documented edge or a regression
bug demands a stable static case (e.g. the bounded-width cases in
`src/int/tests/bounded.rs`).

```text
src/int/tests/                                  public integer behavior
src/int/api/ops/tests/                          operator contract properties
src/int/logic/unsigned/math/mul/tests/          multiplication tier/kernel properties
src/int/logic/unsigned/math/arch/tests/         architecture kernel properties
tests/                                          crate-level integration properties
fuzz/                                           fuzz targets and permanent corpus seeds
```

`mod.rs` files declare categories and hold only genuinely shared test support.
Do not place a large inline `#[cfg(test)] mod tests` beside production code.

## Bench Placement

- `public_api` (`mp vs rug/gmp api`) contains comprehensive public-API comparisons against Rug/GMP across every implemented family listed in `docs/int/api-inventory.md`, including allocation-sensitive (`divan::AllocProfiler`) addition routines and Mp-only workloads.
- `internal_improvement` (`mp vs rug gmp for internal improvement`) contains direct tier, crossover, and architecture-kernel measurements through the internal tuning facade (`_internal-tune`).
- `shared/` holds common support modules and generator utilities across benchmarks.
- Operand generation and allocation belong outside timed closures unless the
  allocation itself is the operation being compared.
- If a production algorithm necessarily allocates internally, label its
  benchmark `end_to_end`; do not imply an allocation-neutral measurement.

## File Shape

Rust files follow this order:

1. module documentation;
2. imports;
3. public or restricted items;
4. private helpers;
5. test-module declarations.

Every public item has Rustdoc. Unsafe blocks have a local `SAFETY` proof, and
every lint allowance has a narrow `reason`.

## Review Checklist

- Does any external path expose `logic`?
- Is every restricted item visible only to the smallest stable absolute scope?
- Is all target selection under `math/arch`?
- Are API methods policy wrappers rather than raw algorithms?
- Are tests and benches in the correct categorized tree?
- Are all Rust files focused and near or below 500 lines?
- Do strict Clippy, import audit, property tests, and architecture checks pass?
