# mp-anafis — Project Guidelines

This file defines the architecture, correctness rules, performance discipline,
organization, and release gates for work on `mp-anafis`.

## 1. Project and priorities

`mp-anafis` is a high-performance multi-precision mathematics library written
in Rust. Its current integer engine provides:

- `MpUint`, an unsigned arbitrary-precision integer backed by native
  `usize` limbs (`Limb = usize`). Serde `[u64]` wire-format serialization is a
  planned roadmap item, not yet implemented (see `docs/int/api-inventory.md`).
- `MpInt`, a signed wrapper that delegates magnitude arithmetic to the
  unsigned engine while enforcing sign and two's-complement boundary rules.

`InternalMpUint` and `InternalMpInt` store up to `INLINE_LIMBS = 4` limbs
inline: 256 bits on 64-bit targets, 128 bits on 32-bit targets, and 64 bits on
16-bit targets. Reusable scratch buffers serve larger algorithms without
changing the value representation.

Priorities, in order:

1. Mathematical correctness and representation invariants.
2. Memory safety, including localized `unsafe` kernels.
3. Portability across 16-, 32-, and 64-bit `usize`.
4. GMP-competitive performance, especially allocation-free and in-place paths.
5. Idiomatic public APIs with native-integer ergonomics.

Performance never excuses an unproved invariant or architecture-specific
assumption. Correctness never excuses avoidable hot-path allocation or branches
once the relevant invariants have been proved.

## 2. Architecture boundaries

- Public constructors, methods, trait implementations, conversions, and
  precision policies belong in `api/`.
- Unsigned storage, raw limb arithmetic, bitwise logic, number theory, scratch
  management, and algorithm dispatch belong in `logic/unsigned/`.
- Signed internals delegate magnitude work to `InternalMpUint` and add only
  sign, signed-domain, and two's-complement semantics.
- Target selection, assembly, SIMD, and architecture-specific `cfg` logic
  belong in `src/int/logic/unsigned/math/arch/`. Generic callers must not
  duplicate target-selection policy.
- Generated crossover thresholds control production dispatch. Do not hard-code
  personal-machine thresholds into generic algorithm code.

## 3. Correctness, domains, and internal contracts

### Representation and arithmetic proofs

- Never assume normalization, active length, initialized capacity, aliasing,
  pointer alignment, or scratch size without checking it at the boundary or
  proving it from a caller invariant.
- Normalized integer representations have no unused high zero limbs. Algorithms
  that intentionally use padded or semi-normalized buffers must document where
  that weaker invariant begins and ends.
- Every bound, carry/borrow argument, pointer offset, shift count, scratch
  layout, and capacity calculation must remain valid for 16-, 32-, and 64-bit
  limbs.
- Use proof comments for non-obvious arithmetic. State the mathematical bound
  or invariant that makes an index, cast, unchecked operation, correction
  count, or destination width valid.
- Use checked arithmetic for genuinely fallible size calculations. Use
  wrapping arithmetic only when reduction modulo `2^N` is intended or a prior
  proof makes overflow impossible, and document which case applies.

### Public domain behavior

- Public APIs accepting arbitrary values must preserve their documented domain
  behavior through `Option`, `Result`, saturation/wrapping, or an explicitly
  documented panic contract.
- Genuine failures such as division by zero, a nonexistent modular inverse,
  bounded-precision overflow, and unsupported conversions must not be erased
  merely to simplify an internal call graph.
- Library functions must not panic for arbitrary inputs unless that panic is
  part of the public contract.

### Validate once; keep proved kernels infallible

- Validate domain conditions once at the public or dispatch boundary. After
  validation, internal hot-path kernels return their result directly or `()`;
  they must not propagate `Option`, `Result`, or a success `bool` for a state
  that their preconditions make impossible.
- Use `debug_assert!` to check proved internal preconditions when useful. It
  must not perform mutation or replace public validation.
- Preserve reachable control flow. Tier rejection, scratch fallback, quotient
  truncation applicability, bounded reciprocal correction, and exact fallback
  are algorithm states, not impossible errors. Represent them with a clear
  boolean or narrowly named internal enum.
- Never remove an exact fallback solely to reduce branch count. First prove the
  stronger invariant and cover its threshold and failure boundary with tests.

## 4. Unsafe code and hot-path performance

`Cargo.toml` denies unsafe code by default. Localized allowances are permitted
only in validated arithmetic kernels, architecture backends, and scratch/storage
internals where the optimization is necessary.

- Every `unsafe` block must have an immediately preceding `// SAFETY:` comment.
  The comment must cover bounds, initialization, aliasing, alignment, lifetime,
  capacity, and architecture assumptions that apply.
- Every `#[allow(...)]` or `#![allow(...)]` must include a specific
  `reason = "..."`. Fix the code instead when no essential architectural or
  measured performance reason exists.
- `get_unchecked`, `unwrap_unchecked`, raw pointers, branchless arithmetic, and
  inline assembly are acceptable only after input validation and a complete
  proof. Prefer the safe form when it compiles to equivalent code.
- `#[inline(always)]`, unchecked indexing, casts, and lint allowances in hot
  paths must earn their complexity through a proven architectural need or
  repeatable benchmark evidence.
- Performance claims require before/after benchmarks. Benchmarking is opt-in;
  if it is not authorized, report it as skipped and do not make a performance
  claim. When run, use reusable buffers, identical operands, isolated sizes and
  shapes, CPU pinning when available, and A/B/B/A ordering for close results.
  Separate forced-tier from production-dispatch measurements.
- Benchmark and autotuner constructors perform shape, non-zero divisor,
  destination-width, and scratch-capacity validation before timing. Reusable
  `run` methods do not repeat impossible checks or return impossible errors in
  the measured loop.

## 5. Modules, namespace impls, and visibility

### Directory and file organization

- Every `mod.rs` file is pure module plumbing: module documentation,
  declarations, wiring imports, and the minimum intentional re-exports only.
  It must not contain function bodies, `impl` blocks, algorithms, helpers,
  state, or tests; place those in focused sibling files. Apply the same rule to
  `lib.rs` except for crate-level attributes and documentation.
- Production Rust source files should normally remain near or below 500 lines.
  This is a cohesion signal, not an artificial split trigger: a cohesive file
  may modestly exceed it, including around 600 lines, when splitting would only
  create tiny files or obscure one responsibility. Do not add logic to an
  oversized file whose responsibilities are already mixed; split it by
  algorithmic responsibility as part of the next substantial change in that
  area, without widening unrelated tasks.
- Do not create new production implementation files below roughly 200 lines
  merely to satisfy a namespace or line-count rule. Keep a cohesive
  responsibility together; small `mod.rs` plumbing, dedicated tests,
  independent algorithm components, and real safety/state/facade boundaries
  are the intentional exceptions. Files below
  `src/int/logic/unsigned/math/arch/` are explicitly exempt: target backends,
  selectors, and runtime dispatchers remain separate even when they are very
  small. Merge other existing small implementation files only when the
  surrounding area is already being substantially reorganized.
- Prefer separate test modules or `tests/` subdirectories when colocated tests
  would obscure the production implementation.
- Use crate-relative or shallow relative imports. Avoid paths such as
  `super::super::super`; `tools/import_audit.py` is authoritative.
- Production implementation files import project-local names only from their
  immediate parent as `super::Thing`; the parent `mod.rs` wires deeper
  relationships once. When crossing a top-level subsystem boundary, import
  through that subsystem's narrow facade as `crate::subsystem::Thing`; do not
  reach into its implementation modules. Import error types directly through
  `crate::error::Thing` instead of propagating them through parent modules.
  Plumbing files may bind a direct child as `child::Thing` and parent items as
  `super::Thing`. Canonical `core::`/`alloc::`/`std::` and external-crate paths
  retain their normal depth. Test imports are exempt.
- Architecture selector/provider files and operation-local
  `runtime_dispatch.rs` files under `math/arch/` may import their concrete
  backend modules directly. This exception covers backend wiring only: deep
  relative paths, cross-subsystem implementation access, and selection policy
  outside `math/arch/` remain forbidden.

### Import order

- Order production private import groups by origin: standard library (`core`
  and `std`), `alloc`, external crates, `crate`, `super`, then `self` or
  direct-child plumbing imports. Omit empty groups and place exactly one blank
  line between groups. Within the standard-library group, `core` sorts before
  `std`. Tests should follow the same order when touched, but do not block the
  production import audit.
- Keep all imports from the same origin group together and sort separate
  statements lexicographically by their source path. Let `rustfmt` order names
  inside a use tree; do not create separate groups for macros, traits, types,
  functions, aliases, or `crate::error`.
- An import's `#[cfg]` or lint attribute stays immediately attached to that
  import and moves with it. Classify the import by its path, not its attribute.
- Intentional `pub use` facade exports are not dependency imports. In plumbing
  files, place them after module declarations and private wiring imports, in a
  separate re-export block sorted by source path.

### Inherent methods and algorithm namespaces

- Put operations naturally owned by a real value or state directly on that
  type. Inherent methods on `InternalMpUint`, scratch owners, plans, and tuner
  runners are preferable to detached wrappers.
- When one cohesive algorithm family spans several files and would otherwise
  expose a large flat function list, use a descriptive zero-sized namespace
  type. Each algorithm file contributes its own `impl` block.
- Use full names such as `Division` and `Multiplication`; do not use `Div` or
  `Mul`, which collide with `core::ops` traits.
- Namespace impls contain the cross-file family surface. Helpers used by only
  one file remain private free functions below the impl block.
- Production implementation files import project-local algorithm surfaces as
  namespace types or real value/state owners from `super`; free helper
  functions remain private in the one file that uses them. A public free
  function is a review signal that the operation belongs on a namespace or
  inherent type.
- A namespace propagated through several unrelated folders is a cohesion
  signal: reconsider the folder boundary or introduce one owning orchestrator.
  Broad propagation is expected for `InternalMpUint` and reusable stateful
  `Tuner` types. Necessary cross-family arithmetic dependencies may be encoded
  in `tools/structure_audit.py` only as narrow namespace-to-consumer paths;
  never suppress every consumer of an algorithm namespace globally.
- Inline trivial one-use wrappers and helpers at their call site instead of
  preserving or introducing a separate function. Keep a small named function
  only when it establishes a real API, trait, invariant, safety, dispatch, or
  reusable algorithm boundary, or when benchmark evidence justifies it.
- Refactor to the clean final design as if the legacy shape had never existed.
  Do not preserve an old function, method, name, module path, or re-export as a
  forwarding compatibility wrapper, deprecated alias, or duplicate entry
  point. Update every source, test, benchmark, example, and documentation call
  site in the same change, then delete the obsolete surface. Preserve backward
  compatibility only when the user explicitly requires it.
- Do not create a namespace type merely because a file is long. Split unrelated
  responsibilities first.

### Visibility

- `lib.rs` keeps implementation modules private and re-exports only the
  supported user-facing surface. Do not expose both a crate-root item and an
  alternative public module path to the same item.
- Expose the smallest internal facade at the highest practical boundary. If an
  ancestor module is private, cross-file items below it use plain `pub`; the
  private ancestor already prevents external access. Use `pub(crate)` at the
  gate only when an externally public ancestor would otherwise expose the
  item. Repeating `pub(crate)` below either gate is redundant and may trigger
  Clippy.
- Sibling subsystems import only that facade as `crate::subsystem::Thing`, not
  the subsystem's `api`, `logic`, storage, or algorithm modules.
- Do not introduce `pub(super)` or `pub(in ...)`. A plain `pub` item below a
  private or `pub(crate)` ancestor is still internal and is not part of the
  supported external API.
- Fields on a type intentionally re-exported to users may remain `pub(crate)`
  when the field is internal representation and must not become externally
  writable. The private-ancestor plain-`pub` rule applies to subtree wiring,
  not to weakening an externally visible type's encapsulation.
- Tighten existing visibility when substantially reorganizing an area, without
  creating unrelated cleanup churn.
- Exception: `src/int/tune_api/` may expose `#[doc(hidden)] pub` items behind
  `_internal-tune`, because Cargo benchmark and binary targets are separate
  crates. That facade must remain narrow and must not leak into the normal
  production API.
- Forced-tier and autotuner behavior belongs on a reusable stateful `Tuner`.
  Tuning convenience must not enlarge production algorithm namespaces.

### Source-file order

Use this order unless a macro or language constraint requires otherwise:

1. Module documentation (`//!`).
2. Inner attributes.
3. Imports.
4. Constants and type declarations.
5. Inherent/trait impls and the crate-visible family surface.
6. Private helper functions.
7. Colocated test modules, when a separate test file is not clearer.

## 6. Lints, documentation, and source changes

- `Cargo.toml` is the authoritative lint configuration. Clippy `all`,
  `pedantic`, `nursery`, `perf`, and the project-specific denies must pass
  without new warnings.
- Do not silence `dead_code`, `unused`, or similar warnings for staged work.
  Integrate the item, keep the warning visible during active local work, or
  remove code known to be obsolete before final readiness.
- Avoid unchecked primitive casts unless the source and destination ranges are
  proved on every supported pointer width. Any permitted cast allowance needs
  that proof in its reason or adjacent comment.
- Mathematical short names such as `lo`/`hi`, `x0`/`x1`, `carry`/`borrow`, and
  `q`/`r` are preferred where conventional. A `similar_names` allowance is
  acceptable only with a reason stating that convention.
- Every externally public item requires accurate Rustdoc, including domain,
  error, and panic behavior. Crate-visible arithmetic surfaces require enough
  documentation to state their invariants and safety preconditions.
- Preserve accurate comments and documentation; update or remove them when a
  refactor makes them stale.
- Do not leave `todo!()`, `unimplemented!()`, placeholder branches, or comment
  stubs in production paths.
- Read each target file and its callers before editing. Make source changes
  manually with the environment's dedicated patch/editing tool. Do not use
  Bash, Python, Perl, or generated rewrite scripts to modify or refactor source
  code. Formatters and repository-provided verification, generation, audit, and
  benchmark scripts may be executed as intended.
- Preserve unrelated work in a dirty checkout. Inspect `git status` and the
  exact diff before attributing failures or claiming readiness.

## 7. Tests and regressions

- Core algebraic operations require property-based or comprehensive fuzz
  coverage across relevant widths, signs, capacities, operand shapes, aliasing,
  and dispatch thresholds.
- Properties must assert meaningful identities or reference equivalence, not
  merely that a function returns `Some` or does not panic. Bound expensive
  generators so every case remains informative and finishes promptly.
- A bug fix requires a minimized static regression test and, where practical, a
  generalized property covering the failure class.
- If fuzzing discovers a crash, panic, or undefined behavior, add the minimized
  input to the permanent fuzz corpus. A unit regression may also document the
  case, but it does not replace the corpus seed.
- Unsafe-heavy code needs shrinkable property coverage for bounds, aliasing,
  initialization, and architecture-sensitive cases. Miri is opt-in; when
  authorized, run it after changing unsafe code or its preconditions.
- Threshold-sensitive algorithms need tests immediately below, at, and above
  each affected crossover, plus algebraic recombination checks such as
  `q*d + r == n` and `r < d`.

## 8. Development and verification workflow

### During development

Use the narrowest applicable checks while iterating, widening them as the
changed invariant requires:

- `cargo test --workspace --all-features --lib --tests`
- `cargo clippy --all-targets --all-features -- -D warnings` and the
  corresponding `--no-default-features` configuration
- `cargo fmt --all`
- `python3 -B tools/import_audit.py`
- `python3 -B tools/structure_audit.py`

Run threshold-neighbor, property, and architecture checks while iterating when
the changed invariant requires them.
