# MpAnafis - Project Guidelines

This document specifies the architecture, invariants, memory safety contracts, performance discipline, module organization, and verification procedures for the `MpAnafis` Rust crate (`mp_anafis` in Cargo).

The current repository map is [docs/organization.md](docs/organization.md).
Benchmark source conventions are in [benches/public_api/README.md](benches/public_api/README.md),
and the shared driver is documented in [tools/benchmark/README.md](tools/benchmark/README.md).

## 1. Priorities & System Architecture

`MpAnafis` is a multi-precision integer library in Rust:
- **`MpUint` / `MpInt`**: Public arbitrary-precision unsigned and signed integers backed by native `usize` limbs.
- **`InternalMpUint` / `InternalMpInt`**: Core representations with 4 inline limbs (`INLINE_LIMBS = 4`), transitioning to heap buffers on overflow.

**Design Priorities (strictly ordered):**
1. Mathematical correctness and representation invariants.
2. Memory safety and sound encapsulation of `unsafe` kernels.
3. Architecture portability across 16-, 32-, and 64-bit targets.
4. Execution efficiency through reduced mathematical work, algorithmic complexity, memory traffic, and control-flow overhead.
5. Ergonomic public APIs conforming to standard Rust idioms.

Correctness precedes performance optimization. Performance never excuses an unproved invariant, and proved invariants must avoid unnecessary allocation or branching.

**Subsystem Boundaries:**
- **`api/`**: Public surface, conversions, standard trait implementations, precision policies, and transactional error boundaries.
- **`logic/signed/`**: `InternalMpInt` magnitude-sign arithmetic, two's-complement bitwise logic, and division rounding policies (Truncated, Floor, Ceil, Euclidean).
- **`logic/unsigned/`**: Raw limb arithmetic, storage representations, bitwise kernels, scratch management, and multiplication dispatch towers (Basecase -> Karatsuba -> Toom-Cook -> Schönhage-Strassen).
- **`math/arch/`**: Architecture-specific SIMD, assembly kernels, and CPU feature detection. Generic algorithms must not perform target-specific branching.
- **`tune_api/`**: Feature-gated interface (`_internal-tune`) exposing tuning parameters to external benchmarks and the `mp-tune` binary.
- **Algorithmic Thresholds**: Performance crossovers use empirical tuning parameters. Mathematical minimum sizes, representation limits, and safety bounds remain explicit correctness constraints.

---

## 2. Correctness, Invariants & Contracts

- **Representation Invariants**:
  - **Canonical Zero**: `InternalMpInt` strictly normalizes zero to positive (`abs.is_zero() ==> is_positive == true`). Negative zero (`-0`) is invalid. `InternalMpUint` uses an empty limb slice for zero; a nonzero magnitude ends in a nonzero most-significant limb.
  - **Inline Capacity**: Fixed at 4 limbs (`INLINE_LIMBS = 4`) across all targets.
- **Transactional Rollback**: Under bounded precision (`BoundedPrecision`), mutating operations (`add_assign`, `sub_assign`, etc.) must not expose unvalidated intermediate residues if a bounded panic occurs. Validate before receiver mutation, or evaluate in temporary scratch.
- **Assignment Precision**: `assign_*` preserves destination precision, as assignment operators do. Failure preserves the destination value and precision; operand or ambient precision must not silently replace destination metadata.
- **Precondition Validation**: Validate preconditions at public and dispatch boundaries. Internal hot-path kernels are infallible and must not propagate unreachable `Option`, `Result`, or error flags.
- **Infallible Unwrapping**: Express an infallible state directly when possible. `unwrap_unchecked()` requires a local proof citing the upstream invariant and a justified hot-path benefit. A proved invariant permits unchecked access; it does not require introducing `unsafe` when safe code has equivalent cost.
- **State Preservation**: Division-by-zero handling and algorithm fallbacks (tier rejection, scratch fallback, exact fallback) are valid execution states and must not be eliminated without mathematical proof.
- **Arithmetic Modes**: Buffer sizing and index calculations must use checked arithmetic. Wrapping arithmetic is restricted to proved modular reductions or bounded ring indices.
- **Pointer-Width Portability**: Slices, shifts, offsets, and capacities must behave identically across 16-, 32-, and 64-bit targets. Primitive casts require proof comments covering all pointer widths.
- **Production Completeness**: Placeholder stubs (`todo!()`, `unimplemented!()`) and dead branches are forbidden in production code.

---

## 3. Memory Safety & Performance Discipline

- **SAFETY Proofs**: Every `unsafe` block requires an immediately preceding `// SAFETY:` comment proving the obligations applicable to that operation: bounds, initialization, aliasing, alignment, lifetimes, capacity, or target prerequisites. State the actual proof rather than a generic checklist.
- **Lint Allowances**: Every `allow` or `expect` attribute requires an explicit, descriptive `reason = "..."` and the narrowest useful scope. Do not suppress a lint instead of correcting invalid code or a misplaced cfg gate.
- **Optimization Order**: Optimize in the following order. Establish correctness, safety, and representation invariants throughout; run correctness checks before performance measurements.
  1. **Mathematical Work Reduction**: Derive the minimum arithmetic needed for the result. Use proved identities, symmetry, bounds, and required precision to eliminate unnecessary products, evaluations, and intermediate results. Reuse shared computations and compute only the required limbs or coefficients when the omitted work is proved unnecessary.
  2. **Algorithmic Complexity & Data Flow**: Reduce asymptotic time and space costs, redundant passes, recursive subproblems, transforms, and memory traffic. Compare finite-size operation counts, constant factors, scratch requirements, allocation, and copying across the relevant operand sizes and shapes. Establish invariants once and reuse buffers and workspaces.
  3. **Generated-Code Review**: Inspect Rust MIR, optimized LLVM IR, and final emitted assembly for the changed routines and their actual compiled callers. Use the relevant optimized profile, target, CPU features, and Cargo features. Check dynamic branches, repeated bounds checks, unnecessary calls, allocation and copying, loads and stores, dependency chains, register spills, inlining, and vectorization. Verify that intended reductions survive compilation. Express mathematical and mechanical reductions explicitly in the source, even when LLVM could discover them. Avoid relying on compiler heuristics to discover or preserve optimizations.
  4. **Empirical Validation**: Benchmark the complete implementation after mathematical, algorithmic, and generated-code review. Compare baseline and candidate using identical operands, relevant sizes and shapes, and explicit worker budgets. Measure below, at, and above affected crossovers. Use measurements to establish workload benefits and choose dispatch thresholds.
- **Optimization Authority**:
  - *Local optimizations*: Directly implement changes supported by a mathematical proof, a mechanical reduction in work, or evidence from emitted code. Verify correctness and inspect the resulting code before benchmarking.
  - *Algorithmic and architectural redesigns*: Document the mathematical derivation, complexity, and storage bounds. The complete implementation may be assembled before running benchmark suites. Verify correctness during development, inspect generated code, then measure the integrated implementation.
  - *Micro-optimizations*: Identify the expected reduction in executed work using mathematical reasoning, emitted code, or profiling. Speculative tie-breakers, unchecked indexing, forced inlining, and branchless rewrites without this justification are forbidden. Supported candidates may be implemented before benchmarking; measured results establish their performance benefit.
- **Buffer Reuse & Commutativity**:
  - *Commutative operations* (`Add`, `Mul`, bitwise): Operands may swap to reuse larger pre-allocated buffers without post-processing.
  - *Non-commutative operations* (`Sub`, `Div`, `Rem`): Swapping ($A - B = -(B - A)$) incurs post-computation fixup (sign negation, zero checks, quotient adjustments). Permit swapping only when reusing the other operand's storage avoids an otherwise necessary heap allocation or reallocation. Prove this from the operation's storage requirements and available capacity; `rhs_len > self_len` alone is insufficient.
  - *Inline storage*: Use the storage class and static `INLINE_LIMBS = 4` capacity for inline operands. When both operands are inline, their capacities are equal. Reserve dynamic capacity comparisons for heap storage.
- **Hot-Path Auditing**: Audit source and emitted loops for duplicate arithmetic, repeated scans, redundant normalization, unnecessary allocations and copies, and invariant checks inside loops. Hoist invariant computations and predicates when the established contracts permit it.
- **Branch Minimization**: Reduce executed hot-path branches by hoisting invariant conditions, combining equivalent predicates, and removing checks proved redundant. Inspect their lowering to branches, conditional moves, or predicated instructions. Account for extra arithmetic, memory accesses, and dependency lengths introduced by branchless rewrites. Retain required domain checks, transactional failure paths, and algorithm fallbacks unless proved unreachable.
- **Execution Flow**: Enforce the sequence: `Validate once` -> `Dispatch once` -> `Establish invariants` -> `Execute infallible kernel` -> `Primitive leaf operations`.
- **Benchmarking Standards**: Use reusable buffers, identical operands, isolated matrix shapes, CPU pinning, and A/B/B/A ordering. Separate forced-tier microbenchmarks from production dispatch benchmarks.

---

## 4. Module Organization & Visibility

- **`mod.rs` as Module Registry**: `mod.rs` serves exclusively as a structural module registry. It contains zero code (no structs, enums, functions, constants, or traits). All types reside in dedicated child files.
  Item ordering in `mod.rs`:
  1. `//!` module documentation
  2. Inner attributes
  3. `use super::{...}`
  4. `mod ...;`
  5. `pub use ...;`
  6. `#[cfg(test)] mod tests;` (last, only if tests exist)
- **Module Facades & Import Paths**: Production dependencies pass through explicit imports or reexports in the parent `mod.rs` or type-owner facade. Parent and sibling dependencies use `super::Thing`, never `super::sibling::Thing`. Cross-subsystem dependencies use facades (`crate::subsystem::Thing`); production imports never bypass a facade to reach an implementation file. A facade may forward its own parent dependencies to its children. Tests may import implementation modules directly.
- **Opaque Type Ownership**: `src/int/api/types.rs` defines `MpInt` and `MpUint` with private fields and declares their implementation children directly under `types/`. Rust's descendant privacy rules permit those children to access the fields. Shared private precision and validation methods reside in the owner file.
- **Registry Macros**: The architecture-only `select_arch_kernel!` module-selection DSL is permitted in architecture registries. Macro definitions and kernel implementations remain in dedicated files; this exception does not admit arbitrary code-generating macros into `mod.rs`.
- **File Sizing**: Keep production files focused and normally at most 500 lines (`math/arch/` backends are exempt). Small cohesive modules and registries need not reach a minimum line count. Do not split or pad a file solely to meet a line target.
- **Function Size**: Avoid functions with fewer than about five lines of logic and pass-through wrappers unless required by the architecture, a trait interface, or an existing public API. Keep cohesive logic at its call site when extraction adds no structural boundary.
- **Namespaces & Workspaces**: Zero-sized types (`Division`, `Gcd`) group cohesive static routines. Stateful structs (`HgcdWorkspace`, `Tuner`) encapsulate scratch memory and algorithm state.
- **Visibility Gates**: Keep local implementation items private. Items shared across sealed module boundaries use plain `pub`, with external exposure controlled by explicit facade reexports.
  - Determine whether a type is externally reachable through `src/lib.rs`, including reexport chains and feature-gated public modules. A `pub` declaration inside a sealed module alone does not make a type externally reachable.
  - Permit `pub(crate)` only on associated functions or methods of externally reachable types when the function belongs in the type's inherent `impl`, requires access outside the owning module and its descendants, and must remain hidden from library users. Parse-error constructors are one example. Keep functions private when the owning module and its descendants provide sufficient access.
  - All other uses of `pub(crate)`, and all uses of `pub(super)` or `pub(in ...)`, are forbidden.
- **Import Grouping**: Apply the same grouping to production and test imports. Separate groups by blank lines in this order: `core`/`std`, `alloc`, external crates, `crate::`, `super::`, `self::`/child. Follow rustfmt's ordering within groups, including grouped imports. Benchmark crates use their own local support modules; generated paired benchmark modules may inherit the category's explicit imports with a reasoned wildcard allowance.
- **Cfg Placement**: Architecture backend and CPU-feature selection belongs in `math/arch`. Pointer-width correctness constraints and Cargo feature gates remain where their types and APIs require them. Gate modules, reexports, and dependent imports consistently; avoid scattering redundant identical gates through a backend already gated at its module boundary.

---

## 5. Source-File Structure & Technical Documentation

### Monotonic Call-Graph Ordering (The Data Cascade)
Source files follow downward execution order matching chronological data flow. Recursive algorithm calls may form cycles; helper placement still follows the main data flow rather than alphabetical order:
1. **Module Header & Declarations**: Module documentation (`//!`), attributes, imports, constants, and type definitions.
2. **Primary Driver / Facade Entry Point**: Inherent method or driver (e.g., `mul`, `div_rem`); validates boundaries and initiates execution.
3. **Workspace Partitioning**: Contiguous scratch memory partitioned linearly via `split_at_mut`.
4. **Algorithmic Sequence**: Functions appear in chronological execution order:
   - Forward transforms and point evaluations ->
   - Recursive subproblems and pointwise multiplications ->
   - Inverse transforms, interpolation, and reconstruction.
5. **Specialized Internal Routines**: Algorithmic helpers called exclusively by intermediate stages.
6. **Infallible Leaves & Leaf Kernels**: Architecture wrappers, bit-manipulation primitives, and raw limb loops placed at the bottom of the file.

### Technical Documentation & Comments
- **Scientific Progression**: Comments must explain the active mathematical identity, transformations, limb widths, carry/borrow limits, guard bits, and representation invariants.
- **Documentation Status**: Source comments, API inventories, and organization guides describe current mechanics. Design specifications may include explicitly labeled requirements and planned behavior; they must distinguish implementation from design. Benchmark records retain dated measurements and measurement details, including explicit uncertainty in archived captures. Avoid historical bug narratives and commented-out code in production source.
- **Rustdoc Standards**: Document API purpose, mathematical domain, panics, errors, and preconditions.
- **Tone & Formality**: Direct, simple, descriptive, scientific, and concise. State the active mathematics, mechanical invariants, and proofs; avoid conversational remarks, subjective commentary, and utility dumping grounds.

---

## 6. Test Layout & Verification

- **Strict Test Separation**: Production files contain no test logic. Tests reside in dedicated `tests.rs` or `tests/` directories located directly in the module folder. The only test-specific content in a production `mod.rs` is the final `#[cfg(test)] mod tests;` declaration; test-only imports and reexports are forbidden.
- **Coverage Requirements**: Core operations require property-based tests covering edge widths, slice aliasing, and algorithm crossover boundaries (testing below, at, and above crossover points).
- **Verification Scope**: Run checks appropriate to the changed behavior. Rust library changes require the library checks below. Benchmark changes require compilation, benchmark Clippy, and execution at selected relevant sizes. Tooling changes require their Python regression tests. Documentation-only changes require link/path and diff checks; do not rerun expensive numeric suites solely for prose edits.
- **Library Verification**:
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test --lib --all-features`
  - `python3 tools/structure_audit.py`
  - `python3 tools/import_audit.py`
  - `cargo bench --bench public_api --features "std,rayon,_internal-tune" --no-run`
- **Benchmark and Tool Verification**:
  - `python3 tools/bench.py check --source-only`
  - `cargo clippy --bench public_api --features "std,rayon,_internal-tune"`
  - `python3 tools/bench.py check --features "std,rayon,_internal-tune" --require-comparison` on a supported host with FLINT installed
  - `python3 -m unittest discover -s tools/benchmark/tests`
  - `python3 -m unittest discover -s tools/audit/tests`
  - Execute selected cases through `tools/bench.py run --smoke`, then measured pinned A/B/B/A runs for performance claims. Report the functions and sizes actually checked. Full ladders are deliberate performance experiments, not an implicit requirement for every edit.

### Benchmark Source and Result Placement

- Executable Rust benchmarks stay in `benches/public_api/` or `benches/internal_improvement/`.
- Public API cases follow domain/category/function/scenario organization with symmetric Mp and comparator setup, sampling, timed loops, and reset strategy. Use shared templates; each case measures one public function or one explicitly named scenario.
- Public API comparisons measure applicable Rug/GMP and FLINT equivalents, including documented composed equivalents. Compare against every registered reference and identify the fastest from measurements for the same operands, result contract, and worker budget. Keep target and dependency gates explicit.
- Use `tools/bench.py` for plans, execution, reports, and documentation export. Extend the shared reporting package for new plot requirements; do not add per-operation plotting scripts.
- Working runs go under ignored `target/bench-results/` (an explicit output directory is required). Keep their plan, metadata, exact raw captures, measurements, and report together. Raw runs, generated scripts, numeric tables, and validation outputs do not belong in Git.
- Reviewed time-versus-size curves go under `docs/int/benchmarks/<suite>/<record-name>/`, separated by category and function, with a short README and measurement manifest. Use `tools/bench.py publish` to validate complete public API runs and export those files. A curve requires at least two measured numeric sizes per engine; single-size comparisons remain tables. Never overwrite a retained record.
- Preserve incomplete legacy captures locally under `target/bench-results/`, with their limitations stated. Do not relabel them as current verified measurements. Retain or share full raw runs separately when publishing reproducible studies; hashes in a figure manifest cannot reconstruct absent evidence. A smoke test or a short harness-validation run is not evidence for a tuning decision.

---

## 7. Scope Discipline & Workflow Limits

- **Scope Discipline**: Modifications must directly improve correctness, performance, or guideline compliance. Unrelated refactoring or aesthetic churn is prohibited.
- **Public API Stability**: Stable public APIs must not be renamed or altered without explicit justification.
- **Direct Source Editing**: Maintain Rust source code directly; automated bulk regex/rewrite scripts are prohibited.
- **Audit Adherence**: All changes must maintain zero findings on `tools/structure_audit.py` and `tools/import_audit.py`.
