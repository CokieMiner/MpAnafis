# Contributing to `MpAnafis`

Thank you for considering contributing to `MpAnafis` (`mp_anafis` on crates.io / `mp-anafis` on PyPI).

Contributions of all sizes are welcome, including bug reports, minimized reproducers, documentation fixes, benchmarks, and partial patches.

`MpAnafis` is built for high-performance arbitrary-precision arithmetic with strict correctness, inline small-value storage, and configurable precision semantics. The target-state architecture and safety rules in [`AGENTS.md`](AGENTS.md) are intentionally comprehensive, but **you do not need to satisfy every maintainer-only check before opening a pull request**.

## 1. Required before opening a PR

Please ensure that:

- **The change builds** using the project’s standard local configuration.
- **Relevant tests pass** locally.
- **The PR has a clear description** of the problem and proposed change.
- **New unsafe code is justified** with clear safety invariants.
- **Licensing and provenance are compatible** with Apache-2.0. Disclose any external implementation, paper, source, or AI-assisted provenance relevant to the patch.

## 2. Helpful, but not required

These steps make review faster when practical:

- Run `cargo fmt` and `cargo clippy`.
- Include a minimized regression test for bug fixes.
- Add or extend a property test or fuzz corpus entry where appropriate.
- Include `divan` benchmark results for changes to performance-critical paths.

A minimized unit regression may be converted into or supplemented by a property test or fuzz corpus entry during review.

## 3. Handled by maintainers and CI

You do **not** need to run these before opening a PR:

- Cross-architecture verification through `tools/check_all_archs.sh`.
- Extended Miri runs over portable and fallback paths.
- The full project lint policy, including pedantic, nursery, restriction, and performance lints.
- Import-boundary validation through `python3 -B tools/import_audit.py`.
- Structural-policy validation through `python3 -B tools/structure_audit.py`.
- Final benchmark calibration and architecture-specific validation.

The policy audits run directly in CI and must report no findings before merge.

## Rough fixes and draft PRs are welcome

If you found a bug or have a working idea but do not want to navigate the full internal architecture, open an issue or draft PR.

A minimized reproducer and a description of the likely cause are already valuable. I can help adapt the change to the project structure and complete the final verification.

Authorship will be preserved where practical by retaining commits, collaborating on the original branch, or providing clear credit in the final PR. `Co-authored-by` trailers will be used when they accurately reflect shared authorship.

## Target-state and autonomous-agent rules

For internal architecture, invariants, and autonomous-agent guidance, see [`AGENTS.md`](AGENTS.md).
