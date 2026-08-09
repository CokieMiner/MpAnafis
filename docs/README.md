# Project documentation

This directory contains design notes, implementation plans, API inventories,
performance records, and external architecture references.

## Layout

- `organization.md`: source organization, module boundaries, visibility rules, directory layout (`src/int/`, `benches/`), and review checklist.
- `int/`: core multi-precision integer architecture documentation (`spec.md` normative specification, `api-inventory.md` public surface, `kernel-matrix.md` ISA dispatch and assembly matrix).
- `float/`: planned floating-point specification and IEEE 754 alignment notes (`spec.md`).
- `rational/`: planned rational-number specification and fraction behavior (`spec.md`).
- `manuals/`: locally stored processor manuals used when reviewing
  architecture-specific kernels.

Source directories contain Rust code and its Rustdoc; planning documents live
here so they do not look like compilation units.
