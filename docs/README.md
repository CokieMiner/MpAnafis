# Project documentation

This directory contains design specifications, API inventories, implementation
references, measured performance records, and processor manuals.

## Layout

- [organization.md](organization.md): current repository map, source boundaries,
  tests, benchmarks, tooling, and result placement.
- [int/spec.md](int/spec.md): integer design contracts, with implemented behavior
  distinguished from planned requirements.
- [int/api-inventory.md](int/api-inventory.md): present public integer surface.
- [int/kernel-matrix.md](int/kernel-matrix.md): architecture dispatch and kernels.
- [int/complex-algorithms.md](int/complex-algorithms.md): mathematical algorithm notes.
- [int/benchmarks/](int/benchmarks/README.md): reviewed time-versus-size curves
  with descriptions and measurement details; full measurement data stays local.
- [float/spec.md](float/spec.md): floating-point design specification.
- [rational/spec.md](rational/spec.md): rational-number design specification.
- [manuals/](manuals/README.md): locally stored processor references.

Rust source and Rustdoc remain in source directories. Executable benchmarks
remain under `benches/`; working data belongs under ignored `target/`.
Documentation records are exported through the
[benchmark driver](../tools/benchmark/README.md).
