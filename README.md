# MpAnafis

`MpAnafis` provides arbitrary-precision unsigned (`MpUint`) and signed (`MpInt`)
integers in Rust, with four inline limbs and configurable precision. The Cargo
crate is `mp_anafis`; its declared minimum Rust version is 1.93.0.

The default feature set is empty: the library uses `core` and `alloc`, without
external runtime dependencies. A `no_std` application still needs an allocator
for values and scratch buffers that exceed inline storage. Rational and
floating-point types are design specifications, not exported APIs.

## Usage

```rust
use mp_anafis::{BoundedPrecision, MpInt, MpUint, Precision};

let large = MpUint::one() << 256_usize;
assert_eq!(large.significant_bits(), 257);
assert_eq!(large.precision(), Precision::Unlimited);

let width = BoundedPrecision::new(8).unwrap();
let a = MpUint::with_precision_checked(200_u16, width).unwrap();
let b = MpUint::with_precision_checked(20_u16, width).unwrap();
let mut result = MpUint::zero_with_precision(width);
result.assign_add(&a, &b);
result += &b;
assert_eq!(result.to_u64(), Some(240));
assert_eq!(result.precision(), Precision::Bounded(width));
assert!(result.checked_add(&b).is_none());
assert_eq!(result.wrapping_add(&b).to_u64(), Some(4));

let negative = MpInt::from(-42_i32);
assert!(negative.is_negative());
assert_eq!(MpUint::factorial(100, Precision::Unlimited).significant_bits(), 525);
```

## Precision

Each value carries `Precision::Unlimited` or a validated `BoundedPrecision`.
`precision()` returns that metadata. Valid bounded widths are `1..usize::MAX`;
unsigned values use an unsigned range and signed values use a two's-complement
range at that width.

Non-assigning binary arithmetic combines two bounded widths using their maximum.
An unlimited operand makes the result unlimited. Assignment operators and
arithmetic `assign_*` methods preserve the destination's precision; bounded
failures preserve its value. Ordinary bounded arithmetic panics on overflow in
both debug and release builds. Checked, wrapping, saturating, overflowing, and
strict methods expose explicit arithmetic policies.

`AmbientPrecision` controls construction through `PrecisionContext`. For `From`
conversions, an ambient bound is a floor: the result widens as needed to preserve
the value. Parsing uses an ambient bound as a cap. Scoped contexts require `std`;
the global default requires pointer-width atomics. `zero()`, `one()`, and
`Default` use unlimited precision. Iterator `Sum` starts from unlimited zero;
`Product` starts from unlimited one, so both return unlimited results.

Some policies differ from Rust primitives. Left-shift policies detect overflow
of the mathematical result and do not mask counts modulo the width. Wrapping
and saturating division/remainder return zero for a zero divisor; overflowing
variants return `(zero, true)`. The [API inventory](docs/int/api-inventory.md)
describes current contracts and differences in detail.

## Features

All features are disabled by default.

| Feature | Effect |
| --- | --- |
| `std` | Standard error integration and scoped ambient precision contexts. |
| `num-traits` | Implementations of `Zero`, `One`, `Num`, `Signed`, and `Unsigned`. |
| `rayon` | Enables `std` and parallel arithmetic through the active or global Rayon pool. |
| `_internal-tune` | Enables `std` and exposes hidden algorithm runners for benchmarks and `mp-tune`. |

### Default pool sizing

With `rayon`, eligible large arithmetic operations use Rayon's active pool.
Applications can supply a pool with `pool.install(|| &a * &b)` or configure the
global pool using `RAYON_NUM_THREADS`. The library does not create a private pool.

When execution first resolves the global pool, the library attempts to narrow
its width to the detected physical-core count if that is below available
parallelism. Physical-core detection currently uses Linux sysfs. This attempt
is skipped when `RAYON_NUM_THREADS` is set or the call is already inside a Rayon
worker. An existing global pool is never resized. Unknown topology leaves the
width to Rayon. Scratch-size queries that resolve parallel execution can also
initialize the global pool.

A custom pool or an explicitly configured global pool therefore controls the
worker budget used by the arithmetic dispatcher. Small operations remain
sequential when their dispatch thresholds do not admit parallel work.

## Arithmetic and storage

Values store up to four native `usize` limbs inline: 256 bits on 64-bit targets,
128 on 32-bit targets, and 64 on 16-bit targets. Larger magnitudes use heap
storage. Inline value storage alone does not imply that every operation or
conversion avoids allocation.

Multiplication dispatch includes basecase, Karatsuba, Toom-Cook, and
Schönhage–Strassen algorithms. Division includes Knuth, Burnikel–Ziegler, and
Newton algorithms. Operand shape, target support, and tuning parameters select
the applicable path. Architecture kernels provide specialized implementations
with portable fallbacks; the [kernel matrix](docs/int/kernel-matrix.md) lists
backends and their contracts. [The tuner](tools/tune/README.md) measures
machine-local crossover settings.

The public API includes arithmetic operators, bit inspection and manipulation,
radix parsing and formatting, byte conversion, roots, GCD/LCM, and modular
arithmetic. Signed byte output uses minimal two's-complement encoding; unsigned
output uses minimal magnitude encoding. Precision metadata is not serialized.
Signed `extended_gcd` returns signed Bézout coefficients; unsigned
`extended_gcd` returns modular coefficient residues.

`is_prime()` is exact for values through `u64::MAX`; larger values use
Baillie–PSW probable-prime testing. `is_probably_prime(k)` uses the exact test for
`u64` values and a fixed prefix of 1–64 prime Miller–Rabin bases for larger
values. `next_prime()` includes its input and uses the same probable-prime
qualification for large candidates. These methods do not provide primality
certificates for arbitrary-size inputs.

Arithmetic is variable-time and is not designed for secret-dependent
cryptographic operations.

## Documentation and development

- [API inventory](docs/int/api-inventory.md): current signatures and contracts.
- [Integer specification](docs/int/spec.md): crate design, including explicitly
  labeled planned behavior.
- [Repository organization](docs/organization.md): source and tool boundaries.
- [Contributing](CONTRIBUTING.md): setup, checks, and pull request requirements.
- [Project guidelines](AGENTS.md): invariants, safety, and module rules.
- [Benchmark driver](tools/benchmark/README.md): selected runs, size sweeps,
  reports, and figure export.

The [CI workflow](.github/workflows/ci.yml) defines formatting, lint, policy,
native/backend, cross-target, and Miri checks. Cross-compilation checks code
generation; runtime tests exercise selected native or emulated targets. Miri
checks portable paths rather than executing architecture assembly.

From a checkout, the basic library checks are:

```sh
cargo check --lib
cargo clippy --lib
cargo test --lib
python3 tools/structure_audit.py
python3 tools/import_audit.py
```

## Citation

Academic citation:

```bibtex
@software{mpanafis,
  author       = {Martins, Pedro},
  orcid        = {0009-0001-8170-2930},
  title        = {MpAnafis: High-performance arbitrary-precision arithmetic in pure Rust},
  year         = {2026},
  url          = {https://github.com/CokieMiner/MpAnafis},
  version      = {0.1.0}
}
```
Contributors who make substantial contributions to the project and would like
academic credit may request inclusion in future citation metadata. Attribution
is limited to the project's principal contributors.

## License

`MpAnafis` is licensed under the Apache License, Version 2.0.
See [LICENSE](https://github.com/CokieMiner/MpAnafis/blob/master/LICENSE) for the full text.
