# arbi-anafis

**High-performance arbitrary-precision arithmetic in pure Rust, with inline storage and configurable precision semantics.**

`arbi-anafis` is an arbitrary-precision integer library for Rust. It provides
`ArbiUint` and `ArbiInt` with inline small-value storage, configurable precision
semantics, architecture-aware arithmetic kernels, and portable 16-, 32-, and
64-bit limb support.

The exported numeric tower contains `ArbiUint` and `ArbiInt`. Rational and
floating-point specifications live under `docs/` as design work and are not
part of the library API.

---

### Core Ergonomics & Precision Operations

```rust
use arbi_anafis::{ArbiUint, ArbiInt, BoundedPrecision, Precision};

// 1. Unlimited dynamic precision with standard Rust operators
let value = ArbiUint::from(1_u8) << 256_usize;
assert_eq!(value.significant_bits(), 257);

// 2. Bounded precision participating in subsequent operations
let precision = BoundedPrecision::new(16).expect("valid bit width");
let x = ArbiUint::with_precision_wrapping(65_535_u32, precision);
let y = x.wrapping_add(&ArbiUint::with_precision_wrapping(2_u32, precision));
assert_eq!(y, ArbiUint::from(1_u8));

// 3. String radix parsing and number theory
let dec = ArbiUint::from_str_radix("123456789012345678901234567890", 10).unwrap();
let fact = ArbiUint::factorial(100, Precision::Unlimited);
assert_eq!(fact.significant_bits(), 525);

// 4. Signed arithmetic (`ArbiInt`) with two's-complement bitwise operations
let neg_val = ArbiInt::from(-123456789_i64);
assert!(neg_val.is_negative());
```

---

## Why `arbi-anafis`?

### Inline Small-Value Storage (`INLINE_LIMBS`)
Values up to `INLINE_LIMBS` limbs (`4` limbs, providing 256 bits of capacity on 64-bit architectures, 128 bits on 32-bit, and 64 bits on 16-bit) are stored inline. This avoids heap allocation (`malloc`/`free`) for construction, movement, and operations whose results and required scratch space remain within the inline or stack-backed capacity. This includes many common 128-bit and 256-bit integer workloads, depending on the target limb width and operation.

### Configurable Precision Semantics
Arithmetic separates existing value metadata from newly-constructed target precision without manual precision-threading boilerplate:
- **Unlimited (`Precision::Unlimited`)**: Results grow dynamically without bound. The default for general mathematical computation.
- **Bounded (`BoundedPrecision`)**: Fixed bit-width arithmetic (`1..usize::MAX`) with explicit overflow handling (`wrapping`, `saturating`, `strict`, or `checked`), mimicking native machine primitives across arbitrary bit widths. Binary operations produce `max(width_a, width_b)`, while assignment (`+=`, `-=`) strictly preserves the left-hand side precision.
- **Ambient (`AmbientPrecision`)**: Inherits precision policies from thread-local or global execution contexts (`PrecisionContext::set_global`). For infallible conversions (`From<T>`), ambient precision acts strictly as a target floor. Results widen automatically (`max(target_bits, required_bits)`), so construction is exact and lossless.

### Hierarchical Multiplication & Division Towers
`arbi-anafis` structures algorithms across magnitude thresholds (`thresholds.rs`, dynamically calibrated via `arbi-tune`):
- **Production multiplication**: Schoolbook $\rightarrow$ Karatsuba $\rightarrow$ Toom-Cook (3, 4, 6, and 8.5) $\rightarrow$ Schönhage–Strassen/Fermat FFT, subject to the generated profile and operand-shape gates.
- **Production division**: Algorithm D (Knuth) $\rightarrow$ Burnikel–Ziegler $\rightarrow$ Newton–Raphson. Division kernels preserve the normalization and correction bounds required by their respective algorithms.
- **Modular reduction**: Barrett, Montgomery, and combined Barrett–Montgomery paths.
- **Disabled NTT backend**: The multi-prime NTT/CRT implementation remains registered for development and verification, but `NTT_THRESHOLD = 0` keeps it out of production dispatch.

### Specialized Architecture Backends
`src/int/logic/unsigned/math/arch/` provides portable fallbacks and specialized assembly or ISA-aware implementations across **x86/x86-64** (with runtime CPUID dispatch selecting ADX/BMI2 or baseline), **`AArch64`**, **ARM**, **RISC-V (32/64)**, **POWER**, **MIPS (32/64)**, **s390x**, and **`LoongArch` (32/64)**.

---

## Performance and tuning

Performance depends on operand width and shape, target ISA, runtime feature
detection, and the active tuning profile. The benchmark targets under
`benches/` compare public APIs, individual tiers, crossover regions, operand
shapes, and architecture kernels. Benchmark reports identify the checkout,
hardware, operands, and active profile used for each result.

`arbi-tune` produces a complete machine-local profile for multiplication,
squaring, division, and SSA dispatch. Its measurement protocol, constant
policy, and invocation are documented in
[`tools/tune/README.md`](tools/tune/README.md). The architecture kernel matrix
is documented in [`docs/int/kernel-matrix.md`](docs/int/kernel-matrix.md).

---

## API & Feature Surface

### Native-Like Arithmetic & Bitwise Families
- **Shared Arithmetic Families**: `checked_*`, `wrapping_*`, `saturating_*`, `overflowing_*`, `strict_*`, Euclidean division (`div_euclid`, `rem_euclid`), and rounding divisions (`div_trunc`, `div_floor`, `div_ceil`).
- **Bitwise Inspection & Modification**: `leading_zeros`, `trailing_zeros`, `count_ones`, `get_bit`, `set_bit`, `toggle_bit`, `rotate_left`/`rotate_right`, bit range extraction (`bit_range`), and endian serialization (`to_le_bytes`, `from_be_bytes`, etc.).
- **Signed-Specific (`ArbiInt`)**: `abs`, `signum`, `is_negative`, `is_minus_one`, and two's-complement boundary semantics.
- **Unsigned-Specific (`ArbiUint`)**: Floor square root (`isqrt`), exact division towers, and unsigned magnitude arithmetic.

### Number Theory & Primality
- **Algebraic Utilities**: `gcd`, `gcd_lcm`, `lcm`, and `invert` (modular inverse). `extended_gcd` computes Bézout identity cofactors, cleanly returning signed structures when operating on unsigned magnitudes.
- **Probabilistic Primality Testing**: `is_probably_prime(k)` uses deterministic Miller–Rabin rounds with fixed prime schedules and deterministic base sets for bounded input ranges, including complete coverage for values up to 64 bits. `is_probably_prime_with_rng` allows caller-provided randomness for applications that should not rely on a fixed public base schedule. Also includes `Baillie–PSW` primality checking and `next_prime`.
- **Ecosystem Integration**:
  - **Standard Trait Compatibility**: Complete `core::ops` (providing all four ownership combinations), `core::iter::Sum` and `Product` (folding from `Zero::zero()`), and `num-traits` (`Zero`, `One`, `Num`, `Unsigned`, `Signed`).
  - **Planned Serialization & Randomness**: `serde` and `rand` integrations are
    roadmap items (tracked in `docs/int/api-inventory.md`), not yet implemented.
    When `serde` is implemented, it targets a canonical limb-width-independent
    wire representation (`[u64]` slices), consistent across 16-bit, 32-bit, and
    64-bit platforms.

---

## Rigorous Verification & Safety

- **Strict Crate-Wide Safety**: `unsafe_code` is denied crate-wide by default via `Cargo.toml` (`[lints.rust] unsafe_code = "deny"`). It is allowed selectively inside performance-critical mathematical loops, conversion/bitwise kernels, internal storage/scratch buffers, and architecture-specific modules (`src/int/logic/unsigned/math/arch/`). Each unsafe operation requires a documented safety invariant (`// SAFETY: ...`) and a justified clippy `reason = "..."`.
- **Cross-Architecture Verification Matrix**: Representative target configurations spanning 16-bit, 32-bit, and 64-bit limb widths, endianness, and unusual pointer ABIs are continuously verified via `tools/check_all_archs.sh`.
- **Property-Based and Memory-Model Testing**: Algebraic identities, edge cases, and serialization round trips are exercised through `proptest` and regression corpora. Portable unsafe paths and fallback implementations are validated with Miri, while architecture-specific kernels are covered through cross-compilation, target-specific checks, and differential testing.

---

## Status and limitations

`arbi-anafis` is functional, extensively tested, and under active development towards stable v1.0:
- **Experimental API Surface**: Public trait structures and methods may undergo refinement prior to the 1.0 release.
- **Machine-Specific Crossovers**: Portable architecture profiles provide defaults, while `arbi-tune` derives machine-local thresholds for workloads that require tighter dispatch calibration.
- **Incomplete NTT Tier**: NTT remains disabled in production dispatch. SSA is the production transform backend when its profile threshold and operand-shape requirements admit it.
- **Not Designed for Constant-Time Execution**: `arbi-anafis` is designed for general-purpose multi-precision arithmetic and is **not** built for constant-time execution. Do not use it for secret-dependent cryptographic operations without auditing target execution paths.

---

## Development & Contribution

We welcome contributions of all skill levels! Please see our human-friendly [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to open PRs, run local tests, or submit bug reports. For deep internal architecture specifications and target-state rules used by autonomous agents, see [`AGENTS.md`](AGENTS.md) and [`docs/`](docs/).

```bash
# Run core test suite
cargo test --lib --no-default-features
cargo test --lib --all-features

# Run Clippy checks
cargo clippy --lib --all-features

# Build local API documentation
cargo doc --lib --all-features --open
```

## Citation

If you use `arbi-anafis` in academic work, please cite:

```bibtex
@software{arbianafis,
  author       = {Martins, Pedro},
  orcid        = {0009-0001-8170-2930},
  title        = {arbi-anafis: High-performance arbitrary-precision arithmetic in pure Rust},
  year         = {2026},
  url          = {https://github.com/CokieMiner/arbi-anafis},
  version      = {0.1.0}
}
```
Contributors who make substantial contributions to the project and would like
academic credit may request inclusion in future citation metadata. Attribution
is limited to the project's principal contributors.

---

## License

`arbi-anafis` is licensed under the Apache License, Version 2.0.
See the [LICENSE](https://github.com/CokieMiner/arbi-anafis/blob/master/LICENSE)
file for the full license text.
