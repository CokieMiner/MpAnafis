# MpAnafis

`MpAnafis` provides high-performance arbitrary-precision unsigned (`MpUint`) and
signed (`MpInt`) integers in pure Rust.

Designed for speed across all numeric scales:
- **Zero-allocation inline storage (<256 bits on 64-bit systems)** keeps up to 4 native limbs inline on the stack.
- **Deep multiplication & division towers** dispatching from fine-grained unrolled assembly loops up to Schönhage–Strassen FFT convolution.
- **Sub-quadratic GCD and division** powered by Half-GCD (HGCD), Lehmer acceleration, and Burnikel–Ziegler divide-and-conquer.
- **Modular arithmetic & number theory** featuring Montgomery CIOS reduction, Barrett reduction, Baillie–PSW primality, and Bézout extended GCD.
- **Zero dependencies by default**, targeting `no_std` environments (requiring only `alloc`).

## Quick Start

Add `mp_anafis` to your `Cargo.toml`:

```toml
[dependencies]
mp_anafis = "0.1.0"
```

### Basic Arithmetic & Number Theory

```rust
use mp_anafis::{MpInt, MpUint, Precision};

// Unsigned arbitrary-precision integers
let a = MpUint::from(100_u64);
let b = MpUint::one() << 128_usize;
let sum = &a + &b;
let product = &a * &MpUint::from(200_u32);

// Parsing and formatting across radices
let hex_val = MpUint::from_str_radix("deadbeefcafebabe12345678", 16).unwrap();
assert_eq!(hex_val.to_str_radix(16), "deadbeefcafebabe12345678");

// Signed integers with standard operations
let neg = MpInt::from(-42_i32);
assert_eq!(neg.abs(), MpUint::from(42_u32));

// Number theory & combinatorics
let fact = MpUint::factorial(100, Precision::Unlimited);
assert_eq!(fact.significant_bits(), 525);

// Exact primality testing & roots
assert!(MpUint::from(1_000_000_007_u64).is_prime());
let root = fact.isqrt().unwrap();
```

## Architectural Highlights

### 1. Hybrid Inline/Heap Storage

Each integer embeds up to 4 native `usize` limbs directly in its struct:
- **256 bits** on 64-bit platforms
- **128 bits** on 32-bit platforms
- **64 bits** on 16-bit platforms

Operations fitting within this threshold perform direct register/stack arithmetic with **zero heap allocations**. Values seamlessly spill to heap-backed storage when exceeding inline capacity.

### 2. Comprehensive Multiplication Ladder

Multiplication and squaring dynamically choose optimal algorithms based on operand size, symmetry, and machine crossover thresholds:

- **Basecase / Schoolbook:** Special unrolled 2×2, 3×3, 4×4, and 8×8 loops, plus SIMD/assembly-accelerated paired-row multiply-accumulate (`mul_2_limbs`, `add_mul_2_limbs`).
- **Karatsuba:** Classical recursive 2-way split for small-to-medium operands.
- **Toom-Cook Family:**
  - **Balanced splits:** Toom-3, Toom-4, Toom-6, and Toom-8.5.
  - **Unbalanced / Asymmetric splits:** Toom-3.2 (3-by-2) and Toom-4.3 (4-by-3) for operands of differing lengths.
  - **Mulders Short Multiplication:** Truncated recursive low/high products using power-law split ratios.
  - **Lopsided Multiplication:** Blocked parallel decomposition for operands with vastly asymmetric limb counts.
- **Schönhage–Strassen (SSA):** Fast Fourier Transform (FFT) convolution over Fermat rings $\mathbb{Z}/(2^{2^n} + 1)\mathbb{Z}$, with negacyclic odd-factor decomposition, split-CRT, and cache-aligned coefficient matrix butterflies for giant integers.

### 3. Advanced Division & Roots

- **Schoolbook Division:** Knuth Algorithm D with hardware-assisted double-limb quotient primitives (`divq`, `divl`, `dlgr`).
- **Burnikel–Ziegler (Divide & Conquer):** Recursive 2-by-1 and 3-by-2 block division matching sub-quadratic multiplication speed.
- **Newton–Raphson Inversion:** High-precision reciprocal approximations for massive quotients.
- **Square Roots & $n$-th Roots:** Karatsuba-style recursive `sqrt_rem` and fast Hensel/Newton $n$-th integer root extractions (`nth_root`).

### 4. GCD & Modular Reduction

- **Binary & Lehmer GCD:** Accelerated matrix-step quotient extraction for medium operands.
- **Half-GCD (HGCD):** Sub-quadratic GCD and extended GCD using $2 \times 2$ polynomial transition matrices with thread-local scratch reuse.
- **Bézout Extended GCD:** Exact cofactors and modular inverses via `extended_gcd()` and `invert()`.
- **Modular Engines:** Inlined Montgomery CIOS multiplication (`montgomery_mul`) and Barrett reduction (`barrett_reduce`).

### 5. Multi-Architecture Hardware Acceleration

Platform-specialized kernels selected at compile-time or via runtime CPU detection:
- **x86-64:** Dual-carry chains with ADX (`adcx`, `adox`), BMI2 (`mulx`), and AVX2/AVX-512 vector shifts.
- **`AArch64` & ARM:** Wide multiply-accumulate (`umaal`, `umull`) and vector pipelines.
- **POWER, s390x, RISC-V & `LoongArch`:** Specialized assembly and native double-limb division routines.
- **Portable Fallback:** Standard Rust implementations for any 16-, 32-, or 64-bit target, fully verifiable under Miri.

### 6. Configurable Precision & Bounded Semantics (Optional)

While values default to `Precision::Unlimited`, applications emulating hardware registers or cryptographic curves can enforce explicit bit-widths using `BoundedPrecision`:

```rust
use mp_anafis::{BoundedPrecision, MpUint, Precision};

let width = BoundedPrecision::new(8).unwrap(); // 8-bit bound
let a = MpUint::with_precision_checked(200_u16, width).unwrap();
let b = MpUint::with_precision_checked(20_u16, width).unwrap();

let mut result = MpUint::zero_with_precision(width);
result.assign_add(&a, &b); // 220
result += &b;              // 240

// Explicit bounded overflow semantics matching Rust primitives:
assert!(result.checked_add(&b).is_none());              // None on 8-bit overflow
assert_eq!(result.wrapping_add(&b).to_u64(), Some(4)); // 260 mod 256 = 4
```

## Cargo Features

All features are optional and disabled by default:

| Feature | Description |
| --- | --- |
| `std` | Integrates with `std::error::Error` and scoped thread-local precision contexts (`PrecisionContext`). |
| `num-traits` | Implements `num_traits` ecosystem traits (`Zero`, `One`, `Num`, `Signed`, `Unsigned`, etc.). |
| `rayon` | Enables multi-threaded parallel execution for eligible large operations via Rayon's work-stealing pool. |
| `_internal-tune` | Exposes internal algorithmic dispatch thresholds for benchmarks and the `mp-tune` harness. |

## Documentation & Architecture

- [API Inventory](docs/int/api-inventory.md): Signatures and contracts for all public APIs.
- [Integer Specification](docs/int/spec.md): Architectural design and mathematical invariants.
- [Kernel Matrix](docs/int/kernel-matrix.md): Architecture-specific backends and instruction set extensions.
- [Repository Organization](docs/organization.md): Project layout and subsystem boundaries.
- [Project Guidelines](AGENTS.md): Coding standards, invariant proofs, and safety contracts.
- [Tuning Guide](tools/tune/README.md): Machine-local crossover benchmark harness.

## Development & Verification

To run tests and audits locally:

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
