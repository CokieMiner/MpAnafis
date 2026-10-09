# MpInt / MpUint: Public API Inventory

This document inventories the public API implemented in `src/int/api/` for
`MpUint` and `MpInt`, plus the metadata and error types exported by `src/lib.rs`.
[spec.md](spec.md) defines the crate design, including future requirements.
This inventory describes only the current implementation; Section 5 records
behavioral details and differences from primitive integers.

---

## 1. Scope, Features, and Shared Types

The integer types expose 159 unsigned and 169 signed public inherent methods.
Trait-provided methods are listed in Section 4. Counts exclude blanket trait
implementations and the explicitly unstable `_internal-tune` surface.

| Configuration | Public availability |
| --- | --- |
| Default features (`[]`) | Integer API using `core` and `alloc`; a heap allocator is required for heap-backed values. |
| `std` | Scoped precision closures and `Error` implementations for public error types. |
| `num-traits` | Numeric trait implementations; does not require `std`. |
| `rayon` | Enables `std` and parallel execution; no additional integer inherent methods. |
| `_internal-tune` | Enables `std`; exposes hidden tuning types and the `mp-tune` binary, outside this stable inventory. |
| `target_has_atomic = "ptr"` | Enables `PrecisionContext::set_global`, independently of `std`. |

Public supporting types are `BoundedPrecision`, `Precision`, `AmbientPrecision`,
`PrecisionContext`, `DebugVerbose<'data, T>`, `MpError`, `ParseMpUintError`,
`ParseMpUintErrorKind`, `ParseMpIntError`, and `ParseMpIntErrorKind`.
`Precision` has `Unlimited` and `Bounded(BoundedPrecision)`;
`AmbientPrecision` additionally has `Unset`. Both enums and `MpError` are
non-exhaustive. Parse-error structs have private fields and expose their cause
through `pub const fn kind(&self) -> &ParseMpIntErrorKind` or
`pub const fn kind(&self) -> &ParseMpUintErrorKind`. Both kind enums are
non-exhaustive and reexported at the crate root. Associated parse-error
constructors are crate-private. Integer representation and precision fields are
private to `src/int/api/types.rs` and its implementation descendants under
`types/`. Public construction and `precision()` provide the corresponding API
boundaries. `DebugVerbose` exposes its wrapped reference as tuple field `.0`.

### Precision Metadata and Context

- `pub const fn BoundedPrecision::new(bits: usize) -> Option<BoundedPrecision>`
- `pub const fn BoundedPrecision::get(self) -> usize`
- `pub const fn Precision::new_bounded(bits: usize) -> Option<Precision>`
- `pub const fn Precision::is_unlimited(self) -> bool`
- `pub const fn Precision::significant_bits(self) -> Option<usize>`
- `pub const fn AmbientPrecision::new_bounded(bits: usize) -> Option<AmbientPrecision>`
- `pub fn PrecisionContext::active() -> AmbientPrecision`
- `pub fn PrecisionContext::set_global(precision: AmbientPrecision) -> AmbientPrecision` *(pointer-width atomics only)*
- `pub fn PrecisionContext::with_bounded<F: FnOnce() -> R, R>(bits: usize, f: F) -> R` *(`std` only)*
- `pub fn PrecisionContext::with_unlimited<F: FnOnce() -> R, R>(f: F) -> R` *(`std` only)*

Bounded widths use the canonical range `1..usize::MAX`; zero and the top
`usize` value are rejected because the ambient encoding reserves them.
`PrecisionContext::active` is `const fn` only without both `std` and
pointer-width atomics; it returns `Unset` in that configuration. No
`AmbientPrecision::active` exists. Both integer types expose their metadata
through `precision()`.

---

## 2. Exact Implemented API: MpUint (Unsigned Native-Width Limbs)

The following are public inherent methods. For `new<T>` and all three
`with_precision_*<T>` constructors, the bound is `Self: From<T>`.

### Constructors & Capacity

- `pub fn zero() -> Self`
- `pub const fn zero_with_precision(bits: BoundedPrecision) -> Self`
- `pub fn one() -> Self`
- `pub fn new<T>(value: T) -> Self` *(where `Self: From<T>`)*
- `pub fn with_capacity(capacity: usize) -> Self`
- `pub fn with_precision_checked<T>(value: T, bits: BoundedPrecision) -> Result<Self, MpError>`
- `pub fn with_precision_wrapping<T>(value: T, bits: BoundedPrecision) -> Self`
- `pub fn with_precision_saturating<T>(value: T, bits: BoundedPrecision) -> Self`
- `pub fn max_for_precision(bits: usize) -> Self`
- `pub fn min_for_precision(bits: usize) -> Self`
- `pub fn reserve(&mut self, additional: usize)`
- `pub fn reserve_exact(&mut self, additional: usize)`
- `pub fn shrink_to_fit(&mut self)`
- `pub const fn capacity(&self) -> usize`
- `pub const fn swap(&mut self, other: &mut Self)`
- `pub const fn as_debug_verbose(&self) -> DebugVerbose<'_, Self>`

> **Note:** `Precision::new_bounded` is the public validated precision constructor. The shared internal ambient-construction resolver is used by `From<T>` constructors and is not part of the public inherent API.

### Core Arithmetic Families
- `pub fn checked_add(&self, rhs: &Self) -> Option<Self>`
- `pub fn checked_sub(&self, rhs: &Self) -> Option<Self>`
- `pub fn checked_mul(&self, rhs: &Self) -> Option<Self>`
- `pub fn checked_div(&self, rhs: &Self) -> Option<Self>`
- `pub fn checked_rem(&self, rhs: &Self) -> Option<Self>`
- `pub fn wrapping_add(&self, rhs: &Self) -> Self`
- `pub fn wrapping_sub(&self, rhs: &Self) -> Self`
- `pub fn wrapping_mul(&self, rhs: &Self) -> Self`
- `pub fn wrapping_div(&self, rhs: &Self) -> Self`
- `pub fn wrapping_rem(&self, rhs: &Self) -> Self`
- `pub fn overflowing_add(&self, rhs: &Self) -> (Self, bool)`
- `pub fn overflowing_sub(&self, rhs: &Self) -> (Self, bool)`
- `pub fn overflowing_mul(&self, rhs: &Self) -> (Self, bool)`
- `pub fn overflowing_div(&self, rhs: &Self) -> (Self, bool)`
- `pub fn overflowing_rem(&self, rhs: &Self) -> (Self, bool)`
- `pub fn saturating_add(&self, rhs: &Self) -> Self`
- `pub fn saturating_sub(&self, rhs: &Self) -> Self`
- `pub fn abs_diff(&self, other: &Self) -> Self`
- `pub fn saturating_mul(&self, rhs: &Self) -> Self`
- `pub fn saturating_div(&self, rhs: &Self) -> Self`
- `pub fn saturating_rem(&self, rhs: &Self) -> Self`
- `pub fn try_add(&self, rhs: &Self) -> Result<Self, MpError>`
- `pub fn try_sub(&self, rhs: &Self) -> Result<Self, MpError>`
- `pub fn try_mul(&self, rhs: &Self) -> Result<Self, MpError>`
- `pub fn try_div(&self, rhs: &Self) -> Result<Self, MpError>`
- `pub fn try_rem(&self, rhs: &Self) -> Result<Self, MpError>`
- `pub fn strict_add(&self, rhs: &Self) -> Self`
- `pub fn strict_sub(&self, rhs: &Self) -> Self`
- `pub fn strict_mul(&self, rhs: &Self) -> Self`
- `pub fn strict_div(&self, rhs: &Self) -> Self`
- `pub fn strict_rem(&self, rhs: &Self) -> Self`
- `pub fn assign_add(&mut self, a: &Self, b: &Self)`
- `pub fn assign_sub(&mut self, a: &Self, b: &Self) -> bool`
- `pub fn assign_mul(&mut self, a: &Self, b: &Self)`
- `pub fn assign_square(&mut self, a: &Self)`
- `pub fn mul_2exp(&self, shift: usize) -> Self`
- `pub fn div_2exp(&self, shift: usize) -> Self`
- `pub fn widening_mul(&self, other: &Self) -> (Self, Self)`
- `pub fn try_widening_mul(&self, other: &Self) -> Result<(Self, Self), MpError>`
- `pub fn carrying_mul(&self, other: &Self, carry: &Self) -> (Self, Self)`
- `pub fn try_carrying_mul(&self, other: &Self, carry: &Self) -> Result<(Self, Self), MpError>`
- `pub fn carrying_mul_add(&self, other: &Self, carry1: &Self, carry2: &Self) -> (Self, Self)`
- `pub fn mul_add(&self, a: &Self, b: &Self) -> Self`
- `pub fn midpoint(&self, other: &Self) -> Self`
- `pub fn is_divisible_by(&self, other: &Self) -> bool`
- `pub fn is_divisor_of(&self, other: &Self) -> bool`
- `pub fn div_rem(&self, rhs: &Self) -> Option<(Self, Self)>`
- `pub fn div_trunc(&self, rhs: &Self) -> Self`
- `pub fn checked_div_trunc(&self, rhs: &Self) -> Option<Self>`
- `pub fn rem_trunc(&self, rhs: &Self) -> Self`
- `pub fn checked_rem_trunc(&self, rhs: &Self) -> Option<Self>`
- `pub fn div_rem_euclid(&self, rhs: &Self) -> Option<(Self, Self)>`
- `pub fn div_euclid(&self, rhs: &Self) -> Self`
- `pub fn checked_div_euclid(&self, rhs: &Self) -> Option<Self>`
- `pub fn rem_euclid(&self, rhs: &Self) -> Self`
- `pub fn checked_rem_euclid(&self, rhs: &Self) -> Option<Self>`
- `pub fn div_rem_floor(&self, rhs: &Self) -> Option<(Self, Self)>`
- `pub fn div_floor(&self, rhs: &Self) -> Self`
- `pub fn checked_div_floor(&self, rhs: &Self) -> Option<Self>`
- `pub fn mod_floor(&self, rhs: &Self) -> Self`
- `pub fn checked_mod_floor(&self, rhs: &Self) -> Option<Self>`
- `pub fn div_ceil(&self, rhs: &Self) -> Self`
- `pub fn checked_div_ceil(&self, rhs: &Self) -> Option<Self>`
- `pub fn pow(&self, exp: u32) -> Self`
- `pub fn checked_pow(&self, exp: u32) -> Option<Self>`
- `pub fn try_pow(&self, exp: u32) -> Result<Self, MpError>`
- `pub fn square(&self) -> Self`

### Bitwise Operations & Shifts
- `pub fn checked_shl(&self, shift: usize) -> Option<Self>`
- `pub fn wrapping_shl(&self, shift: usize) -> Self`
- `pub fn overflowing_shl(&self, shift: usize) -> (Self, bool)`
- `pub fn saturating_shl(&self, shift: usize) -> Self`
- `pub fn try_shl(&self, shift: usize) -> Result<Self, MpError>`
- `pub fn rotate_left(&self, n: u32, width: usize) -> Option<Self>`
- `pub fn rotate_right(&self, n: u32, width: usize) -> Option<Self>`
- `pub fn reverse_bits(&self, width: usize) -> Option<Self>`
- `pub fn swap_bytes(&self) -> Self`
- `pub fn not_with_width(&self, width: usize) -> Option<Self>`
- `pub fn try_not(&self) -> Result<Self, MpError>`
- `pub fn leading_zeros(&self) -> Option<usize>`
- `pub fn leading_ones(&self) -> Option<usize>`
- `pub fn trailing_zeros(&self) -> usize`
- `pub fn trailing_ones(&self) -> usize`
- `pub fn count_ones(&self) -> usize`
- `pub fn count_zeros(&self) -> Option<usize>`
- `pub fn get_bit(&self, bit: usize) -> bool`
- `pub fn set_bit(&self, bit: usize) -> Self`
- `pub fn clear_bit(&self, bit: usize) -> Self`
- `pub fn toggle_bit(&self, bit: usize) -> Self`
- `pub fn test_bit(&self, bit: usize) -> bool`
- `pub fn set_bit_to(&self, bit: usize, value: bool) -> Self`
- `pub fn find_first_set_bit(&self) -> Option<usize>`
- `pub fn find_next_set_bit(&self, from: usize) -> Option<usize>`
- `pub fn find_first_zero_bit(&self) -> usize`
- `pub fn find_next_zero_bit(&self, from: usize) -> usize`
- `pub fn bit_range(&self, from: usize, to: usize) -> Self`

### Properties & Comparisons
- `pub const fn precision(&self) -> Precision`
- `pub fn is_zero(&self) -> bool`
- `pub fn is_one(&self) -> bool`
- `pub fn is_even(&self) -> bool`
- `pub fn is_odd(&self) -> bool`
- `pub fn is_power_of_two(&self) -> bool`
- `pub fn checked_next_power_of_two(&self) -> Option<Self>`
- `pub fn min(self, other: Self) -> Self`
- `pub fn max(self, other: Self) -> Self`
- `pub fn clamp(self, min: Self, max: Self) -> Self`
- `pub fn significant_bits(&self) -> usize`

### Number Theory, Roots & Primality
- `pub fn is_prime(&self) -> bool`
- `pub fn is_probably_prime(&self, k: u32) -> bool`
- `pub fn next_prime(&self) -> Option<Self>`
- `pub fn prev_prime(&self) -> Option<Self>`
- `pub fn isqrt(&self) -> Option<Self>`
- `pub fn sqrt_rem(&self) -> Option<(Self, Self)>`
- `pub fn nth_root(&self, n: u32) -> Option<Self>`
- `pub fn is_perfect_square(&self) -> bool`
- `pub fn euler_phi(&self) -> Option<Self>`
- `pub fn jacobi_symbol(&self, other: &Self) -> Option<i8>`
- `pub fn factorial(n: u32, precision: Precision) -> Self`
- `pub fn gcd(&self, other: &Self) -> Self`
- `pub fn gcd_lcm(&self, other: &Self) -> Option<(Self, Self)>`
- `pub fn lcm(&self, other: &Self) -> Option<Self>`
- `pub fn is_coprime(&self, other: &Self) -> bool`
- `pub fn extended_gcd(&self, other: &Self) -> Option<(Self, Self, Self)>`
- `pub fn add_mod(&self, other: &Self, modulus: &Self) -> Option<Self>`
- `pub fn sub_mod(&self, other: &Self, modulus: &Self) -> Option<Self>`
- `pub fn mul_mod(&self, other: &Self, modulus: &Self) -> Option<Self>`
- `pub fn pow_mod(&self, exp: &Self, modulus: &Self) -> Option<Self>`
- `pub fn invert(&self, modulus: &Self) -> Option<Self>`
- `pub fn montgomery_mul(&self, other: &Self, modulus: &Self) -> Option<Self>`
- `pub fn barrett_reduce(&self, modulus: &Self) -> Option<Self>`

### Conversions, Formatting & Serialization
- `pub fn to_u64(&self) -> Option<u64>`
- `pub fn to_u128(&self) -> Option<u128>`
- `pub fn to_usize(&self) -> Option<usize>`
- `pub fn to_i64(&self) -> Option<i64>`
- `pub fn to_i128(&self) -> Option<i128>`
- `pub fn to_isize(&self) -> Option<isize>`
- `pub fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseMpUintError>`
- `pub fn to_string_radix(&self, radix: u32) -> String`
- `pub fn to_f64(&self) -> Option<f64>`
- `pub fn to_f32(&self) -> Option<f32>`
- `pub fn to_le_bytes(&self) -> Vec<u8>`
- `pub fn from_le_bytes(bytes: &[u8]) -> Self`
- `pub fn to_be_bytes(&self) -> Vec<u8>`
- `pub fn from_be_bytes(bytes: &[u8]) -> Self`

Primitive construction uses the standard `From`/`TryFrom` traits. The optional
`num-traits` integration also implements `FromPrimitive` directly.

---

## 3. Exact Implemented API: MpInt (Signed Magnitude / Two's Complement Boundary)

The following are public inherent methods. For `new<T>` and all three
`with_precision_*<T>` constructors, the bound is `Self: From<T>`.

### Constructors & Capacity
- `pub fn zero() -> Self`
- `pub const fn zero_with_precision(bits: BoundedPrecision) -> Self`
- `pub fn one() -> Self`
- `pub fn minus_one() -> Self`
- `pub fn new<T>(value: T) -> Self` *(where `Self: From<T>`)*
- `pub fn with_capacity(capacity: usize) -> Self`
- `pub fn with_precision_checked<T>(value: T, bits: BoundedPrecision) -> Result<Self, MpError>`
- `pub fn with_precision_wrapping<T>(value: T, bits: BoundedPrecision) -> Self`
- `pub fn with_precision_saturating<T>(value: T, bits: BoundedPrecision) -> Self`
- `pub fn max_for_precision(bits: usize) -> Self`
- `pub fn min_for_precision(bits: usize) -> Self`
- `pub fn reserve(&mut self, additional: usize)`
- `pub fn reserve_exact(&mut self, additional: usize)`
- `pub fn shrink_to_fit(&mut self)`
- `pub const fn capacity(&self) -> usize`
- `pub const fn swap(&mut self, other: &mut Self)`
- `pub const fn as_debug_verbose(&self) -> DebugVerbose<'_, Self>`

> **Note:** `Precision::new_bounded` is the public validated precision constructor. The shared internal ambient-construction resolver is used by `From<T>` constructors and is not part of the public inherent API.

### Sign & Core Arithmetic Families
- `pub fn abs(&self) -> Self`
- `pub fn abs_sub(&self, other: &Self) -> Self` *(positive difference `max(0, a - b)` for `num_traits::Signed`; see `abs_diff` below for `|a - b|`)*
- `pub fn abs_assign(&mut self)`
- `pub fn checked_abs(&self) -> Option<Self>`
- `pub fn signum(&self) -> Self`
- `pub fn checked_add(&self, rhs: &Self) -> Option<Self>`
- `pub fn checked_sub(&self, rhs: &Self) -> Option<Self>`
- `pub fn checked_mul(&self, rhs: &Self) -> Option<Self>`
- `pub fn checked_div(&self, rhs: &Self) -> Option<Self>`
- `pub fn checked_rem(&self, rhs: &Self) -> Option<Self>`
- `pub fn wrapping_add(&self, rhs: &Self) -> Self`
- `pub fn wrapping_sub(&self, rhs: &Self) -> Self`
- `pub fn wrapping_mul(&self, rhs: &Self) -> Self`
- `pub fn wrapping_div(&self, rhs: &Self) -> Self`
- `pub fn wrapping_rem(&self, rhs: &Self) -> Self`
- `pub fn overflowing_add(&self, rhs: &Self) -> (Self, bool)`
- `pub fn overflowing_sub(&self, rhs: &Self) -> (Self, bool)`
- `pub fn overflowing_mul(&self, rhs: &Self) -> (Self, bool)`
- `pub fn overflowing_div(&self, rhs: &Self) -> (Self, bool)`
- `pub fn overflowing_rem(&self, rhs: &Self) -> (Self, bool)`
- `pub fn saturating_add(&self, rhs: &Self) -> Self`
- `pub fn saturating_sub(&self, rhs: &Self) -> Self`
- `pub fn abs_diff(&self, other: &Self) -> MpUint`
- `pub fn saturating_mul(&self, rhs: &Self) -> Self`
- `pub fn saturating_div(&self, rhs: &Self) -> Self`
- `pub fn saturating_rem(&self, rhs: &Self) -> Self`
- `pub fn try_add(&self, rhs: &Self) -> Result<Self, MpError>`
- `pub fn try_sub(&self, rhs: &Self) -> Result<Self, MpError>`
- `pub fn try_mul(&self, rhs: &Self) -> Result<Self, MpError>`
- `pub fn try_div(&self, rhs: &Self) -> Result<Self, MpError>`
- `pub fn try_rem(&self, rhs: &Self) -> Result<Self, MpError>`
- `pub fn strict_add(&self, rhs: &Self) -> Self`
- `pub fn strict_sub(&self, rhs: &Self) -> Self`
- `pub fn strict_mul(&self, rhs: &Self) -> Self`
- `pub fn strict_div(&self, rhs: &Self) -> Self`
- `pub fn strict_rem(&self, rhs: &Self) -> Self`
- `pub fn assign_add(&mut self, a: &Self, b: &Self)`
- `pub fn assign_sub(&mut self, a: &Self, b: &Self)`
- `pub fn assign_mul(&mut self, a: &Self, b: &Self)`
- `pub fn assign_square(&mut self, a: &Self)`
- `pub fn mul_2exp(&self, shift: usize) -> Self`
- `pub fn div_2exp(&self, shift: usize) -> Self`
- `pub fn widening_mul(&self, other: &Self) -> (Self, Self)`
- `pub fn try_widening_mul(&self, other: &Self) -> Result<(Self, Self), MpError>`
- `pub fn carrying_mul(&self, other: &Self, carry: &Self) -> (Self, Self)`
- `pub fn try_carrying_mul(&self, other: &Self, carry: &Self) -> Result<(Self, Self), MpError>`
- `pub fn carrying_mul_add(&self, other: &Self, carry1: &Self, carry2: &Self) -> (Self, Self)`
- `pub fn mul_add(&self, a: &Self, b: &Self) -> Self`
- `pub fn midpoint(&self, other: &Self) -> Self`
- `pub fn is_divisible_by(&self, other: &Self) -> bool`
- `pub fn is_divisor_of(&self, other: &Self) -> bool`
- `pub fn div_rem(&self, rhs: &Self) -> Option<(Self, Self)>`
- `pub fn div_trunc(&self, rhs: &Self) -> Self`
- `pub fn checked_div_trunc(&self, rhs: &Self) -> Option<Self>`
- `pub fn rem_trunc(&self, rhs: &Self) -> Self`
- `pub fn checked_rem_trunc(&self, rhs: &Self) -> Option<Self>`
- `pub fn div_rem_euclid(&self, rhs: &Self) -> Option<(Self, Self)>`
- `pub fn div_euclid(&self, rhs: &Self) -> Self`
- `pub fn checked_div_euclid(&self, rhs: &Self) -> Option<Self>`
- `pub fn rem_euclid(&self, rhs: &Self) -> Self`
- `pub fn checked_rem_euclid(&self, rhs: &Self) -> Option<Self>`
- `pub fn div_rem_floor(&self, rhs: &Self) -> Option<(Self, Self)>`
- `pub fn div_floor(&self, rhs: &Self) -> Self`
- `pub fn checked_div_floor(&self, rhs: &Self) -> Option<Self>`
- `pub fn mod_floor(&self, rhs: &Self) -> Self`
- `pub fn checked_mod_floor(&self, rhs: &Self) -> Option<Self>`
- `pub fn div_ceil(&self, rhs: &Self) -> Self`
- `pub fn checked_div_ceil(&self, rhs: &Self) -> Option<Self>`
- `pub fn pow(&self, exp: u32) -> Self`
- `pub fn checked_pow(&self, exp: u32) -> Option<Self>`
- `pub fn try_pow(&self, exp: u32) -> Result<Self, MpError>`
- `pub fn square(&self) -> Self`

### Bitwise Operations & Shifts
- `pub fn checked_shl(&self, shift: usize) -> Option<Self>`
- `pub fn wrapping_shl(&self, shift: usize) -> Self`
- `pub fn overflowing_shl(&self, shift: usize) -> (Self, bool)`
- `pub fn saturating_shl(&self, shift: usize) -> Self`
- `pub fn try_shl(&self, shift: usize) -> Result<Self, MpError>`
- `pub fn count_ones(&self) -> Option<usize>`
- `pub fn count_zeros(&self) -> Option<usize>`
- `pub fn leading_zeros(&self) -> Option<usize>`
- `pub fn leading_ones(&self) -> Option<usize>`
- `pub fn trailing_zeros(&self) -> usize`
- `pub fn trailing_ones(&self) -> Option<usize>`
- `pub fn swap_bytes(&self) -> Option<Self>`
- `pub fn reverse_bits(&self, width: usize) -> Option<Self>`
- `pub fn not_with_width(&self, width: usize) -> Option<Self>`
- `pub fn try_not(&self) -> Result<Self, MpError>`
- `pub fn rotate_left(&self, n: u32, width: usize) -> Option<Self>`
- `pub fn rotate_right(&self, n: u32, width: usize) -> Option<Self>`
- `pub fn get_bit(&self, bit: usize) -> bool`
- `pub fn set_bit(&self, bit: usize) -> Self`
- `pub fn clear_bit(&self, bit: usize) -> Self`
- `pub fn toggle_bit(&self, bit: usize) -> Self`
- `pub fn test_bit(&self, bit: usize) -> bool`
- `pub fn set_bit_to(&self, bit: usize, value: bool) -> Self`
- `pub fn bit_range(&self, from: usize, to: usize) -> Self`
- `pub fn find_first_set_bit(&self) -> Option<usize>`
- `pub fn find_next_set_bit(&self, from: usize) -> Option<usize>`
- `pub fn find_first_zero_bit(&self) -> Option<usize>`
- `pub fn find_next_zero_bit(&self, from: usize) -> usize`

### Number Theory, Roots & Primality
- `pub fn is_prime(&self) -> bool`
- `pub fn is_probably_prime(&self, k: u32) -> bool`
- `pub fn next_prime(&self) -> Option<Self>`
- `pub fn prev_prime(&self) -> Option<Self>`
- `pub fn checked_isqrt(&self) -> Option<Self>`
- `pub fn sqrt_rem(&self) -> Option<(Self, Self)>`
- `pub fn nth_root(&self, n: u32) -> Option<Self>`
- `pub fn is_perfect_square(&self) -> bool`
- `pub fn euler_phi(&self) -> Option<Self>`
- `pub fn jacobi_symbol(&self, other: &Self) -> Option<i8>`
- `pub fn factorial(n: u32, precision: Precision) -> Self`
- `pub fn gcd(&self, other: &Self) -> Self`
- `pub fn gcd_lcm(&self, other: &Self) -> Option<(Self, Self)>`
- `pub fn lcm(&self, other: &Self) -> Option<Self>`
- `pub fn is_coprime(&self, other: &Self) -> bool`
- `pub fn extended_gcd(&self, other: &Self) -> Option<(Self, Self, Self)>`
- `pub fn add_mod(&self, other: &Self, modulus: &Self) -> Option<Self>`
- `pub fn sub_mod(&self, other: &Self, modulus: &Self) -> Option<Self>`
- `pub fn mul_mod(&self, other: &Self, modulus: &Self) -> Option<Self>`
- `pub fn pow_mod(&self, exp: &Self, modulus: &Self) -> Option<Self>`
- `pub fn invert(&self, modulus: &Self) -> Option<Self>`
- `pub fn montgomery_mul(&self, other: &Self, modulus: &Self) -> Option<Self>`
- `pub fn barrett_reduce(&self, modulus: &Self) -> Option<Self>`

### Properties & Comparisons
- `pub const fn precision(&self) -> Precision`
- `pub fn is_zero(&self) -> bool`
- `pub fn is_one(&self) -> bool`
- `pub fn is_positive(&self) -> bool`
- `pub const fn is_negative(&self) -> bool`
- `pub fn is_minus_one(&self) -> bool`
- `pub fn is_even(&self) -> bool`
- `pub fn is_odd(&self) -> bool`
- `pub fn is_power_of_two(&self) -> bool`
- `pub fn checked_next_power_of_two(&self) -> Option<Self>`
- `pub fn min(self, other: Self) -> Self`
- `pub fn max(self, other: Self) -> Self`
- `pub fn clamp(self, min: Self, max: Self) -> Self`
- `pub fn significant_bits(&self) -> usize`
- `pub fn unsigned_abs(&self) -> MpUint`

### Conversions & Formatting
- `pub fn to_u64(&self) -> Option<u64>`
- `pub fn to_u128(&self) -> Option<u128>`
- `pub fn to_usize(&self) -> Option<usize>`
- `pub fn to_i64(&self) -> Option<i64>`
- `pub fn to_i128(&self) -> Option<i128>`
- `pub fn to_isize(&self) -> Option<isize>`
- `pub fn from_str_radix(str: &str, radix: u32) -> Result<Self, ParseMpIntError>`
- `pub fn to_string_radix(&self, radix: u32) -> String`
- `pub fn to_f64(&self) -> Option<f64>`
- `pub fn to_f32(&self) -> Option<f32>`
- `pub fn to_le_bytes(&self) -> Vec<u8>`
- `pub fn from_le_bytes(bytes: &[u8]) -> Self`
- `pub fn to_be_bytes(&self) -> Vec<u8>`
- `pub fn from_be_bytes(bytes: &[u8]) -> Self`

Primitive construction uses the standard `From`/`TryFrom` traits. The optional
`num-traits` integration also implements `FromPrimitive` directly.

---

## 4. Implemented Trait Implementations (Both Types)

- **`core::ops` operators:** `Add`, `Sub`, `Mul`, `Div`, `Rem`, `BitAnd`, `BitOr`, `BitXor` across all four same-type ownership combinations (`T op T`, `&T op T`, `T op &T`, `&T op &T`). Assign variants accept owned or borrowed same-type operands. Arithmetic with primitive RHS values or mixed `MpInt`/`MpUint` is not implemented; convert explicitly. `Not` accepts owned or borrowed values of either type (`MpUint` panics on `Unlimited`); `Neg` accepts owned or borrowed `MpInt`.
- **Shifts:** `Shl`, `Shr`, `ShlAssign`, `ShrAssign` support all twelve primitive integer RHS types (`u8` through `u128`, `usize`, `i8` through `i128`, `isize`). Counts must convert to `usize`; negative or unrepresentable counts panic. No arbitrary-precision shift operand is implemented.
- **Comparisons & Hashing:** `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash` (value-based equality ignoring precision metadata). Cross-type `PartialEq<MpInt>` and `PartialOrd<MpInt>` implemented for `MpUint` and reverse.
- **Value management and formatting:** `Clone` (including `clone_from`), `Default`, `FromStr`, `Display`, `Debug`, `Binary`, `Octal`, `LowerHex`, `UpperHex`. The integers are `Send + Sync` but not `Copy`. `DebugVerbose` implements `Debug` for both integer types.
- **Iterators:** `Sum<T>`, `Sum<&T>`, `Product<T>`, `Product<&T>`; all start with an unlimited identity value and return unlimited results.
- **`num-traits` feature only:** `Zero`, `One`, `Num`, `ToPrimitive`, `FromPrimitive`, plus `Unsigned` for `MpUint` and `Signed` for `MpInt`. Floating-point `FromPrimitive` methods use the trait's default conversion through primitive integers, not an arbitrary-size IEEE-754 decoder. `num_traits::Bounded` and `num_integer::Integer` are not implemented.

### Owned Conversion Matrix

Here `U` means any unsigned primitive integer and `I` any signed primitive integer.

| Conversion | Implementation |
| --- | --- |
| `U -> MpUint`, `U -> MpInt`, `I -> MpInt` | `From`, exact, ambient precision is a floor. |
| `I -> MpUint` | `TryFrom`; `NegativeInput` for negative values, otherwise exact ambient construction. |
| `MpUint -> MpInt` | `From`; preserves value and widens bounded precision by one bit when a sign bit is needed. |
| `MpInt -> MpUint` | `TryFrom`; rejects negative values, otherwise preserves precision. |
| `MpUint -> U`, `MpInt -> U`, `MpInt -> I` | `TryFrom`; `IntegerConversionLoss` when out of range. |
| `MpUint -> I` | No `TryFrom`; use the inherent `to_i64`, `to_i128`, or `to_isize` where applicable. |
| `f32`/`f64 -> MpUint`/`MpInt` | No `From`/`TryFrom`; optional `FromPrimitive` support is described above. |

No borrowed-source conversion matrix is implemented. Standard blanket
`Into`/`TryInto` implementations follow from the listed traits.

---

## 5. Current Behavioral Contracts

### Precision and Assignment

- Binary arithmetic combines bounded widths by taking the maximum; any
  unlimited operand makes the result unlimited. Bounded overflow does not
  automatically widen that result.
- Assignment operators and `assign_add`, `assign_sub`, `assign_mul`, and
  `assign_square` preserve destination precision. Operand widths do not limit
  fused assignment. Bounded overflow panics before changing the destination.
  Unlimited fused destinations write directly into their reusable buffer;
  bounded `assign_add` also writes directly when operand widths prove the sum
  fits. Other bounded fused results are validated in temporary storage before
  copying them back.
- Unsigned `assign_sub` returns `true` on underflow and leaves the destination
  unchanged; successful assignment returns `false`. Signed `assign_sub`
  returns `()`. `clone_from` copies source precision; `swap` exchanges it.
- Exact primitive and nonempty byte construction treats ambient bounded
  precision as a floor and widens to fit. Parsing treats it as a cap.
  Unsigned empty byte input follows ambient construction; signed empty byte
  input returns unlimited zero. Same-type `new(existing)` preserves its
  precision through the identity `From` implementation.
- `Default`, `zero`, and `one` use unlimited precision. Iterator `Sum` and
  `Product` start with unlimited identities and return unlimited results.
- `precision()` returns the value's `Precision`; `as_debug_verbose` displays
  it. Reserving capacity does not change precision. `reserve_exact`
  avoids deliberate growth but does not promise exact allocator capacity;
  inline capacity is four limbs.

### Arithmetic and Primitive Differences

- Ordinary bounded arithmetic panics on overflow in debug and release builds.
  Unsigned ordinary subtraction panics on underflow. Checked arithmetic returns
  `None` for its checked failures; it does not make allocation failure recoverable.
- Wrapping and saturating division/remainder return zero for a zero divisor;
  overflowing variants return `(zero, true)`. These behaviors differ from
  Rust primitive methods, which panic for a zero divisor.
- Left-shift policies check mathematical result overflow. `checked_shl` rejects
  shifted-out value bits; wrapping shifts do not reduce counts modulo width.
  Overflowing shifts flag mathematical overflow. These are different contracts
  from primitive checked, wrapping, and overflowing shifts.
- Signed `div_2exp` uses arithmetic right shift, rounding negative results down;
  ordinary signed division truncates toward zero.
- Signed `MIN / -1` overflow depends on the resolved result width. An unlimited
  divisor removes the bounded result limit.
- Bounded signed width one contains only `-1` and `0`; operations whose
  mathematical result is positive one can fail, including `pow(0)` and
  `factorial()` on zero.

### Bits, Roots, and Number Theory

- Explicit-width complement, rotations, and bit reversal accept unlimited
  inputs and return bounded results. Width zero or `usize::MAX` is invalid.
  `try_not` requires bounded precision even on signed values; signed `!`
  supports infinite two's complement for unlimited values.
- Bit updates return new values. Bounded reads above the width are false and
  updates there leave the value unchanged. Signed bit-range extraction returns
  a nonnegative value and may widen to fit a sign bit.
- Next-bit scans include their starting index. Unsigned zero-bit scans include
  the zero extension beyond bounded width. Signed zero-bit scans use the
  bounded width or `usize::MAX` as their no-match sentinel; the first-zero
  method converts it to `None`. `trailing_zeros(0)` returns zero. Unlimited
  negative signed `trailing_ones` returns `None`.
- Signed `checked_isqrt` rejects negative values. Signed `sqrt_rem`, `nth_root`,
  and `is_perfect_square` use the magnitude. Unsigned `isqrt` and `sqrt_rem`
  always return `Some`; signed roots also enforce result precision.
- Signed modular addition, subtraction, multiplication, reduction, inversion,
  and exponentiation use operand magnitudes. A negative signed exponent
  requests inversion of the base magnitude.
- `montgomery_mul` returns `a*b*R^-1 mod m`, where
  `R = 2^(usize::BITS * modulus_limb_count)`. It requires an odd nonzero
  modulus, and its value can depend on pointer width.
- Signed `extended_gcd` returns signed Bézout coefficients subject to precision.
  Unsigned `extended_gcd` returns modular coefficients, not signed Bézout
  coefficients: for nonzero inputs, `a*x = gcd(a,b) (mod b)` and
  `b*y = gcd(a,b) (mod a)`. A zero second operand returns `None`.
- Signed `gcd(MIN, 0)` can panic if the nonnegative gcd does not fit the
  resolved width. `lcm` and `gcd_lcm` report out-of-range results with `None`.
- `is_prime` is exact through `u64::MAX`; larger inputs use Baillie–PSW probable
  primality. `is_probably_prime(k)` also uses the exact test through `u64::MAX`;
  larger inputs use the first `clamp(k, 1, 64)` prime bases. Fixed bases do not
  provide the independent-random-round error bound. `next_prime` and `prev_prime`
  exclude the input and preserve its precision. `prev_prime` returns `None` for
  inputs at most two. Both use the same probable-prime qualification for large
  candidates.

### Encoding and Identity

- Byte output is minimal unsigned magnitude or minimal signed two's complement;
  both encode zero as an empty vector. Signed positive output includes a zero
  sign byte when necessary. Input accepts redundant zero/sign extension.
  Precision is not serialized in these byte formats.
- `to_f32` and `to_f64` round to nearest, ties to even; `None` reports exponent
  overflow, not loss of integer precision through rounding.
- Equality, ordering, and hashing within each type ignore precision. Hashing
  uses native limbs and, for signed values, the sign; its encoding is not a
  portable wire format or a shared hash representation between the two types.
- Radix parsing and formatting support bases 2 through 36. No custom digit
  alphabet, public mutable limb access, float `TryFrom`, or arbitrary-precision
  float decoder is exposed.
