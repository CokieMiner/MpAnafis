# MpInt / MpUint — Exact API Inventory & Gap Analysis

This document inventories the public API implemented in `src/int/api/` for
`MpUint` and `MpInt`. The normative target
specification lives in [spec.md](spec.md), and future gaps
are listed separately below.

---

## 1. Executive Summary & Resolutions

This inventory reflects the comprehensive implementation of core number theory, bitwise parity, and division APIs across both `MpUint` and `MpInt`. Previously noted architectural parity gaps have been resolved:
1. **Number Theory & Modular Math Parity:** `MpInt` now wraps and delegates all major number theory operations (`gcd`, `lcm`, `is_prime`, `pow_mod`, `invert`, `checked_isqrt`, `factorial`, `square`, `montgomery_mul`, `barrett_reduce`, etc.) while correctly managing sign and domain invariants.
2. **Bitwise & Byte Serialization Parity:** `MpInt` now implements the complete suite of bitwise inspection and manipulation methods (`leading_zeros`, `trailing_zeros`, `count_ones`, `trailing_ones`, `get_bit`, `set_bit`, `clear_bit`, `toggle_bit`, `test_bit`, `set_bit_to`, `rotate_left`, `rotate_right`, `swap_bytes`, `reverse_bits`), along with endian byte serialization (`to_le_bytes`, `to_be_bytes`, `from_le_bytes`, `from_be_bytes`).
3. **Euclidean & Rounding Division:** Both types now natively implement Euclidean division (`div_euclid`, `rem_euclid`), truncation/floor/ceiling rounding division (`div_trunc`, `rem_trunc`, `div_floor`, `mod_floor`, `div_ceil`), `div_rem`, `mul_add`, `midpoint`, divisibility predicates (`is_divisor_of`, `is_divisible_by`), `try_pow`, and `significant_bits`.
4. **`MpInt` Arithmetic Parity:** `MpInt` now has full arithmetic family parity with `MpUint`: `wrapping_div`, `wrapping_rem`, `saturating_div`, `saturating_rem`, `assign_add`, and `checked_next_power_of_two` are all implemented.

### Spec Alignment Pass

The following divergences between this inventory and [spec.md](spec.md) have been
reconciled. Where the two disagreed, the table records which one moved:

| Item | Resolution |
|---|---|
| `not_bits(width)` | **Code renamed** to `not_with_width(width)`, matching the spec. `try_not()` added alongside it, completing the pair the spec had always named. |
| `nth_root(n)` | **Spec renamed** from `root`/`root_rem` to `nth_root`/`nth_root_rem`. A bare `root` reads as a square root; the degree is an argument. `nth_root_rem` remains unimplemented. |
| `abs_sub` | **Kept and specified.** It is the positive difference required by `num_traits::Signed`, not the absolute difference, and is now documented adjacent to `abs_diff` in both files because the names invite confusion. |
| `assign_add` / `assign_sub` / `assign_mul` / `assign_square` | **Added to spec §7** and to this inventory. `assign_mul` and `assign_square` were absent from both despite being implemented and benchmarked. |
| `with_capacity`, `reserve_exact` | **Added to spec §7** Memory & Iterators, alongside the `reserve`/`shrink_to_fit`/`capacity` that were already listed. |
| Precision accessors | **Spec §13 corrected.** It claimed no precision accessor is exposed; the `Precision` and `BoundedPrecision` types are in fact publicly readable. No accessor exists on a *value*, which is what the paragraph meant to say. |

Still outstanding, tracked in §5 below: `nth_root_rem`, and the half-GCD the
spec's §10 algorithm ladder calls for.

---

## 2. Exact Implemented API — `MpUint` (Unsigned Native-Width Limbs)

The following public inherent methods are currently implemented for `MpUint`:

### Precision Metadata
- `pub const fn BoundedPrecision::new(bits: usize) -> Option<BoundedPrecision>`
- `pub const fn BoundedPrecision::get(self) -> usize`
- `pub const fn Precision::new_bounded(bits: usize) -> Option<Precision>`
- `pub const fn AmbientPrecision::new_bounded(bits: usize) -> Option<AmbientPrecision>`
- `pub fn PrecisionContext::active() -> AmbientPrecision`
- `pub fn PrecisionContext::set_global(precision: AmbientPrecision) -> AmbientPrecision`
- `PrecisionContext::with_bounded(bits, closure)` / `with_unlimited(closure)` with `std`

Bounded widths use the canonical range `1..usize::MAX`; zero and the top
`usize` value are rejected because the ambient encoding reserves them.

### Constructors & Capacity
- `pub fn zero() -> Self`
- `pub const fn zero_with_precision(bits: BoundedPrecision) -> Self`
- `pub fn one() -> Self`
- `pub fn new<T>(value: T) -> Self` *(where `T: Into<MpUint>`)*
- `pub fn with_capacity(capacity: usize) -> Self`
- `pub fn with_precision_checked<T>(value: T, bits: BoundedPrecision) -> Result<Self, MpError>`
- `pub fn with_precision_wrapping<T>(value: T, bits: BoundedPrecision) -> Self`
- `pub fn with_precision_saturating<T>(value: T, bits: BoundedPrecision) -> Self`
- `pub fn max_for_precision(bits: usize) -> Self`
- `pub const fn min_for_precision(bits: usize) -> Self`
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
- `pub fn mul_add(&self, a: &Self, b: &Self) -> Self`
- `pub fn midpoint(&self, other: &Self) -> Self`
- `pub fn is_divisible_by(&self, other: &Self) -> bool`
- `pub fn is_divisor_of(&self, other: &Self) -> bool`
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
- `pub fn is_zero(&self) -> bool`
- `pub fn is_one(&self) -> bool`
- `pub fn is_even(&self) -> bool`
- `pub fn is_odd(&self) -> bool`
- `pub fn is_power_of_two(&self) -> bool`
- `pub fn checked_next_power_of_two(&self) -> Option<Self>`
- `pub fn min(&self, other: &Self) -> Self`
- `pub fn max(&self, other: &Self) -> Self`
- `pub fn clamp(&self, min: &Self, max: &Self) -> Self`
- `pub fn significant_bits(&self) -> usize`

### Number Theory, Roots & Primality
- `pub fn is_prime(&self) -> bool`
- `pub fn is_probably_prime(&self, k: u32) -> bool`
- `pub fn next_prime(&self) -> Option<Self>`
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
- `pub fn to_f64(&self) -> Option<f64>` / `to_f32(&self) -> Option<f32>`
- `pub fn to_le_bytes(&self) -> Vec<u8>` / `from_le_bytes(bytes: &[u8]) -> Self`
- `pub fn to_be_bytes(&self) -> Vec<u8>` / `from_be_bytes(bytes: &[u8]) -> Self`

Primitive construction uses the standard `From`/`TryFrom` traits. The optional
`num-traits` integration also implements `FromPrimitive` directly.

---

## 3. Exact Implemented API — `MpInt` (Signed Magnitude / Two's Complement Boundary)

The following public inherent methods are currently implemented for `MpInt`:

### Constructors & Capacity
- `pub fn zero() -> Self`
- `pub const fn zero_with_precision(bits: BoundedPrecision) -> Self`
- `pub fn one() -> Self`
- `pub fn minus_one() -> Self`
- `pub fn new<T>(value: T) -> Self` *(where `T: Into<MpInt>`)*
- `pub fn with_capacity(capacity: usize) -> Self`
- `pub fn with_precision_checked<T>(value: T, bits: BoundedPrecision) -> Result<Self, MpError>`
- `pub fn with_precision_wrapping<T>(value: T, bits: BoundedPrecision) -> Self`
- `pub fn with_precision_saturating<T>(value: T, bits: BoundedPrecision) -> Self`
- `pub fn max_for_precision(bits: usize) -> Self`
- `pub fn min_for_precision(bits: usize) -> Self`
- `pub fn reserve(&mut self, additional: usize)`
- `pub fn reserve_exact(&mut self, additional: usize)`
- `pub fn shrink_to_fit(&mut self)`
- `pub fn capacity(&self) -> usize`
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
- `pub fn mul_add(&self, a: &Self, b: &Self) -> Self`
- `pub fn midpoint(&self, other: &Self) -> Self`
- `pub fn is_divisible_by(&self, other: &Self) -> bool`
- `pub fn is_divisor_of(&self, other: &Self) -> bool`
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
- `pub fn is_zero(&self) -> bool`
- `pub fn is_one(&self) -> bool`
- `pub fn is_positive(&self) -> bool`
- `pub fn is_negative(&self) -> bool`
- `pub fn is_minus_one(&self) -> bool`
- `pub fn is_even(&self) -> bool`
- `pub fn is_odd(&self) -> bool`
- `pub fn is_power_of_two(&self) -> bool`
- `pub fn checked_next_power_of_two(&self) -> Option<Self>`
- `pub fn min(&self, other: &Self) -> Self`
- `pub fn max(&self, other: &Self) -> Self`
- `pub fn clamp(&self, min: &Self, max: &Self) -> Self`
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
- `pub fn to_f64(&self) -> Option<f64>` / `to_f32(&self) -> Option<f32>`
- `pub fn to_le_bytes(&self) -> Vec<u8>` / `from_le_bytes(bytes: &[u8]) -> Self`
- `pub fn to_be_bytes(&self) -> Vec<u8>` / `from_be_bytes(bytes: &[u8]) -> Self`

Primitive construction uses the standard `From`/`TryFrom` traits. The optional
`num-traits` integration also implements `FromPrimitive` directly.

---

## 4. Implemented Trait Implementations (Both Types)

- **`core::ops` Operators:** `Add`, `Sub`, `Mul`, `Div`, `Rem`, `BitAnd`, `BitOr`, `BitXor` across all 4 ownership combinations (`T op T`, `&T op T`, `T op &T`, `&T op &T`). The corresponding assign variants are implemented for arithmetic/bitwise operators. `Shl`/`Shr` and `ShlAssign`/`ShrAssign` are implemented for primitive RHS types (`u32`, `u64`, `usize`). `Not` is implemented for both (`MpUint` panics on `Unlimited`). `Neg` is implemented for `MpInt`.
- **Comparisons & Hashing:** `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash` (value-based equality ignoring precision metadata). Cross-type `PartialEq<MpInt>` and `PartialOrd<MpInt>` implemented for `MpUint` and reverse.
- **Conversions & Formatting:** `Default`, `FromStr`, `Display`, `Debug`, `Binary`, `Octal`, `LowerHex`, `UpperHex`.
- **Iterators & `num-traits`:** `core::iter::Sum`, `core::iter::Product`, `num_traits::Zero`, `One`, `Num`, `Unsigned` (`MpUint`), `Signed` (`MpInt`), `ToPrimitive`, `FromPrimitive`.

---

## 5. Detailed List of Missing APIs & Gaps vs. `IMPLEMENTATION.md`

### A. Completely Missing Methods on Both `MpUint` and `MpInt`
The following methods and families specified in Section 7 of `IMPLEMENTATION.md` are **unimplemented on both types**:

1. **Core & Advanced Arithmetic:**
   - Widening / Carrying Multiplication: `widening_mul`, `carrying_mul`, `carrying_mul_add`, `try_widening_mul`, `try_carrying_mul`.
   - Power-of-Two Division / Modulo: `div_2exp`, `mul_2exp`, `mod_floor_2exp`, `rem_trunc_2exp`.

2. **Sign, Properties & Precision Helpers:**
   - `apply_sign`: Branchless sign application.
   - Precision Identity Helpers: `same_precision`, `same_value_and_precision`, `bit_identical`.
   - Public precision accessor: `precision(&self) -> Precision`.
   - Primitive Bit Casts: `cast_signed`, `cast_unsigned`.

3. **Bitwise Modifications & Bit Scanning:**
   - Advanced Bit Range / Scanning: `set_bit_range`, `take_lowest_one_bit`, `take_highest_one_bit`.

4. **Number Theory & Modular Math:**
   - Parameterized Primality: `is_probably_prime_with_rng(k: u32, rng: &mut R)`, `prev_prime`.
   - Factorization & Divisors: `factor`, `prime_factors`, `batch_shared_factor_detection`, `divisors`, `divisor_count`, `divisor_sum`, `is_smooth`, `remove_factor`, `extended_gcd_cofactors`, `gcd_slice`.
   - Advanced Roots & Logarithms: `nth_root_rem`, `sqrt_mod`, `ilog`, `ilog2`, `ilog10`, `checked_ilog*`. (`nth_root` is implemented; the spec was renamed from `root`/`root_rem` to match.)
   - Symbols & Congruence: `kronecker_symbol`, `legendre_symbol`, `is_congruent`.
   - Modular and advanced math: `chinese_remainder`, `carmichael_lambda`, `moebius_mu`, `multiplicative_order`, `primitive_root`, `discrete_log`. Any future `pow_mod_sec` requires a separately audited constant-time contract.
   - `hamming_distance`.

5. **Planned Combinatorics & Special Functions:**
   - `double_factorial`, `subfactorial`, `primorial`, `binomial`, `multinomial`, `catalan`, `fibonacci`, `lucas`, `stirling_first`, `stirling_second`, `bell`, `partition`, `euler_number`, `bernoulli`, `harmonic_number`, `tetration`, `hyperoperation`, `rising_factorial`, `falling_factorial`.

6. **Constructors, Conversions, Parsing & Slice Inspection:**
   - Limb Slice Constructors: `from_limbs_le`, `from_limbs_be`.
   - ASCII Parsing: `from_ascii`, `from_ascii_radix`.
   - Radix Digit Helpers: `digits_in_base`, `to_radix_be`, `to_radix_le`, `from_radix_be`, `from_radix_le`.
   - Native Endian & Buffer I/O: `to_native_endian_bytes`, `from_native_endian_bytes`, `write_be_bytes`, `write_le_bytes`, `write_native_endian_bytes`.
   - Float Conversion Controls: `try_from_f64_exact`, `_trunc`, `_floor`, `_ceil`.
   - Read-Only Accessors: `limbs(&self)`, `bits()`, `digits(base)`. Raw mutable
     limb access deliberately remains internal because it could violate
     normalization and bounded-precision invariants.

7. **Ecosystem Integrations:**
   - `serde` Serialize / Deserialize support.
   - `rand` Distribution / Uniform trait integrations and random constructors (`random_bits`, `random_below`, `random_range`).
   - `pyo3` Python bindings (`mp-int-pyo3`).
   

### Extra API

| API | Current source evidence | Decision for this audit |
|---|---|---|
| Custom alphabets/Base58/Base64 | Only radix `2..=36` parsing/formatting exists; no digit-alphabet or caller-buffer surface | Reject for now. First specify alphabet uniqueness, zero encoding, sign handling, canonical parsing, and buffer sizing; then benchmark a generic direct encoder against the existing string path. |
| Direct hashing | `Hash` is implemented for value identity, but there is no stable byte-stream sink or crypto-digest integration | Reject conflation. Keep Rust `Hash` separate from canonical serialization; design a no-alloc byte visitor only after its endian/sign wire contract is fixed. |
| Exact-bit-length random primes | No `rand` feature/dependency or RNG API exists; current primality is deterministic `is_probably_prime(k)`/`is_prime` | Reject addition in the current feature surface. A correct API needs an RNG boundary, exact high-bit/odd candidate generation, sieve policy, probable-prime semantics, and `no_std` ownership rules. |
| Other planned families | The source/spec comparison identifies buffer I/O, limb/radix accessors, random values, factorization/divisors, advanced roots/logs, and combinatorics as absent | Keep as roadmap gaps. They need separate design and tests; adding convenience wrappers before the core buffer, RNG, and error contracts would multiply API surface without resolving the underlying semantics. |
