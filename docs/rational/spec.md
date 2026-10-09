# MpRational Implementation Planning

**Status: planned.** The crate currently exports integer types only. All types,
methods, features, and algorithms below are design requirements. The
[integer inventory](../int/api-inventory.md) records implemented APIs.

## 0. The Precision System (Delegated)

`MpRational` does **not** have its own independent precision metadata field or hierarchy. Instead, it relies entirely on the precision tracking of its constituent parts: `numer: MpInt` and `denom: MpUint`.

- **Component Precision**: A rational number's precision is simply the precision of its numerator and denominator. They will typically share the same bounds (e.g. if constructed via `MpRational::from(1)` in a 256-bit global context), but can technically have asymmetric bounds if explicitly constructed that way.
- **Rule Delegation**: All rules regarding `Context`, `Global`, `Bounded + Unlimited = Unlimited`, and `max(width_a, width_b)` are natively handled by the underlying integer operators when arithmetic is performed on the components.
- **Intermediate Overflows**: See Section 0.1 for the intermediate overflow policy.

### 0.1 Intermediate Overflow Policy
To satisfy the invariant that overflow checks apply only to the *final canonical result*, `MpRational` arithmetic algorithms (like addition and multiplication) must perform intermediate coefficient scaling in an unbounded or widening workspace. They then reduce by the final GCD, and only then attempt to fit the canonical result back into the target `max` precision bounds.
**Example**: Adding two rationals with 256-bit coefficient limits can require
up to 512 magnitude bits, requiring 513 signed bits before reduction: each
signed cross-product has magnitude below $2^{511}$, and their sum has magnitude
below $2^{512}$.
Only the reduced coefficients are checked against the destination limits.
- **Mixed-Type Promotion**: `MpInt` / `MpUint` promote exactly into `MpRational` for mixed rational operations. Any operation between `MpRational` and `MpFloat` promotes to `MpFloat`, rounded using the active float target precision and rounding mode.

> **Note:** Bounded precision on rationals is a resource and representation bound over the coefficients. Unlike bounded integers, bounded rationals do not form a closed algebraic structure.

## Type Definition
- **Type**: `InternalMpRational`
- **Description**: The core data structure for arbitrary precision rational numbers.
- **Invariants**: 
  1. **Strictly Positive Denominator**: `denom > 0`. The sign is exclusively carried by `numer`.
  2. **Canonical Form**: The fraction is always reduced to lowest terms, meaning $\gcd(|numer|, denom) == 1$.
  3. **Normalization of Zero**: `0` is uniquely represented as `0 / 1`.

## Methods

### 1. Parts Management & Constructors
- `new(numer: impl Into<MpInt>, denom: impl Into<MpUint>) -> Result<Self, RationalError>`: Converts the components, rejects a zero denominator, and reduces by their GCD. This signature accepts only infallibly unsigned-convertible denominators. A separate fallible signed-input constructor may return `NegativeDenominator`.
- `from_parts(numer: MpInt, denom: MpUint) -> Result<Self, RationalError>`: Accepts `Mp` types directly. Will reduce by `gcd(n, d)`.
- `new_raw(numer: MpInt, denom: MpUint) -> Result<Self, RationalError>`: Validates canonical components without reducing them. Returns `NonCanonical` for unreduced input and `DenominatorZero` for zero denominator in every build mode.
- `new_unchecked(numer: MpInt, denom: MpUint) -> Self`: `unsafe fn`. Caller MUST guarantee: 1) `denom != 0`, 2) `gcd(|numer|, denom) == 1`, and 3) if `numer == 0`, then `denom == 1`.
- `numer` / `denom`: Returns `&MpInt` and `&MpUint`.
- `into_numer_denom`: Returns `(MpInt, MpUint)`.

### 2. Core Arithmetic
- `add` / `sub` (Addition and Subtraction):
  - *Implementation Details:* For $x = a/b$ and $y = c/d$, define $g = \gcd(b, d)$, $b' = b/g$, $d' = d/g$. Then $x \pm y = \frac{a d' \pm c b'}{b' d}$. Finally reduce by $h = \gcd(|a d' \pm c b'|, g)$, resulting in $\frac{(a d' \pm c b') / h}{(b' d) / h}$.
- `mul` (Multiplication):
  - *Implementation Details:* Cross-reduction removes common factors before multiplication: $g_1 = \gcd(|a|, d)$ and $g_2 = \gcd(|c|, b)$. Result: $\frac{(a/g_1) \times (c/g_2)}{(b/g_2) \times (d/g_1)}$.
- `div` (Division):
  - *Implementation Details:* $\frac{a}{b} \div \frac{c}{d} = \frac{a}{b} \times \frac{d}{c}$. Uses the same cross-reduction. The sign of $c$ transfers to the numerator.
- `neg` (Negation):
  - *Implementation Details:* Negates the numerator.
- `square` (Square):
  - *Implementation Details:* $\frac{a^2}{b^2}$. Natively reduced since $\gcd(|a|, b) = 1 \implies \gcd(a^2, b^2) = 1$.
- `pow` (Exponentiation):
  - *Implementation Details:* $\frac{a^k}{b^k}$ remains reduced. The exponent is `i32`; its unsigned magnitude handles `i32::MIN` without signed negation. Negative exponents invert the fraction before exponentiation. Zero to a negative exponent is rejected. Exponent type alone does not bound allocation.
- `try_pow` (Safe Exponentiation with large exponents):
  - *Implementation Details:* Takes an `i64` exponent and returns `Result<Self, RationalError>`. Checked size arithmetic bounds workspace growth. A conservative bit-size estimate may reject an explicit allocation ceiling; it must not reject a coefficient precision limit when a tighter exact fit remains possible. Allocation failures are recoverable only if the implementation uses fallible allocation.
- `pow_assign` / `square_assign` / `recip_assign`:
  - *Implementation Details:* In-place variants for standard mutators.
- `mul_add` (Fused multiply-add):
  - *Implementation Details:* `self * b + c`.
- `mediant` (Mediant of two rationals):
  - *Implementation Details:* $\frac{a+c}{b+d}$. Mathematically between the two fractions. Calls `reduce()` at the end, since the mediant is not guaranteed to be in lowest terms (e.g. $\text{mediant}(\frac{1}{3}, \frac{1}{3}) = \frac{2}{6}$).
- `try_*` (Precision variants):
  - *Implementation Details:* Returns `Result<Self, RationalError>` for precision boundary or allocation failures.
- `checked_*` (Arithmetic variants):
  - *Implementation Details:* Returns `None` for domain errors or a final canonical result exceeding coefficient precision. Intermediate coefficient overflow is evaluated in widening scratch, matching the integer checked-arithmetic policy at the final boundary.

- `recip` / `checked_recip` (Reciprocal):
  - *Implementation Details:* Swap `numer` and `denom`. If `numer` was negative, transfer `-` to new `numer`. `recip` panics if `numer == 0` (like std integer division by zero). `checked_recip` returns `Option<Self>`.

### 3. Rounding & Truncation
- `round`: rounds half away from zero, matching `f64::round`.
- `round_ties_even`: bankers rounding, matching `f64::round_ties_even`.
- `floor` / `ceil` / `trunc`: Delegate to division semantics.
- `fract` (Fractional part): Uses truncating division semantics, not Euclidean semantics.
- `continued_fraction`: Returns a finite iterator yielding coefficients. Uses floor-division semantics.
- `from_continued_fraction(coeffs: &[MpInt])`: Inverse of `continued_fraction`. Coefficients after `coeffs[0]` must be strictly positive. If any $a_i \le 0$ for $i > 0$, returns `RationalError::InvalidFormat`.

### 4. Approximations & Advanced Math
- `best_approximation(max_denom)`
- `best_approximation_with_max_error(error)`
- `lower_approximation(max_denom)`
- `upper_approximation(max_denom)`
- `limit_denominator(max_denom)`: Alias for best approximation within bounds.
- `rational_reconstruction(residue: &MpUint, modulus: &MpUint, numer_bound: &MpUint, denom_bound: &MpUint) -> Result<Self, RationalError>`: Reconstructs a rational number from modular data. **Precondition:** `2 * numer_bound * denom_bound < modulus` must hold for a unique reconstruction.
- `farey_neighbors(max_denom)`
- `stern_brocot_path()` / `from_stern_brocot_path(path)`
- `egyptian_fraction()` (Optional `cas` feature).

### 5. Properties & Math
- `abs` / `abs_assign` / `signum`.
- `is_zero`: checks `numer.is_zero()`.
- `is_one`: checks `numer == 1` and `denom == 1`.
- `is_positive`: checks `numer > 0`.
- `is_negative`: checks `numer < 0`.
- `is_integer`: True if `denom == 1`.
- `is_dyadic` / `is_terminating_in_base(b)`.
- `conditional_neg(condition)`: Negates the numerator when the condition is true. No constant-time execution contract is implied.

### 6. Conversions, Parsing & Formatting
- `from_str_radix` / `to_string_radix`: 
  - For `from_str_radix(radix)`, scientific notation is only accepted for radix 10 by default.
  - For non-decimal radices: use fraction syntax (`a/b`), point syntax (`a.b`), and scientific notation requires an explicit parser option or distinct exponent marker.
  - If a string in base $r$ has integer part $I$ and fractional part with $k$ digits representing $F$, then the value is $I + \frac{F}{r^k}$.
- `Float Conversions`:
  ```rust
  from_f64_exact(value: f64) -> Result<Self, FloatConversionError>
  from_f32_exact(value: f32) -> Result<Self, FloatConversionError>
  from_f64_approx(value: f64, max_denom: u64) -> Result<Self, FloatConversionError>
  from_f64_within_tolerance(value: f64, tol: MpRational) -> Result<Self, FloatConversionError>
  to_f64() -> Option<f64>
  to_f64_lossy() -> f64
  ```
  Decode a float bit pattern with `from_f64_exact(f64::from_bits(bits))`.
  The current integer API exposes no `from_f64_bits` method.
  - `to_f64` follows `num_traits::ToPrimitive` expectations. Returns `None` if the exact value overflows `f64::MAX` to infinity.
- **Cross-Type Conversions**:
  - `From<MpInt>` and `From<MpUint>` to `MpRational` (infallible, `denom=1`).
  - `From<i32>`, `From<i64>`, etc. (infallible).
- **Formatting (`core::fmt`)**:
  - `Display`: exact `a/b`, or `a` when denominator is 1.
  - `Binary`, `Octal`, `LowerHex`, `UpperHex`: exact `numer/denom` formatting in the selected radix.
  - `LowerExp` and `UpperExp`: Planned rounded decimal formatting controlled by `Formatter::precision`, using base 10 with `floor(log10(|self|))` as the exponent for nonzero values. Zero uses exponent zero.

### 7. Iterators
- `to_radix_fractional(base, max_digits)`: Returns a bounded `Vec<u32>` of fractional digits.
- `fractional_digits(base)`: Returns a potentially infinite `Iterator<Item = u32>` of fractional digits.
- `digits_with_period(base: u32) -> Result<(Vec<u32>, Vec<u32>), RationalError>` (requires base >= 2).

### 8. Comparisons & Equality
- `cmp` / `eq` / `cmp_abs` / `abs_diff` / `min` / `max` / `clamp`.
- When cross-comparing with `f64`, converts `f64` strictly to exact `MpRational` via `from_f64_exact` to avoid false positives.

### 9. Memory Management
- `clone_from` / `swap`.
- `capacity(&self) -> (usize, usize)`: Returns tuple of (numer_limbs, denom_limbs).
- `reserve(numer_limbs: usize, denom_limbs: usize)`
- `shrink_to_fit()`.

### 10. Ecosystem & Randomness
- **Hash Guarantees**: Equal canonical rationals feed equal component data into the same hasher. Standard `Hash` does not promise a portable byte encoding or cross-platform hash output; a portable digest requires a specified canonical serialization.
- **Cross-Type Contract**: `MpRational` is the exact middle layer of the numeric tower (`int -> rational -> float`). It must preserve exactness when receiving integers and must yield to `MpFloat` when mixed with approximate floating values.
- **Serialization**: Binary serialization is deterministic only for the crate-defined canonical serialization format, not necessarily for arbitrary `serde` backends.
- **Randomness (`rand`)**:
  - No `Standard` distribution is implemented.
  - Planned constructors are `random_with_denominator_bits`, `random_with_max_denominator`, and `random_canonical_in_range`. A uniform distribution requires a finite interval and denominator bound, a caller-supplied RNG, and a specified measure over canonical fractions. Precision metadata alone does not define the distribution.

## 11. Errors
```rust
pub enum RationalError {
    DenominatorZero,
    NegativeDenominator,
    PrecisionExceeded,
    DivisionByZero,
    NegativeExponentOfZero,
    InvalidRadix,
    InvalidDigit,
    InvalidFormat,
    NonFiniteFloat,
    NegativeTolerance,
    MaxDenominatorZero,
    AllocationRequired,
    TooLarge,
    NonCanonical,
    IntegerError(MpError),
}
```
*Note: Operations that propagate integer domain/precision errors wrap them in `RationalError::IntegerError`.*

