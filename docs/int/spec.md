# Mp integer crate specification

This is the crate's design specification, including future APIs and integrations.
The [API inventory](api-inventory.md) describes the implemented public surface
and current behavior. A design requirement here does not imply that its method,
feature, or behavioral contract is already implemented. Sections explicitly
marked **planned** describe future work.

Rust's standard integer API is the compatibility target wherever a finite-width
integer has an equivalent operation. Unlimited precision, per-value widths,
ambient construction, and fused destination APIs require additional contracts.
Differences must be explicit rather than hidden behind familiar method names.

## 1. Precision and Interoperability Goals
Values carry precision metadata so arithmetic can resolve result policy from
its operands without separate precision arguments at each call.

Primary goal:
- Users set precision policy once, through an explicit value, scoped context, or global default.
- After that, `MpInt`, `MpUint`, `MpRational`, and `MpFloat` carry enough precision metadata for normal arithmetic, conversions, parsing, formatting, and generic APIs to behave predictably.
- Passing an Mp value into a function must not require the callee to separately receive precision unless that function is explicitly constructing new independent values or requesting a different precision.

Interoperability goal:
- Mp numeric types should participate in Rust's standard numeric ecosystem as far as soundly possible: `From`, `TryFrom`, `FromStr`, `Display`, `core::ops`, iterator `Sum`/`Product`, `num-traits`, `serde`, `rand`, and cross-type comparisons/conversions.
- Generic numeric functions should accept Mp types whenever their trait contracts can be satisfied.
- The crate should prefer trait compatibility over bespoke APIs, but must not implement a trait whose semantic contract is impossible for unlimited or ambient-precision values.
- Rust has no implicit argument conversion. A function declared as `fn f(x: f64)` will not accept `MpFloat` automatically. The guarantee targets generic APIs (`T: Num`, `T: Into<MpFloat>`, `T: TryInto<MpInt>`, crate umbrella traits, etc.) and explicit conversions at concrete primitive boundaries.

Design tension:
- Ambient precision supplies a construction policy; explicit `_with_precision` and `try_*` APIs provide local control.
- Ambient precision is convenience policy, not hidden mutation: existing values keep their precision metadata, and operations resolve precision from operands before consulting context/global defaults for newly-created values.

## 2. Numeric Tower & Cross-Type Contracts
**Planned numeric tower:** the package currently exports `MpUint` and `MpInt`.
Rational/float types and mixed-type arithmetic are design targets:

```text
MpUint / MpInt -> MpRational -> MpFloat
```

- Integer with integer returns an integer type when the operation is closed over that type.
- Integer with rational promotes to `MpRational`.
- Any operation involving `MpFloat` promotes to `MpFloat`.
- Primitive integers promote into the matching Mp integer type unless the other operand is rational or float.
- `f32` and `f64` promote to `MpFloat` by decoding the exact IEEE-754 value first, then rounding to the resolved MpFloat target precision if necessary.
- Rust has no implicit conversion at concrete argument boundaries; this promotion policy applies to Mp operator impls, constructors, explicit conversions, and generic helper traits.

Precision combination is type-specific:
- Integer and rational coefficient precision combines bounded widths as `max(width_a, width_b)`. This chooses the result width; it does not widen further when a bounded arithmetic result overflows.
- Float precision narrows to the lower trusted target precision in mixed-float operations, because extra result bits would imply accuracy the lower-precision operand did not carry.
- Conversion from exact values (`MpInt`, `MpUint`, `MpRational`) into `MpFloat` uses the active float target precision and rounding mode.

## 3. Precision System

`MpInt` and `MpUint` support unlimited and bounded precision modes.

### 3.1 Modes (Unlimited / Bounded)
1. **Unlimited precision**:
   - Signed arithmetic grows as needed, limited only by allocation failure.
   - Unsigned addition and multiplication grow as needed.
   - Unsigned subtraction below zero is an underflow, not a precision overflow.
   - Width-dependent operations require either bounded precision or an explicit width.

2. **Bounded precision**:
   - Values are constrained to an explicit bit width N in `1..usize::MAX`.
   - `MpUint` behaves as an unsigned N-bit integer ($0 \le x \le 2^N - 1$).
   - `MpInt` behaves as a signed N-bit two's-complement integer at the public API boundary ($-2^{N-1} \le x \le 2^{N-1} - 1$).
   - Internally, signed values may use sign-magnitude representation, but observable behaviour must match Rust integer semantics.

Precision resolution has two distinct roles:

1. **Value precision metadata:**
   Existing Mp values carry their own precision. Integer and rational arithmetic derives result precision from operand metadata. Ambient precision does not cap or rewrite results of operations on existing values.

2. **Ambient construction target:**
   When constructing a new value without explicit precision, the active context/global precision supplies a target precision. For exact primitive construction, this target acts as a **floor**: 300u16 under ambient `Bounded(8)` produces `Bounded(9)`. Primitive signed-to-unsigned `TryFrom` rejects negative values but also widens for exactness; fallibility alone does not imply a precision cap. String parsing enforces the bounded ambient width. Explicit checked precision constructors enforce their requested width.

### 3.2 Operation Precision
Resolution order for ambient construction:
1. Scoped context precision
2. Global default precision
3. Unlimited

Explicit `_with_precision` APIs bypass ambient resolution.

For bounded binary operations:
- Non-assigning bounded/bounded arithmetic produces `Bounded(max(width_a, width_b))`.
- Bounded/unlimited arithmetic produces `Unlimited`.
- Assignment operators and fused `assign_*` preserve destination precision exactly. They compute the mathematical result from the operand values and validate it against the destination width before committing. Bounded overflow leaves the destination unchanged.
- `clone_from` copies source precision and `swap` exchanges precision with the value. Explicit-width bit operations return the requested bounded width.

### 3.3 Creation Semantics
- **`From<T>` is infallible and exact:** Converts to the ambient target width. For exact construction under ambient `Bounded(N)`, the result width is `max(N, required_bits(value))`. With no ambient precision, it produces `Unlimited`.
  - `required_unsigned_bits(x)`: `floor(log2(x)) + 1` (or `1` for `0`, as zero-width bounded integers are not representable).
  - `required_signed_bits(x)`: The minimum bounded signed width satisfying `-2^(N-1) <= x <= 2^(N-1) - 1`. Specifically: `0` and `-1` need `1` bit; `1` and `-2` need `2` bits; `127` and `-128` need `8` bits, etc.
- **`Default` and `Zero::zero()` are stable:** They always return an `Unlimited` zero.
- **Iterator `Sum` and `Product`:** This crate folds from unlimited zero and one, respectively, and therefore returns unlimited results even for bounded items or empty iterators. Rust's traits do not prescribe this precision policy. Bounded accumulation uses an explicit bounded fold.
- **Planned allocation policy:** A memory ceiling is separate from precision. No `AllocationPolicy` type or `no_alloc` mode is implemented.

### 3.4 Context & Global
- **Context (Scoped Closure)**: 
  - *With `std` feature:* A `std::thread_local!` stack scopes synchronous closure execution through `PrecisionContext::with_bounded(256, || { ... })`. No RAII guard is exposed.
  - *Without `std`:* Thread-local scoped contexts are unavailable. The crate still uses `alloc`; `no_std` is not allocation-free.
  - The scope covers synchronous closure execution and unwinding. Returning a future does not carry the context into subsequent polling.
- **Global**: `PrecisionContext::set_global(p)` exists only with `target_has_atomic = "ptr"`, independently of `std`. It returns the previous setting and affects subsequent ambient-aware construction without an active scoped context. The encoding uses `0 = Unset`, `usize::MAX = Unlimited`, and other nonzero values for bounded widths. Targets without pointer-width atomics have no mutable global fallback; without `std`, `active()` is const and returns `Unset`.
- **Internal Representation**: `BoundedPrecision` is an opaque validated width in `1..usize::MAX`. Both `AmbientPrecision::Bounded` and `Precision::Bounded` contain that type, making zero and the unlimited sentinel unrepresentable as bounded states.

## 4. Error Model (MpError)
```rust
pub enum MpError {
    Overflow, Underflow, DivisionByZero, NegativeRoot, EvenModulusUnsupported,
    ModulusZero, NoInverse, NoPrimitiveRoot, NotCoprime, WidthRequired,
    PrecisionRequired, PrecisionMismatch, PrecisionExceeded, ShiftTooLarge, 
    AllocationRequired, InvalidRadix, InvalidDigit, FactorizationRequired,
    NonCanonical, EmptyInput, NonPositiveInput, NegativeInput, InvalidModulus, 
    NonCyclicGroup, NotInGeneratedSubgroup, InvalidInput, IntegerConversionLoss, EmptySlice
}
// Note: `Overflow` means arithmetic result exceeded destination precision/range. 
// `PrecisionExceeded`: The requested explicit bounded constructor width is insufficient.
// `AllocationRequired`: Reserved for allocation-policy failures; no no_alloc mode exists.
// `EmptyInput`: For parsing empty strings.
// `EmptySlice`: For empty slice operations.
// `InvalidInput`: Generic input error for parsing/construction failures not covered above.
```
`MpError` is non-exhaustive. Not every variant is produced by an implemented
public API. Parsing uses `ParseMpIntError` / `ParseMpUintError`, whose `kind()`
methods return references to their classifications. Both kind enums are
non-exhaustive and reexported at the crate root. `TooLarge` is a parse classification,
not an `MpError` variant. Public errors implement `Display` and `Debug`, plus
`core::error::Error` with `std`.

## 5. Primitive Parity & Safe Defaulting
Bounded operations in Mp preserve mathematical correctness as strictly as possible:
- Standard operators (`+`, `-`, `*`) always panic on overflow in both debug and release modes.
- Assigning operators (`+=`, `-=`, `*=`) similarly panic on overflow or precision limits.
- Wrapping wrapper types are planned; named `wrapping_*` methods provide the implemented opt-in surface. No `primitive-overflow-semantics` feature exists. Always-panicking ordinary arithmetic is an intentional difference from primitives compiled with overflow checks disabled.

The library mirrors Rust primitive integer method families wherever the semantics make sense:

- Plain methods/operators: ergonomic, panicking on invalid bounded results, division by zero, missing width in width-dependent operator paths, and impossible unsigned underflow.
- `checked_* -> Option<Output>`: reports the operation's checked failure condition. It may allocate, panic on address-space limits, or encounter allocator failure; `Option` does not promise recoverable allocation. Any future allocation-constrained mode must specify its additional failures explicitly.
- `try_* -> Result<Output, MpError>`: failures needing explanation (`WidthRequired`, `AllocationRequired`, `ShiftTooLarge`, `FactorizationRequired`, `InvalidRadix`, etc.).
- `wrapping_*`, `overflowing_*`, `saturating_*`, `strict_*`, and `unchecked_*`: follow primitive integer naming and tuple shapes. Because Mp bounded operators already panic on overflow in all build modes, `strict_*` methods are mostly explicit aliases for plain checked-then-panic arithmetic. They are provided for primitive API parity and for codebases that want overflow intent visible at call sites.
- Cross signed/unsigned variants should use primitive-style names where possible: `checked_add_signed`, `checked_sub_unsigned`, `wrapping_add_signed`, `overflowing_add_signed`, `strict_add_signed`, etc.

For bounded values:
- `wrapping_*` always wraps within the active two's-complement width.
- `checked_*` returns `Option<Output>`.
- `overflowing_*` always returns `(wrapped_result, overflowed)`.
- `saturating_*` always saturates to the minimum or maximum boundary.
- For always-wrapping ergonomic environments, explicitly use wrappers like `WrappingMpUint(pub MpUint)` or `WrappingMpInt(pub MpInt)`.
- **Planned** `unchecked_*` are `unsafe fn`, with operation-specific preconditions matching primitive contracts. Each requires an unsafe API review; no unchecked inherent arithmetic is currently exported.

**Standard-library compatibility requirements** (implementation status is in
the inventory):

| Family | Required bounded semantics |
| --- | --- |
| Ordinary shifts and shift assignments | Reject counts at least the width; a valid-count left shift retains the low N bits, and signed right shift extends the sign. Ordinary shifts must be distinguished from exact multiplication by a power of two. |
| Wrapping, overflowing, saturating division | A zero divisor panics; wrapping/overflowing applies only to representational overflow, such as signed `MIN / -1`. |
| Wrapping/overflowing remainder | A zero divisor panics; signed `MIN % -1` wraps to zero and overflowing remainder reports `true`. |
| `checked_shl` / planned `checked_shr` | Return `None` when the count is at least the width. Left-shift bit loss within a valid count does not make the checked result fail. |
| `wrapping_shl` / planned `wrapping_shr` | Reduce the count modulo the width, then shift within that width. Modulo generalizes the primitive power-of-two-width count mask to arbitrary widths. |
| `overflowing_shl` / planned `overflowing_shr` | Return the wrapping shift result and a flag for count >= width, not for shifted-out value bits. |
| `trailing_zeros(0)` | Return the bounded width, as primitive integers do. Unlimited zero retains the crate convention of zero because no finite width exists. |

These rules follow Rust's [checked shifts](https://doc.rust-lang.org/std/primitive.u32.html#method.checked_shl),
[wrapping shifts](https://doc.rust-lang.org/std/primitive.u32.html#method.wrapping_shl), and
[wrapping division](https://doc.rust-lang.org/std/primitive.i32.html#method.wrapping_div).
`saturating_shl`, `try_shl`, and `saturating_rem` have no direct primitive
counterparts. The shift extensions retain mathematical multiplication by a
power of two: `try_shl` reports a bounded fit failure and `saturating_shl`
clamps an out-of-range result. The remainder extension must panic on a zero
divisor and otherwise return the representable remainder. `mul_2exp` remains
exact mathematical multiplication with a bounded fit check. Unlimited shifts
cannot mask counts by a nonexistent finite width.

For unlimited values:
- Standard arithmetic operators grow to fit. `MpUint` subtraction panics on underflow.
- `wrapping_add`, `wrapping_mul`, and signed `wrapping_sub` are equivalent to normal arithmetic.
- Width-dependent wrapping variants should use `try_*` or `_with_width(bits)` APIs.
- `checked_sub` on `MpUint` returns `None` under zero; `saturating_sub` clamps to zero.

## 6. Type Definitions & Internal Layout
### 6.1 Core Layout
- **Types**: `InternalMpUint` (Magnitude) and `InternalMpInt` (Signed)
- **Precision Enum**: `#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)] pub enum Precision { Unlimited, Bounded(BoundedPrecision) }`.
- **Implementation Strategy**:
  - **Storage**: Small-Inline Representation (`enum UintRepr { Inline { len: u8, limbs: [usize; 4] }, Heap(alloc::vec::Vec<usize>) }`). The limb count is fixed at four; inline bit capacity is 64, 128, or 256 bits on 16-, 32-, or 64-bit targets. Both `MpUint` and `MpInt` are inherently `Send + Sync`.
  - **Limbs**: native arithmetic limbs are `usize` (`Limb = usize`). Stable wire formats must convert explicitly instead of exposing native limb width accidentally.
  - **Sign**: `InternalMpInt` stores a positive-sign flag beside the unsigned magnitude.
  - **Canonical Zero**: `InternalMpInt` normalizes zero to positive (`abs.is_zero() ==> is_positive == true`). A nonzero `InternalMpUint` ends in a nonzero most-significant limb; zero uses an empty limb slice.
  - **Value identity**: `MpUint::with_precision_checked(5_u8, BoundedPrecision::new(8).unwrap()).unwrap()` and `MpUint::from(5_u8)` compare equal and hash equally within `MpUint`. Precision is not part of key identity.

### 6.2 Limits & Constants
- `zero()` and `one()`, plus signed `minus_one()`, are implemented constructors. `zero_with_precision` is const. Public `ZERO`, `ONE`, `TWO`, and `TEN` constants and `two()`/`ten()` constructors are planned, not exported.
- `num_traits::Bounded`: **Not implemented.** A fixed bound cannot represent unlimited precision. Bounded operations query explicit bounds via `Self::max_for_precision(bits)` and `Self::min_for_precision(bits)`.
- Bounded signed invariant: A bounded `MpInt` strictly enforces $-2^{N-1} \le x \le 2^{N-1} - 1$. The edge case $N = 1$ enforces $-1 \le x \le 0$.

## 7. Method Reference

This section includes the design surface. The inventory is the authority for
which signatures are currently implemented; Sections 7's advanced methods,
Section 9's integrations, and the mixed numeric tower are not implied exports.

> [!IMPORTANT]
> **API Separation Strategy:**
> - **Only on `MpInt` (Signed):** `abs`, `signum`, `is_negative`, `is_minus_one`.
> - **Omitted from `MpUint`:** `is_positive`, `is_negative`. Signed `is_positive` means `self > 0`. Unsigned square root is `isqrt`; there is no `sqrt` method. Signed `checked_isqrt` rejects negatives.
> - **Shared / Delegated:** Signed number theory often uses magnitude, but domain behavior is method-specific: nonpositive signed primality returns `false`, totient returns `None`, and Jacobi returns `None` for a nonpositive or even denominator. Jacobi retains the numerator's sign. These APIs do not return `MpError::NonPositiveInput`.

### Core Arithmetic
- `add`, `sub`, `neg`, `mul`, `mul_add`, `square`.
- `pow(exp: u32)`, `checked_pow(exp: u32)`, `try_pow(exp: u32)`: the mathematical value of `0^0` is one, subject to result precision. Signed `Bounded(1)` cannot represent positive one.
- `MpUint::isqrt`: Floor of the unsigned square root. It already returns `Option`, so an identical unsigned `checked_isqrt` alias is intentionally omitted. `MpInt::checked_isqrt` remains meaningful because negative inputs return `None`.
- `div_rem`, `div_euclid`, `rem_euclid`, `div_trunc`, `rem_trunc`, `div_floor`, `mod_floor`, `div_ceil` satisfy `a = q*b+r`. Truncation gives a nonzero remainder the dividend's sign; floor gives it the divisor's sign; Euclidean remainder is in `0..|b|`. Signed `MIN / -1` overflows only when MIN is the minimum of the **resolved** bounded width. An unlimited RHS makes the combined precision unlimited.
- `is_divisor_of` / `is_divisible_by` (Divisibility predicate).
- `abs_diff` (Absolute difference): Returns `MpUint`. Result precision is `Bounded(max(width_a, width_b))` for both unsigned and signed bounded inputs (as the maximum absolute difference between two signed $N$-bit integers in $[-2^{N-1}, 2^{N-1}-1]$ is $2^N - 1$, which fits exactly in an unsigned $N$-bit integer without overflow).
- `abs_sub` (Positive difference, `MpInt` only): Returns `max(0, self - other)` as an `MpInt`, matching `num_traits::Signed`. For example, `(-5).abs_sub(3) == 0` and `(-5).abs_diff(3) == 8`.
- `div_2exp` is a right shift: signed negatives round toward negative infinity, so `-3.div_2exp(1)` means -2. `mul_2exp` is exact multiplication. Planned `mod_floor_2exp` is nonnegative and planned `rem_trunc_2exp` carries the dividend's sign.
- `widening_mul`, `carrying_mul`, `carrying_mul_add`: Return `(Self, Self)` representing `(lower, upper)` half-words. For bounded operands of width $W$, the exact result is partitioned at bit $W$ into a lower word and an upper carry word. For unlimited operands, `lower` holds the exact full product/sum, and `upper` is zero.
  For signed words, reconstruction uses the unsigned W-bit encoding of `lower`
  plus signed `upper * 2^W`; adding a negative lower word directly is incorrect.
  `try_widening_mul` and `try_carrying_mul` reject unlimited combined precision
  with `WidthRequired` rather than returning the unlimited fallback pair.
- `try_widening_mul`, `try_carrying_mul`: Return `Result<(Self, Self), MpError>`. On unlimited operands, return `MpError::WidthRequired` rather than panicking.
- `mul_add`: Computes `(self * a) + b` as a single operation without intermediate capacity limitations.
- `midpoint`: Computed without intermediate overflow. For odd sums, rounding follows Rust primitive integer midpoint semantics: `(a + b) / 2` rounded toward zero (e.g. `(-1).midpoint(0) == 0`). Result precision is `max(width_a, width_b)`.

#### Destination-Reusing Assignment
- `assign_add(&mut self, a, b)`, `assign_sub(&mut self, a, b)`, `assign_mul(&mut self, a, b)`, `assign_square(&mut self, a)`: Write `a op b` into `self`, reusing the existing allocation instead of returning a fresh value.
- Destination reuse can avoid output allocation. It does not promise zero allocation: growth, scratch, or algorithm-specific work can allocate. Owned `core::ops` operands can also reuse their buffers.
- **Precision**: assignment preserves `self.precision`, matching `+=` and the other assignment operators. Operand precision does not constrain the fused result.
- **Transactional Rollback**: Under bounded precision (`BoundedPrecision`), mutating assignment operations (`assign_add`, `assign_sub`, `assign_mul`, `assign_square`) do not expose unvalidated intermediate residues if an overflow occurs. Operations validate before receiver mutation or evaluate in temporary scratch.
- `assign_sub` on `MpUint` returns `true` on underflow and leaves the destination unchanged; success returns `false`. This reporting signature is retained. The `MpInt` form returns `()`.

### Sign & Properties
- `abs`, `abs_assign`, `unsigned_abs`, `signum`, `apply_sign`. `signum(0)` returns an `MpInt` with value 0 in canonical zero representation (`abs.is_zero() ==> is_positive == true`), preserving its input precision.
- `is_zero`, `is_one`.
- `is_positive`, `is_negative`, `is_minus_one` (Only on `MpInt`).
- `is_even`, `is_odd`.
- `is_power_of_two`, `next_power_of_two`, `checked_next_power_of_two`.
- `significant_bits`: Number of bits representing the magnitude; `significant_bits(0) == 0`. The `required_*_bits` names below describe internal storage calculations, not public inherent methods.

| Function | `0` | `1` | `-1` (signed) |
|---|---|---|---|
| `significant_bits` | 0 | 1 | 1 |
| `required_unsigned_bits` | 1 | 1 | n/a |
| `required_signed_bits` | 1 | 2 | 1 |

- `same_precision`, `same_value_and_precision`, `bit_identical`: Helpers for exact encoding matches where `Eq` is insufficient.
- `pub const fn precision(&self) -> Precision` exposes the per-value policy without parsing debug output. The precision identity helpers above are planned.
- `cast_signed`, `cast_unsigned`: Primitive parity for reinterpreting two's-complement bits without modifying them (bounded only).

### Bitwise Operations & Counting
For bounded signed bitwise operations, `MpInt` is first interpreted as an N-bit two's-complement value, operated on, then converted back into canonical sign-magnitude form.

**Current behavior matrix** (bounded zero trailing counts differ from the
primitive-compatibility target in Section 5):
| Operation | `MpUint` unlimited | `MpInt` positive unlimited | `MpInt` negative unlimited | Bounded |
|---|---|---|---|---|
| `not` | `WidthRequired` | Defined (infinite two's complement) | Defined | Defined |
| `count_ones` | Absolute | Finite | `None` (width-dependent) | Defined |
| `count_zeros` | `None` (width-dependent) | `None` (width-dependent) | `None` (width-dependent) | Defined |
| `leading_zeros/ones` | `None` (width-dependent) | `None` (width-dependent) | `None` (width-dependent) | Defined |
| `trailing_zeros` | Defined (0 on 0) | Defined (0 on 0) | Defined (0 on 0) | Defined (0 on 0) |
| `trailing_ones` | Defined | Defined | `None` (width-dependent) | Defined |

- `not`: `Not` trait panics on `MpUint` unlimited since it requires width. It is fully defined on `MpInt` unlimited, where the sign supplies the infinite extension.
  - `not_with_width(bits) -> Option<Self>`: complement within an explicit width. `None` for zero or `usize::MAX`. Accepts unlimited inputs and returns the requested bounded precision.
  - `try_not() -> Result<Self, MpError>`: complement within the value's *own* bounded precision. `MpError::WidthRequired` when the precision is unlimited.
- `bitand`, `bitor`, `bitxor`, `bitand_assign`, `bitor_assign`, `bitxor_assign` (Provided via standard `core::ops` traits to avoid confusing inherent method name collisions).
- `shl`, `shr`, `shl_assign`, `shr_assign`: For `MpInt` unlimited, arithmetic `shr` on negatives preserves sign and extends infinitely (i.e. `-1 >> n == -1`). For massive shifts without allocation panic, use `try_shl_big(&huge_shift)` / `try_shr_big`.
- `rotate_left`, `rotate_right`, `reverse_bits`: Accept an explicit valid width even for unlimited input and return that bounded precision. `swap_bytes` on `MpUint` uses the bounded width or significant byte span; on `MpInt` it returns `None` for unlimited precision.
- `count_ones`, `count_zeros`, `trailing_zeros`, `trailing_ones`, `leading_zeros`, `leading_ones`: current behavior follows the matrix. `trailing_zeros(0)` currently returns zero even when bounded; the design target is the bounded width. Bounded `leading_zeros` and `leading_ones` count against the configured width.
- `find_first_set_bit`, `find_first_zero_bit`, `find_next_set_bit(from: usize)`, `find_next_zero_bit(from: usize)` (Note: for unlimited `MpUint`, `find_next_zero_bit` returns `from` if `from` is past the highest set bit).
  Next-bit scans include `from` itself. Current unsigned zero-bit scans include
  the zero extension beyond bounded width. Signed zero-bit scans use width
  (or `usize::MAX` when unlimited) as their no-match sentinel; the first-zero
  method converts that sentinel to `None`.
- `get_bit`, `set_bit_to`, `set_bit`, `clear_bit`, `toggle_bit`, `test_bit` (Note: `set_bit`, `clear_bit`, `toggle_bit`, `test_bit` are convenience wrappers around `get_bit`/`set_bit_to`).
  Single-bit updates return new values. Bounded reads above the width are false
  and bounded updates there leave the value unchanged. Signed `bit_range`
  returns a nonnegative extraction and can widen for an additional sign bit.
- `bit_range`, `set_bit_range`, `take_lowest_one_bit`, `take_highest_one_bit`.

### Number Theory & Advanced Math
- `gcd`, `lcm`: Edge cases: `gcd(0, 0) = 0`, `lcm(0, b) = 0`, `lcm(0, 0) = 0`. Signed results are nonnegative; `gcd` panics when that value exceeds the resolved precision, and `lcm` / `gcd_lcm` return `None` on overflow.
- `gcd_lcm(a, b)`.
- `gcd_slice(values)`: `gcd_slice([]) = 0`.
- `batch_shared_factor_detection`.
- `is_coprime(other)`.
- `extended_gcd`: signed results satisfy the Bézout identity subject to result precision. The current unsigned form returns modular coefficients: for nonzero inputs, `a*x = gcd(a,b) (mod b)` and `b*y = gcd(a,b) (mod a)`; a zero second operand returns `None`. `extended_gcd_cofactors` is planned.
- `divisors`, `divisor_count`, `divisor_sum`.
- `is_smooth(b)`: Fast check if all prime factors are $\le b$.
- `remove_factor`: Base 0 or 1 returns `MpError::InvalidInput`.
- `euler_phi`, `carmichael_lambda`, `moebius_mu`.
- `chinese_remainder`.
- `add_mod`, `sub_mod`, `mul_mod`, `pow_mod`: current signed modular operations use operand magnitudes. A negative exponent requests inversion of the base magnitude. These magnitude semantics are distinct from signed Euclidean residue arithmetic.
- `montgomery_mul`, `barrett_reduce`: no implicit timing guarantee. Montgomery multiplication returns `a*b*R^-1 mod m` for odd nonzero `m`, with `R = 2^(usize::BITS * modulus_limb_count)`. Its result can depend on pointer width; it is not ordinary `mul_mod`.
- `invert`.
- `multiplicative_order`, `primitive_root`, `discrete_log` *(planned, including an `experimental` feature gate)*.
- `is_prime`: Deterministic primality test for inputs $\le 2^{64} - 1$ (via verified deterministic bases). For inputs $> 2^{64} - 1$, executes the Baillie-PSW test (strong base-2 Miller-Rabin followed by strong Lucas-Selfridge), returning probable primality (no composite counterexamples are currently known, but it is not a certified deterministic proof).
- `is_probably_prime(k: u32)`: exact through `u64::MAX`, independently of `k`. Larger inputs use the first `clamp(k, 1, 64)` prime bases `[2, 3, 5, 7, ...]`. This is deterministic probable-prime screening, without the independent-random-round error bound.
- **Planned** `is_probably_prime_with_rng(k: u32, rng: &mut R)`: Miller-Rabin with random bases drawn from the caller's RNG. Reproducibility depends on RNG state.
- `next_prime` returns the least prime strictly greater than the input, subject to result precision. `prev_prime` returns the greatest positive prime strictly less than the input and returns `None` for inputs $\le 2$. Both preserve operand precision; candidates above `u64::MAX` use probable primality.
- **Planned** `factor` / `prime_factors`: factorization returns bases and multiplicities. Recognized factors above `u64::MAX` must be identified as probable primes unless certificates are supplied.
- `is_squarefree`, `radical`.
- `is_perfect_square`, `is_perfect_power`, `is_prime_power`.
- `jacobi_symbol` / `kronecker_symbol` / `legendre_symbol`: `jacobi_symbol(a, 1) = 1`.
- `is_congruent(b, m)`.
- `nth_root(n)`, `nth_root_rem(n)`, `sqrt_rem`, `sqrt_mod`. `nth_root` accepts the root degree explicitly. `nth_root` and `sqrt_rem` are implemented; `nth_root_rem` and `sqrt_mod` are planned.
  Current signed `nth_root`, `sqrt_rem`, and `is_perfect_square` use the
  magnitude, including for negative inputs; `checked_isqrt` rejects negatives.
- `ilog`, `ilog2`, `ilog10`, `checked_ilog*`.
- `hamming_distance`.

### Combinatorics & Special Functions
Only `factorial` is implemented in this section. All other methods and the
`combinatorics` feature are planned. Mathematical identities remain subject to
result precision, including the unrepresentable positive one in signed width one.

- `factorial(0) = 1`, `double_factorial(0) = 1`, `double_factorial(1) = 1`, `subfactorial(0) = 1`.
- `binomial`, `rising_factorial`, `falling_factorial`, `primorial`.
- `multinomial(values: &[usize])`.
- `catalan(0) = 1`.
- `fibonacci`, `lucas`.
- `stirling_first`, `stirling_second`.
- `bell`, `partition`: planned algorithms require separate correctness and performance validation. A Hardy–Ramanujan–Rademacher implementation for partitions depends on certified rounding support in the planned `MpFloat` type.
- `euler_number`.
- `bernoulli` / `harmonic_number`: planned integer methods returning `MpRational`, under a planned `combinatorics` feature.
- `tetration` / `hyperoperation`: `a^^0 = 1`.

### Constructors, Conversions, Parsing & Formatting
- **Explicit Constructor Matrix**:
  - `MpUint::new(value)`
  - `MpUint::with_precision_checked(value, bits) -> Result<Self, MpError>`
  - `MpUint::with_precision_wrapping(value, bits) -> Self`
  - `MpUint::with_precision_saturating(value, bits) -> Self`
  - `MpUint::zero_with_precision(bits) -> Self`
  - *(Same applies to `MpInt`)*
- **Planned** `from_limbs_le(limbs: &[usize])`, `from_limbs_be(limbs: &[usize])`.
- `from_str_radix`, `to_string_radix`.
- **Planned** `from_ascii`, `from_ascii_radix`.
- **Planned** `digits_in_base(b)`.
- **Planned** `to_radix_be`, `to_radix_le`, `from_radix_be`, `from_radix_le`.
- `to_be_bytes`, `to_le_bytes`, `to_native_endian_bytes`: These allocate and return `alloc::vec::Vec<u8>`.
- `write_be_bytes(buf: &mut [u8]) -> Result<(), MpError>`, `write_le_bytes`, `write_native_endian_bytes`: Non-allocating variants that write into a provided buffer (fails if buffer is too small).
- `from_be_bytes`, `from_le_bytes`, `from_native_endian_bytes`.
- `to_f64`, `to_f32`.
  Implemented float output rounds to nearest, ties to even; `None` reports
  exponent overflow. Native-endian aliases and buffer-writing methods above
  are planned. Existing byte output is minimal unsigned magnitude or signed
  two's complement; both encode zero as an empty vector. Byte formats do not
  preserve precision, and input accepts redundant zero/sign extension.
  Ambient construction should handle empty input consistently; the current
  signed empty-input path returns unlimited zero (see the inventory).

### Memory & Iterators
- `clone_from`, `swap`.
- `with_capacity(limbs)`, `reserve`, `reserve_exact`, `shrink_to_fit`, `capacity`. Capacity is measured in limbs and is independent of precision. `reserve_exact` avoids deliberate amortized growth but the allocator may provide more capacity; inline capacity is at least four limbs.
- **Planned** `bits()` (LSB-first by default), `digits(base)`, `limbs()`. Mutable raw limb access remains internal to protect normalization and bounded invariants.

## 8. Trait Implementations
- `core::ops` (`Add`, `Sub`, `Mul`, `Div`, `Rem`, `Not`, `BitAnd`, `BitOr`, `BitXor` and assignment variants; `Neg` only for signed values). Binary arithmetic and bitwise operators implement all four ownership combinations for the same Mp type. Shifts and shift assignments accept all twelve primitive integer RHS types, with checked conversion to `usize`; negative and unrepresentable counts panic. Mixed Mp/primitive arithmetic operators are planned.
- `core::iter::Sum` and `core::iter::Product` for iterator `.sum()` and `.product()`.
- `core::str::FromStr` (base 10, delegates to `from_str_radix(s, 10)`).
- `From` / `Into` / `TryFrom` for primitive integers follow the implemented matrix in the inventory. Completing owned unsigned-to-signed primitive `TryFrom` and borrowed conversion coverage is planned. **Planned** `TryFrom<MpRational>` returns `MpError::IntegerConversionLoss` on fractional parts.
- **Planned float input conversions**: `TryFrom<f32>` / `TryFrom<f64>` truncate toward zero and fail on NaN/infinity. For `MpUint`, finite negative input returns `NegativeInput` before truncation; `-0.0` maps to zero. Additional explicit policies are `try_from_f64_exact`, `_trunc`, `_floor`, `_ceil`. Current `num_traits::FromPrimitive` float defaults convert through primitive integer ranges; they do not decode arbitrary finite floats into unbounded integers.
- **Cross-Type Conversions**: `From<MpUint> for MpInt` (infallible). `TryFrom<MpInt> for MpUint` (fails with `NegativeInput` if negative).
- `Display`, `Debug`, `Binary`, `Octal`, `LowerHex`, `UpperHex`. `Debug` prints only the numeric value (like primitives). `as_debug_verbose()` provides explicit precision visibility.
- `Eq`, `PartialEq`, `Ord`, `PartialOrd`, `Hash`. `Ord` is implemented within each concrete type. Cross-type numeric comparisons are implemented exclusively through `PartialEq` and `PartialOrd` (e.g., `MpInt(5) == MpUint(5)` is `true`).
- `num_traits`: `Zero`, `One`, `Num`, `ToPrimitive`, `FromPrimitive`, plus `Signed` for `MpInt` and `Unsigned` for `MpUint`, are implemented behind `num-traits`. `num_integer::Integer` and `num_traits::Bounded` are not implemented; the latter has no fixed bounds compatible with unlimited values.
- **Conservative trait implementations**: A trait is only implemented if its semantic contract is completely satisfied.
- **Not Supported**: `bytemuck` (`Pod`, `Zeroable`) is intentionally omitted as instances are heap-allocated and variable size.

## 9. Ecosystem Integrations
**Planned:** these integrations, package names, and feature gates are design
targets. The current Cargo features are `std`, `num-traits`, `rayon`, and
`_internal-tune`, with an empty default set. No Python binding package is present.

- `serde`: The target serialized format uses fixed-width `u64` words rather than native `usize` limbs: `MpUint` serializes as `{ precision: u64, limbs_le: [u64] }` and `MpInt` as `{ precision: u64, sign: i8, limbs_le: [u64] }`. Equality/hash identity is numeric-value identity, not serialization identity.
- `arbitrary`: For fuzzing. Implementations limit the generated number of limbs or default to bounded precision to prevent Out-Of-Memory (OOM) crashes during generation.
- `rand`: Random generation APIs for bounded bit widths and ranges, with integration into `rand`'s `Distribution` / `Uniform` traits where allowed by the active `rand` version. Specific constructors include `MpUint::random_bits(rng, bits)`, `MpUint::random_below(rng, upper)`, `MpUint::random_range(rng, range)`, and `MpInt::random_range(rng, range)`.
- `pyo3`: In `mp-int-pyo3`.
- `zeroize`: Memory wiping. With feature `secure-buffer`, the internal allocation owns initialized capacity and wipes the entire reserved region before deallocation.
- `num-bigint` compatibility: `From<num_bigint::BigUint> for MpUint`, `From<num_bigint::BigInt> for MpInt`, and inverses. The separate `num-traits` integration is already implemented as described in Section 8.

## 10. Algorithms & Complexity

- `mul`: Schoolbook -> Karatsuba -> Toom-3 -> Toom-4 -> Toom-6/6.5 -> Toom-8/8.5 -> exact NTT/CRT or recursive SSA when benchmarked thresholds enable them.
- `div`: Schoolbook -> Burnikel-Ziegler -> Newton-Raphson reciprocal.
- `isqrt`: Newton-Raphson or Zimmermann.
- `gcd`: binary GCD (Stein) / Lehmer -> Stehle-Zimmermann half-GCD.
- **Planned** `factor`: Trial division -> Pollard rho -> ECM -> optional Quadratic Sieve / GNFS, subject to validation and benchmarking.
- `is_probably_prime(k: u32)`: up to 64 rounds of Miller-Rabin using the first `k` primes as bases (deterministic, reproducible). For inputs up to 64 bits, uses a proven deterministic base set unconditionally. Fixed bases do not carry the independent-random-round `(1/4)^k` error bound for larger adversarial inputs.
- **Planned** `is_probably_prime_with_rng(k, rng)`: bases are drawn from the caller's RNG.

## 11. Determinism & Platform Guarantees
For identical numeric inputs and precision, ordinary exact arithmetic returns
the same values across supported pointer widths and worker counts. Internal
limb layout, capacity, tuning, and execution time may differ. Native-width
operations need explicit qualifications: Montgomery multiplication uses a
limb-rounded radix, and `usize` conversion ranges depend on the target.

`Hash` ignores precision but feeds native limbs into the chosen hasher; neither
the encoding nor the final hash is a portable serialized representation. Signed
and unsigned types have separate hash encodings. Byte serialization has an
explicit endian contract and does not include precision. Ambient construction
also depends on active context/global policy. Current primality screening uses
fixed bases; planned RNG APIs depend on the caller's RNG state.

## 12. Edge Case Conventions
- **Math**: `gcd(0, 0) = 0`, `factorial(0) = 1`, `jacobi(a, 1) = 1`, `pow(0, 0) = 1`, subject to result precision. Planned methods use `catalan(0) = 1`, `tetration(a, 0) = 1`, and `binomial(n, k) = 0` for `k > n`.
- **Logic**: For unlimited unsigned and nonnegative signed values, a zero-bit scan starting above all significant bits returns its starting index. Negative signed values use infinite sign extension instead.

## 13. Semantic Examples

Public assignment example, independent of ambient precision:

```rust
use mp_anafis::{BoundedPrecision, MpUint, Precision};

let width = BoundedPrecision::new(8).unwrap();
let a = MpUint::with_precision_checked(200_u16, width).unwrap();
let b = MpUint::with_precision_checked(20_u16, width).unwrap();
let mut destination = MpUint::zero_with_precision(width);
destination.assign_add(&a, &b);
assert_eq!(destination.to_u64(), Some(220));
assert_eq!(destination.precision(), Precision::Bounded(width));
destination += &b;
assert_eq!(destination.to_u64(), Some(240));
assert_eq!(destination.precision(), Precision::Bounded(width));
assert!(destination.checked_add(&b).is_none());

// Unsigned underflow reports failure and preserves the destination.
assert!(destination.assign_sub(&b, &a));
assert_eq!(destination.to_u64(), Some(240));
assert_eq!(destination.precision(), Precision::Bounded(width));
```

`precision()` is available on both integer types, including in const contexts.
Tests check unchanged precision on both success and failure, ambient
construction floors, and unlimited iterator identities.

## 14. Architecture & Subsystem Boundaries

The multi-precision integer implementation is partitioned across explicit subsystem boundaries:
- `api/`: Public surface, conversions, standard trait implementations, precision policies, and transactional error boundaries.
- `logic/signed/`: `InternalMpInt` magnitude-sign arithmetic, two's-complement bitwise logic, and division rounding policies (Truncated, Floor, Ceil, Euclidean).
- `logic/unsigned/`: Raw limb arithmetic, storage representations, bitwise kernels, scratch management, and multiplication dispatch towers (Basecase -> Karatsuba -> Toom-Cook -> Schönhage-Strassen).
- `math/arch/`: Architecture-specific SIMD, assembly kernels, and CPU feature detection. Generic algorithms do not perform target-specific branching.
- `tune_api/`: Feature-gated interface (`_internal-tune`) exposing tuning parameters to external benchmarks and the `mp-tune` binary.
- Algorithmic Thresholds: Dispatch transitions are determined by empirical tuning parameters rather than hard-coded constants.
- `parallel/`: Execution-policy adapters. With the `rayon` feature, the crate initializes Rayon's *global* pool at `min(available parallelism, physical cores)` on the first operation that forks, because transform and lopsided multiplication are memory-bound and measured to regress at symmetric-multithreading widths. The narrowing is skipped when `RAYON_NUM_THREADS` is set, when the global pool already exists, and when the call runs inside an application-owned pool; it never widens and is a no-op without SMT topology. The crate never creates a private worker pool.

## 15. Internal Structure & Invariants

- **Canonical Zero**: `InternalMpInt` normalizes zero to positive (`abs.is_zero() ==> is_positive == true`). `InternalMpUint` uses an empty slice for zero and a nonzero most-significant limb for nonzero values.
- **Inline Capacity**: Fixed at 4 limbs (`INLINE_LIMBS = 4`) across all targets (`[usize; 4]`).
- **Transactional Rollback**: Under bounded precision (`BoundedPrecision`), mutating operations (`add_assign`, `sub_assign`, `assign_add`, etc.) do not expose unvalidated intermediate residues if a bounded panic occurs. Preconditions are validated before receiver mutation, or evaluated in temporary scratch.
- **Buffer Reuse & Commutativity**:
  - Commutative operations (`Add`, `Mul`, bitwise): Operands may swap to reuse larger pre-allocated buffers without post-processing.
  - Non-commutative operations (`Sub`, `Div`, `Rem`): Swapping ($A - B = -(B - A)$) is permitted only when avoiding an otherwise mandatory heap reallocation (`rhs_len > self_len`).
  - Inline storage: Capacity conditions (`rhs.capacity() > self.capacity()`) are never evaluated on inline operands; inline capacity is statically 4 for both operands.
- **Bounded Storage Invariant**: Bounded values strictly fit their defined precision.

**Invariant Testing**: Add comprehensive property tests for:
- **Representation**: equality, hashing matches equality, trailing/leading zeros, bounds fitting.
- **Arithmetic**: `checked_` agrees with ops, `wrapping_` agrees with modulo, division `a = q*b + r`.
- **Precision**: assign ops preserve `self.precision`, bounded+bounded results in `max(w1, w2)`, and exact primitive construction widens to fit. `Sum` and `Product` use unlimited identities by crate policy; bounded accumulation requires an explicit bounded fold.
- **Cross-Type**: `MpInt(-1) < MpUint(0)`, large unsigned to signed conversions.
