//! Factorial from odd prime swings and balanced products.
//!
//! For O(n)=n!/2^(n-popcount(n)), O(n)=O(floor(n/2))^2*S(n).
//! An odd prime p occurs in S(n) once for each odd floor(n/p^j).
//! Native tables stop when their exact values exceed inline storage.
//!
//! Reference: P. Luschny, "Fast Factorial Functions" (`PrimeSwing`).
//! <https://www.luschny.de/math/factorial/FastFactorialFunctions.htm>

#![expect(
    unsafe_code,
    reason = "bounded table indices, positive prime divisors, and exact carry recurrences prove unchecked access and arithmetic"
)]

use core::{
    mem::{MaybeUninit, swap},
    num::NonZeroU32,
    ops::Div,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use alloc::vec::Vec;

use super::{
    ArchKernels, DoubleLimb, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, MulScratch,
    Multiplication, ODD_COMPOSITE, Schoolbook,
};

/// Storage for native swings through the inline bit width.
const TABLE_SIZE: usize = INLINE_LIMBS * LIMB_BITS + 1;

struct FactorialTables {
    odd: [[Limb; INLINE_LIMBS]; TABLE_SIZE],
    swing: [[Limb; INLINE_LIMBS]; TABLE_SIZE],
    odd_count: usize,
    swing_count: usize,
}

const TABLES: FactorialTables = factorial_tables();

impl InternalMpUint {
    /// Computes n! by squaring odd factorial prefixes and restoring `v_2(n!)`.
    #[must_use]
    pub fn factorial(n: u32) -> Self {
        // SAFETY: popcount(n)<=n, including n=0, gives the exact valuation.
        let valuation = unsafe { n.unchecked_sub(n.count_ones()) };
        // SAFETY: both table prefixes contain at most TABLE_SIZE<=257 entries,
        // fitting u32 on every pointer width.
        let (odd_limit, swing_limit) = unsafe {
            (
                u32::try_from(TABLES.odd_count).unwrap_unchecked(),
                u32::try_from(TABLES.swing_count).unwrap_unchecked(),
            )
        };
        let mut steps = 0_u32;
        let mut head = n;
        while head >= odd_limit {
            head >>= 1;
            // SAFETY: n has at most 32 bits; each step removes one bit.
            steps = unsafe { steps.unchecked_add(1) };
        }
        // SAFETY: head<odd_limit<=TABLE_SIZE bounds the complete table row.
        let mut result = unsafe {
            Self::from_limbs_slice(
                TABLES
                    .odd
                    .get_unchecked(usize::try_from(head).unwrap_unchecked()),
            )
        };
        if steps != 0 {
            let sieve;
            let composite = if usize::try_from(n >> 4).is_ok_and(|last| last < ODD_COMPOSITE.len())
            {
                ODD_COMPOSITE
            } else {
                sieve = factorial_sieve(n);
                &sieve
            };
            let mut factors = Vec::new();
            let mut square = Self::zero();
            let mut inline_square = [MaybeUninit::uninit(); 2 * INLINE_LIMBS];
            let mut product = Self::zero();
            let mut scratch = MulScratch::default();
            for bit in (0..steps).rev() {
                let width = n >> bit;
                let swing = if width < swing_limit {
                    // SAFETY: width<swing_limit<=TABLE_SIZE bounds the row.
                    unsafe {
                        Self::from_limbs_slice(
                            TABLES
                                .swing
                                .get_unchecked(usize::try_from(width).unwrap_unchecked()),
                        )
                    }
                } else {
                    prime_swing(width, composite, &mut factors, &mut scratch)
                };
                let input = result.limbs();
                let square_limbs = if input.len() <= INLINE_LIMBS {
                    // SAFETY: table products are positive; doubling their
                    // inline width gives 2..=2*INLINE_LIMBS without overflow.
                    let length = unsafe { input.len().unchecked_mul(2) };
                    // SAFETY: length<=inline_square.len() bounds this reserved
                    // span, disjoint from the initialized positive input.
                    Schoolbook::sqr_nonempty(
                        unsafe { inline_square.get_unchecked_mut(..length) },
                        input,
                    );
                    // SAFETY: squaring filled all length slots. A normalized
                    // positive square has length or length-1 limbs, so its high
                    // guard is the only possible leading zero.
                    unsafe {
                        let initialized = from_raw_parts(inline_square.as_ptr().cast(), length);
                        let guard = *initialized.get_unchecked(length.unchecked_sub(1)) == 0;
                        initialized.get_unchecked(..length.unchecked_sub(usize::from(guard)))
                    }
                } else {
                    square.assign_square_with_scratch(&result, &mut scratch);
                    square.limbs()
                };
                // SAFETY: both materialized Limb slices have byte spans at
                // most isize::MAX; adding their element counts fits usize on
                // every supported pointer width. Allocation checks byte capacity.
                let length = unsafe { square_limbs.len().unchecked_add(swing.limbs().len()) };
                if bit == 0 {
                    // Restoring v_2(n!) adds at most ceil(valuation/LIMB_BITS)
                    // limbs. Reserve that span before writing the final odd
                    // product, so the shift cannot require a second allocation.
                    // SAFETY: the odd table covers n<=4. For n=5..8,
                    // n!>=4^valuation directly; for n>=9, 9!>4^9 and each
                    // later factor>4 give n!>=4^n>=4^valuation. Thus the odd
                    // product is >=2^valuation, so padding<=length. Each
                    // source spans at most isize::MAX bytes with >=2 bytes per
                    // limb, giving 2*length<=2*isize::MAX<usize::MAX.
                    let capacity = unsafe {
                        let padding =
                            usize::try_from(valuation.div_ceil(Limb::BITS)).unwrap_unchecked();
                        length.unchecked_add(padding)
                    };
                    product.reserve_exact(capacity.saturating_sub(product.limbs().len()));
                }
                let mut output = product.prepare_limb_write(length);
                // SAFETY: the reserved output is disjoint from the stack or
                // owned square and the positive swing. Multiplication fills
                // every slot; normalized operands permit one high zero guard.
                let active = unsafe {
                    let initialized = Multiplication::mul_nonempty_distinct_into_uninit(
                        square_limbs,
                        swing.limbs(),
                        from_raw_parts_mut(output.as_mut_ptr().cast(), length),
                        &mut scratch,
                    );
                    let guard = *initialized.get_unchecked(length.unchecked_sub(1)) == 0;
                    length.unchecked_sub(usize::from(guard))
                };
                // SAFETY: multiplication initialized length limbs; active
                // omits only its optional zero guard and remains positive.
                unsafe {
                    product.set_len(active);
                }
                swap(&mut result, &mut product);
            }
        }

        let mut remaining = valuation;
        while remaining != 0 {
            let chunk = usize::try_from(remaining).unwrap_or(usize::MAX);
            result.shl_assign(chunk);
            // SAFETY: chunk is a converted u32 or a smaller 16-bit usize::MAX;
            // it fits u32 and is no larger than remaining.
            remaining = unsafe { remaining.unchecked_sub(u32::try_from(chunk).unwrap_unchecked()) };
        }
        result
    }
}

/// Generates swings beyond the native table.
/// For each p, `p^(sum_j (floor(n/p^j) mod 2))<=n`, so each prime power
/// fits u32. Native u64 chunks flush only on a genuine product overflow.
fn prime_swing(
    n: u32,
    composite: &[u8],
    factors: &mut Vec<u64>,
    scratch: &mut MulScratch,
) -> InternalMpUint {
    factors.clear();
    let mut chunk = 1_u64;
    // SAFETY: the caller provides all odd sieve bits through n.
    let bytes = unsafe { usize::try_from(n >> 4).unwrap_unchecked().unchecked_add(1) };
    // SAFETY: three is a nonzero u32 divisor.
    let third = Div::div(n, unsafe { NonZeroU32::new_unchecked(3) });
    'primes: for (index, &marked) in composite.iter().take(bytes).enumerate() {
        let mut candidates = !marked;
        while candidates != 0 {
            let offset = candidates.trailing_zeros();
            // SAFETY: index<=n>>4 and offset<=7 imply 16*index+2*offset+1
            // fits u32. Bit zero marks one as composite, so prime>=3.
            let prime = unsafe {
                u32::try_from(index)
                    .unwrap_unchecked()
                    .unchecked_mul(16)
                    .unchecked_add(offset.unchecked_mul(2))
                    .unchecked_add(1)
            };
            // SAFETY: the loop condition proves a nonzero candidate mask.
            candidates &= unsafe { candidates.unchecked_sub(1) };
            // SAFETY: every unmarked candidate is a positive odd prime.
            let divisor = unsafe { NonZeroU32::new_unchecked(prime) };
            if prime > n {
                break 'primes;
            }
            let power = if prime > n >> 1 {
                u64::from(prime)
            } else if prime > third {
                // floor(n/p)=2 and p^2>n: the swing exponent is zero.
                continue;
            } else {
                let mut quotient = n;
                let mut power = 1_u64;
                loop {
                    quotient = Div::div(quotient, divisor);
                    if quotient & 1 != 0 {
                        // SAFETY: at most floor(log_p(n)) prime factors
                        // are selected; every partial power is <=n<=u32::MAX.
                        power = unsafe { power.unchecked_mul(u64::from(prime)) };
                    }
                    if quotient < prime {
                        break;
                    }
                }
                power
            };
            if let Some(combined) = chunk.checked_mul(power) {
                chunk = combined;
            } else {
                factors.push(chunk);
                chunk = power;
            }
        }
    }
    factors.push(chunk);
    factorial_product(factors, scratch)
}

/// Combines packed factors through a balanced multiplication tree.
/// Every nonfinal chunk exceeds 2^32: its next factor is at most 2^32-1
/// and caused u64 overflow. Splitting by count therefore balances widths.
fn factorial_product(factors: &[u64], scratch: &mut MulScratch) -> InternalMpUint {
    if let [value] = factors {
        return InternalMpUint::from_u64(*value);
    }
    debug_assert!(!factors.is_empty(), "the final chunk is always appended");
    let (low, high) = factors.split_at(factors.len() >> 1);
    let left = factorial_product(low, scratch);
    let right = factorial_product(high, scratch);
    let mut product = InternalMpUint::zero();
    product.assign_product_with_scratch(&left, &right, scratch);
    product
}

/// Constructs an odd sieve beyond the precomputed bitmap's domain.
/// Only primes at most `sqrt(u32::MAX)<2^16` mark composites; the
/// precomputed bitmap covers those primes on every pointer width.
fn factorial_sieve(n: u32) -> Vec<u8> {
    let bytes = usize::try_from(n >> 4)
        .expect("factorial sieve exceeds addressable memory")
        .checked_add(1)
        .expect("factorial sieve exceeds addressable memory");
    let mut composite = alloc::vec![0_u8; bytes];
    // SAFETY: n exceeds the nonempty native bitmap, so bytes>0.
    unsafe {
        *composite.get_unchecked_mut(0) = 1;
    }
    let mut prime = 3_u32;
    while u64::from(prime).pow(2) <= u64::from(n) {
        // SAFETY: prime<=sqrt(n)<2^16, within the native bitmap.
        let marked =
            unsafe { *ODD_COMPOSITE.get_unchecked(usize::try_from(prime >> 4).unwrap_unchecked()) };
        if marked & (1 << ((prime >> 1) & 7)) == 0 {
            // SAFETY: the loop proves prime^2<=n<=u32::MAX and prime<2^16.
            let (mut multiple, step) =
                unsafe { (prime.unchecked_mul(prime), prime.unchecked_mul(2)) };
            loop {
                // SAFETY: multiple<=n bounds its byte below bytes; the
                // exclusive vector owns the initialized composite marks.
                unsafe {
                    *composite
                        .get_unchecked_mut(usize::try_from(multiple >> 4).unwrap_unchecked()) |=
                        1 << ((multiple >> 1) & 7);
                }
                // SAFETY: multiple<=n gives an exact nonnegative distance.
                if unsafe { n.unchecked_sub(multiple) } < step {
                    break;
                }
                // SAFETY: the distance test proves multiple+step<=n.
                multiple = unsafe { multiple.unchecked_add(step) };
            }
        }
        // SAFETY: prime<=65535, so its successor fits u32.
        prime = unsafe { prime.unchecked_add(2) };
    }
    composite
}

/// Evaluates the largest exact prefixes fitting `INLINE_LIMBS`.
/// S(2m+1)=(2m+1)*S(2m), S(2m)=S(2m-1)/odd(m).
#[expect(
    clippy::as_conversions,
    reason = "constant evaluation widens native limbs into DoubleLimb; quotient limbs and remainders are bounded by exact scalar division"
)]
#[cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "remainder < divisor bounds quotient to Limb on 32-bit targets"
    )
)]
const fn factorial_tables() -> FactorialTables {
    let mut tables = FactorialTables {
        odd: [[0; INLINE_LIMBS]; TABLE_SIZE],
        swing: [[0; INLINE_LIMBS]; TABLE_SIZE],
        odd_count: 1,
        swing_count: 1,
    };
    tables.odd[0][0] = 1;
    tables.swing[0][0] = 1;
    let mut kind = 0_usize;
    while kind < 2 {
        let mut value = [0; INLINE_LIMBS];
        value[0] = 1;
        let mut factor = 1_usize;
        while factor < TABLE_SIZE {
            if kind == 1 && factor & 1 == 0 {
                let half = factor >> 1;
                let divisor = half >> half.trailing_zeros();
                let mut remainder = DoubleLimb::MIN;
                let mut limb = INLINE_LIMBS;
                while limb != 0 {
                    // SAFETY: limb>0, and decrement selects a valid slot.
                    limb = unsafe { limb.unchecked_sub(1) };
                    // SAFETY: limb<INLINE_LIMBS bounds the initialized row.
                    // factor>=2 makes odd(factor/2)>0; remainder<divisor
                    // gives a quotient limb below B.
                    unsafe {
                        let wide =
                            (remainder << Limb::BITS) | (*value.as_ptr().add(limb) as DoubleLimb);
                        *value.as_mut_ptr().add(limb) =
                            wide.checked_div(divisor as DoubleLimb).unwrap_unchecked() as Limb;
                        remainder = wide.checked_rem(divisor as DoubleLimb).unwrap_unchecked();
                    }
                }
                assert!(remainder == 0, "the swing recurrence divides exactly");
            } else {
                let multiplier = if kind == 0 {
                    factor >> factor.trailing_zeros()
                } else {
                    factor
                };
                let mut carry = 0;
                let mut limb = 0;
                while limb < INLINE_LIMBS {
                    // SAFETY: the loop bounds the initialized row; adding
                    // the previous carry to a scalar product gives a next
                    // carry below multiplier<=TABLE_SIZE<=257<B.
                    unsafe {
                        let (low, high) =
                            ArchKernels::mul_limb_lo_hi(*value.as_ptr().add(limb), multiplier);
                        let (sum, overflow) = low.overflowing_add(carry);
                        *value.as_mut_ptr().add(limb) = sum;
                        carry = high.unchecked_add(overflow as Limb);
                        limb = limb.unchecked_add(1);
                    }
                }
                if carry != 0 {
                    break;
                }
            }
            // SAFETY: factor<TABLE_SIZE bounds the row; its successor
            // fits usize because TABLE_SIZE<=257 on all pointer widths.
            unsafe {
                if kind == 0 {
                    *tables.odd.as_mut_ptr().add(factor) = value;
                    tables.odd_count = factor.unchecked_add(1);
                } else {
                    *tables.swing.as_mut_ptr().add(factor) = value;
                    tables.swing_count = factor.unchecked_add(1);
                }
                factor = factor.unchecked_add(1);
            }
        }
        // SAFETY: kind<2 bounds its successor.
        kind = unsafe { kind.unchecked_add(1) };
    }
    tables
}
