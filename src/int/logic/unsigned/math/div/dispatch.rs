//! The division surface: what callers outside `div/` actually use.
//!
//! Two layers, and they answer different needs.
//!
//! The inherent methods on [`InternalMpUint`] are the whole everyday surface —
//! five operations, no scratch to thread, no algorithm to pick. Importing the
//! type is enough to divide with it.
//!
//! [`Division`] carries the rest: the tower dispatch and the `*_into` forms that
//! take a reusable [`DivScratch`], for callers that divide in a loop and would
//! otherwise pay an allocation per iteration.
//!
//! Dispatch order is fixed: a trivial-shape check, then the quotient-only
//! truncation when only a quotient is wanted, then Algorithm D,
//! Burnikel-Ziegler or Newton-Raphson according to the divisor length and the
//! generated crossover thresholds.

use core::{cmp::Ordering, mem::replace};

use super::{
    ArchKernels, BURNIKEL_ZIEGLER_THRESHOLD, DivScratch, InternalMpUint, LIMB_BITS, Limb,
    NEWTON_RAPHSON_THRESHOLD,
};

/// Namespace for the division tower.
///
/// This zero-sized marker is never constructed; its associated functions group
/// the dispatch surface and the algorithms implemented across this folder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Division;

impl InternalMpUint {
    /// Computes quotient and remainder of `self / rhs`.
    ///
    /// `rhs` must be non-zero.
    #[must_use]
    pub fn div_rem(&self, rhs: &Self) -> (Self, Self) {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a non-zero divisor"
        );
        let mut q = Self::zero();
        let mut r = Self::zero();
        if Division::small_quotient_div_rem(self, rhs, &mut q, &mut r) {
            return (q, r);
        }
        if Division::try_algorithm_d_unscratched::<true, true, true>(self, rhs, &mut q, &mut r) {
            return (q, r);
        }
        let mut scratch = DivScratch::default();
        Division::div_rem_into(self, rhs, &mut q, &mut r, &mut scratch);
        (q, r)
    }

    /// Computes `ceil(self / rhs)` for a caller-validated non-zero divisor.
    #[inline]
    #[must_use]
    pub fn div_ceil(&self, rhs: &Self) -> Self {
        debug_assert!(
            !rhs.is_zero(),
            "internal ceiling division requires a non-zero divisor"
        );
        let (mut quotient, remainder) = self.div_rem(rhs);
        if !remainder.is_zero() {
            quotient.increment();
        }
        quotient
    }

    /// Computes only the quotient of `self / rhs`.
    ///
    /// `rhs` must be non-zero. Prefer this over discarding the remainder of
    /// [`InternalMpUint::div_rem`]: producing a remainder costs the full
    /// divisor width on every quotient limb, and this path is free to skip it
    /// entirely.
    #[must_use]
    pub fn div(&self, rhs: &Self) -> Self {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a non-zero divisor"
        );
        let mut q = Self::zero();
        if Division::truncated_quotient(self, rhs, &mut q) {
            return q;
        }
        let mut rem = Self::zero();
        if Division::try_algorithm_d_unscratched::<true, false, false>(self, rhs, &mut q, &mut rem)
        {
            return q;
        }
        let mut scratch = DivScratch::default();
        let mut dummy_rem = replace(&mut scratch.dummy_rem, Self::zero());
        Division::div_rem_into(self, rhs, &mut q, &mut dummy_rem, &mut scratch);
        scratch.dummy_rem = dummy_rem;
        q
    }

    /// Computes only the remainder of `self % rhs`.
    ///
    /// `rhs` must be non-zero.
    #[must_use]
    pub fn rem(&self, rhs: &Self) -> Self {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a non-zero divisor"
        );
        let mut dummy_quot = Self::zero();
        let mut r = Self::zero();
        if Division::try_algorithm_d_unscratched::<false, true, true>(
            self,
            rhs,
            &mut dummy_quot,
            &mut r,
        ) {
            return r;
        }
        let mut scratch = DivScratch::default();
        Division::rem_into(self, rhs, &mut r, &mut scratch);
        r
    }

    /// Returns whether `self` is an exact multiple of `rhs`.
    ///
    /// Division by zero follows the integer predicate convention: only zero is
    /// divisible by zero.
    #[must_use]
    pub fn is_divisible_by(&self, rhs: &Self) -> bool {
        let rhs_limbs = rhs.limbs();
        let self_limbs = self.limbs();

        if rhs_limbs.is_empty() {
            return self_limbs.is_empty();
        }
        if self_limbs.is_empty() {
            return true;
        }

        let m_orig = rhs_limbs.len();
        let n_orig = self_limbs.len();

        if n_orig < m_orig {
            return false;
        }

        // The 2-adic valuation precheck is a necessary condition for
        // divisibility, and it is also the whole test when the divisor is a
        // power of two: `self >> tz` is integral exactly when `v_2(self) >= tz`,
        // and `2^tz` divides every such value.  Both checks read a constant
        // number of low limbs in the common case, keeping this predicate O(1)
        // for the shapes that used to be special-cased before the general
        // remainder engine ran.
        let tz = rhs.trailing_zeros();
        if tz > 0 && self.trailing_zeros() < tz {
            return false;
        }
        if rhs.is_power_of_two() {
            return true;
        }

        if m_orig == 1 {
            let mut dummy = Self::zero();
            return Division::div_rem_1::<false>(
                self_limbs,
                *rhs_limbs.first().unwrap_or(&0),
                &mut dummy,
            ) == 0;
        }

        if self < rhs {
            return false;
        }

        // Fall back to the sub-quadratic remainder tower if the quotient and divisor are large.
        if m_orig >= BURNIKEL_ZIEGLER_THRESHOLD
            && n_orig.wrapping_sub(m_orig) >= BURNIKEL_ZIEGLER_THRESHOLD
        {
            let mut dummy = Self::zero();
            let mut scratch = DivScratch::default();
            Division::rem_into(self, rhs, &mut dummy, &mut scratch);
            return dummy.is_zero();
        }

        if n_orig == m_orig {
            return is_divisible_by_equal_length(self_limbs, rhs_limbs, tz);
        }

        let mut a = self.clone();
        let mut d_cloned;
        let d_limbs = if tz > 0 {
            a.shr_assign(tz);
            d_cloned = rhs.clone();
            d_cloned.shr_assign(tz);
            d_cloned.limbs()
        } else {
            rhs_limbs
        };

        let a_limbs = a.limbs_mut();
        let d_len = d_limbs.len();
        let a_len = a_limbs.len();

        if a_len < d_len {
            return false;
        }

        if d_len == 1 {
            let mut dummy = Self::zero();
            return Division::div_rem_1::<false>(
                a_limbs,
                *d_limbs.first().unwrap_or(&0),
                &mut dummy,
            ) == 0;
        }

        is_divisible_by_shifted_loop(a_limbs, d_limbs)
    }

    /// Replaces `self` with the quotient of `self / rhs`.
    ///
    /// `rhs` must be non-zero.
    pub fn div_assign(&mut self, rhs: &Self) {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a non-zero divisor"
        );
        let source = replace(self, Self::zero());
        if Division::truncated_quotient(&source, rhs, self) {
            return;
        }
        let mut rem = Self::zero();
        if Division::try_algorithm_d_unscratched::<true, false, false>(&source, rhs, self, &mut rem)
        {
            return;
        }
        let mut scratch = DivScratch::default();
        let mut dummy_rem = replace(&mut scratch.dummy_rem, Self::zero());
        Division::div_rem_into(&source, rhs, self, &mut dummy_rem, &mut scratch);
        scratch.dummy_rem = dummy_rem;
    }

    /// Replaces `self` with the remainder of `self % rhs`.
    ///
    /// `rhs` must be non-zero.
    pub fn rem_assign(&mut self, rhs: &Self) {
        debug_assert!(
            !rhs.is_zero(),
            "internal division requires a non-zero divisor"
        );
        let source = replace(self, Self::zero());
        let mut dummy_quot = Self::zero();
        if Division::try_algorithm_d_unscratched::<false, true, true>(
            &source,
            rhs,
            &mut dummy_quot,
            self,
        ) {
            return;
        }
        let mut scratch = DivScratch::default();
        Division::rem_into(&source, rhs, self, &mut scratch);
    }
}

impl Division {
    /// Computes both halves of `num_a / den_b` into caller-owned outputs,
    /// choosing the divider from the divisor length.
    pub fn div_rem_into(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        debug_assert!(
            !den_b.is_zero(),
            "internal division requires a non-zero divisor"
        );
        if Self::trivial::<true, true>(num_a, den_b, quotient_out, rem_out) {
            return;
        }
        let v_limbs = Self::significant_limbs(den_b.limbs());
        let u_limbs = Self::significant_limbs(num_a.limbs());

        if v_limbs.len() == 1 {
            // SAFETY: v_limbs.len() == 1 so index 0 is valid.
            let v0 = unsafe { *v_limbs.get_unchecked(0) };
            let rem = Self::div_rem_1::<true>(u_limbs, v0, quotient_out);
            *rem_out = InternalMpUint::from_limb(rem);
            return;
        }

        if u_limbs.len() <= v_limbs.len().wrapping_add(1) {
            Self::algorithm_d(num_a, den_b, quotient_out, rem_out, scratch);
            return;
        }
        if v_limbs.len() >= NEWTON_RAPHSON_THRESHOLD {
            Self::newton(num_a, den_b, quotient_out, rem_out, scratch);
            return;
        }
        if v_limbs.len() >= BURNIKEL_ZIEGLER_THRESHOLD {
            Self::burnikel_ziegler(num_a, den_b, quotient_out, rem_out, scratch);
            return;
        }
        Self::algorithm_d(num_a, den_b, quotient_out, rem_out, scratch);
    }

    /// Computes only `num_a % den_b` using reusable division scratch.
    pub fn rem_into(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        rem_out: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        debug_assert!(
            !den_b.is_zero(),
            "internal division requires a non-zero divisor"
        );
        let mut dummy_quot = replace(&mut scratch.dummy_quot, InternalMpUint::zero());
        if !Self::trivial::<false, true>(num_a, den_b, &mut dummy_quot, rem_out) {
            let v_limbs = Self::significant_limbs(den_b.limbs());
            let u_limbs = Self::significant_limbs(num_a.limbs());
            if v_limbs.len() == 1 {
                // SAFETY: v_limbs.len() == 1 so index 0 is valid.
                let v0 = unsafe { *v_limbs.get_unchecked(0) };
                let rem = Self::div_rem_1::<false>(u_limbs, v0, &mut dummy_quot);
                *rem_out = InternalMpUint::from_limb(rem);
            } else if u_limbs.len() <= v_limbs.len().wrapping_add(1) {
                Self::algorithm_d_rem(num_a, den_b, rem_out, scratch);
            } else if v_limbs.len() >= NEWTON_RAPHSON_THRESHOLD {
                Self::newton(num_a, den_b, &mut dummy_quot, rem_out, scratch);
            } else if v_limbs.len() >= BURNIKEL_ZIEGLER_THRESHOLD {
                Self::burnikel_ziegler(num_a, den_b, &mut dummy_quot, rem_out, scratch);
            } else {
                Self::algorithm_d_rem(num_a, den_b, rem_out, scratch);
            }
        }
        scratch.dummy_quot = dummy_quot;
    }

    /// Resolves quotients that are provably `0` or `1`.
    ///
    /// Returns `false` when the operands need the tower.
    pub fn trivial<const WRITE_QUOTIENT: bool, const WRITE_REMAINDER: bool>(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
    ) -> bool {
        let v_limbs = Self::significant_limbs(den_b.limbs());
        debug_assert!(!v_limbs.is_empty(), "division requires a non-zero divisor");
        let u_limbs = Self::significant_limbs(num_a.limbs());

        if u_limbs.len() < v_limbs.len() {
            if WRITE_QUOTIENT {
                quotient_out.clear();
            }
            if WRITE_REMAINDER {
                rem_out.clone_from(num_a);
            }
            return true;
        }

        if u_limbs.len() == v_limbs.len() {
            match InternalMpUint::cmp_limbs(u_limbs, v_limbs) {
                Ordering::Less => {
                    if WRITE_QUOTIENT {
                        quotient_out.clear();
                    }
                    if WRITE_REMAINDER {
                        rem_out.clone_from(num_a);
                    }
                    return true;
                }
                Ordering::Equal => {
                    if WRITE_QUOTIENT {
                        *quotient_out = InternalMpUint::one();
                    }
                    if WRITE_REMAINDER {
                        rem_out.clear();
                    }
                    return true;
                }
                Ordering::Greater => {
                    if Self::less_than_double(u_limbs, v_limbs) {
                        if WRITE_QUOTIENT {
                            *quotient_out = InternalMpUint::one();
                        }
                        if WRITE_REMAINDER {
                            *rem_out = num_a.sub(den_b);
                        }
                        return true;
                    }
                }
            }
        }

        false
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    unsafe_code,
    reason = "The equal-width quotient proof bounds every index and cast; wrapping arithmetic computes the inverse modulo the native limb base"
)]
/// Tests exact divisibility when both normalized operands occupy the same
/// number of limbs.
///
/// The caller supplies `tz = v_2(d)` after proving `v_2(a) >= tz`. Therefore
/// `d' = d >> tz` is odd. Equal width also gives `a / d < B`, where
/// `B = 2^LIMB_BITS`, so an exact quotient is one limb. Its residue is uniquely
/// recovered as `(a' mod B) * (d' mod B)^(-1) mod B`; multiplying that candidate
/// by the original divisor then decides exact equality without division.
fn is_divisible_by_equal_length(a_limbs: &[Limb], d_limbs: &[Limb], tz: usize) -> bool {
    let d_len = d_limbs.len();
    let a_len = a_limbs.len();
    if d_len == 0 || a_len != d_len {
        return false;
    }
    let limb_shift = tz.wrapping_div(LIMB_BITS);
    let bit_shift = (tz.wrapping_rem(LIMB_BITS)) as u32;
    let bit_shift_comp = (LIMB_BITS as u32).wrapping_sub(bit_shift);
    debug_assert!(
        limb_shift < d_len,
        "a non-zero divisor has its first set bit within its active limbs"
    );

    let get_shifted = |limbs: &[Limb]| -> Limb {
        if bit_shift == 0 {
            // SAFETY: limb_shift < limbs.len() by caller bounds
            unsafe { *limbs.get_unchecked(limb_shift) }
        } else {
            // SAFETY: limb_shift < limbs.len() by caller bounds
            let low = unsafe { *limbs.get_unchecked(limb_shift) }.wrapping_shr(bit_shift);
            let next_idx = limb_shift.wrapping_add(1);
            let high = if next_idx < limbs.len() {
                // SAFETY: next_idx < limbs.len() checked here
                unsafe { *limbs.get_unchecked(next_idx) }.wrapping_shl(bit_shift_comp)
            } else {
                0
            };
            low | high
        }
    };

    let d_prime_0 = get_shifted(d_limbs);
    let a_prime_0 = get_shifted(a_limbs);
    debug_assert_eq!(
        d_prime_0 & 1,
        1,
        "shifting a non-zero divisor by its 2-adic valuation produces an odd value"
    );

    #[cfg(not(any(
        target_pointer_width = "16",
        target_pointer_width = "32",
        target_pointer_width = "64"
    )))]
    compile_error!("Modular inverse iterations must be updated for >64-bit platforms");

    // Every odd `x` satisfies `x^2 = 1 (mod 8)`, so using `x` itself as the
    // initial inverse starts with at least three correct bits. Newton's update
    // `y <- y * (2 - x*y)` doubles the number of correct low bits. Thus three,
    // four and five updates provide at least 24, 48 and 96 correct bits,
    // respectively, covering the supported 16-, 32- and 64-bit limbs.
    let mut inv = d_prime_0;
    inv = inv.wrapping_mul(2_usize.wrapping_sub(d_prime_0.wrapping_mul(inv)));
    inv = inv.wrapping_mul(2_usize.wrapping_sub(d_prime_0.wrapping_mul(inv)));
    inv = inv.wrapping_mul(2_usize.wrapping_sub(d_prime_0.wrapping_mul(inv)));
    #[cfg(not(target_pointer_width = "16"))]
    {
        inv = inv.wrapping_mul(2_usize.wrapping_sub(d_prime_0.wrapping_mul(inv)));
    }
    #[cfg(not(any(target_pointer_width = "16", target_pointer_width = "32")))]
    {
        inv = inv.wrapping_mul(2_usize.wrapping_sub(d_prime_0.wrapping_mul(inv)));
    }

    let quotient = a_prime_0.wrapping_mul(inv);

    let mut carry = 0;
    for (idx, &d_limb) in d_limbs.iter().enumerate() {
        let (prod, high_carry) = ArchKernels::mul_limb_lo_hi(d_limb, quotient);
        let (sum, sum_carry) = prod.overflowing_add(carry);
        // SAFETY: idx < a_limbs.len() because a_len == d_len
        if unsafe { *a_limbs.get_unchecked(idx) } != sum {
            return false;
        }
        // A limb product has `high_carry <= B - 2`; adding the one-bit carry
        // from the low-half sum therefore stays at most `B - 1`. The wrapping
        // addition cannot wrap and merely keeps this hot loop branch-free.
        carry = high_carry.wrapping_add(Limb::from(sum_carry));
    }
    carry == 0
}

#[allow(
    unsafe_code,
    reason = "The caller-proved slice lengths, odd divisor and disjoint owners satisfy the exact low-to-high division kernel's pointer contracts"
)]
/// Tests exact divisibility by cancelling quotient limbs from least to most
/// significant.
///
/// The sole caller provides `a_len >= d_len >= 2` and an odd divisor after
/// removing the divisor's complete power-of-two factor from both operands.
/// At step `i`, `q_i = a_i * d_0^(-1) mod B` makes limb `i` exactly zero. Once
/// every possible quotient limb has been cancelled, divisibility is equivalent
/// to every unprocessed high limb and the final borrow being zero.
fn is_divisible_by_shifted_loop(a_limbs: &mut [Limb], d_limbs: &[Limb]) -> bool {
    let d_len = d_limbs.len();
    let a_len = a_limbs.len();
    debug_assert!(d_len >= 2, "the single-limb divisor path runs first");
    debug_assert!(
        a_len >= d_len,
        "division requires a dividend at least as wide as the divisor"
    );
    // SAFETY: the sole caller reaches this helper only after handling
    // `d_len == 1`, so the first initialized, limb-aligned element exists.
    let divisor = unsafe { *d_limbs.get_unchecked(0) };
    debug_assert_eq!(
        divisor & 1,
        1,
        "the caller removes the divisor's complete power-of-two factor"
    );

    #[cfg(not(any(
        target_pointer_width = "16",
        target_pointer_width = "32",
        target_pointer_width = "64"
    )))]
    compile_error!("Modular inverse iterations must be updated for >64-bit platforms");

    // For odd `divisor`, the seed is correct modulo 8 because every odd square
    // is 1 modulo 8. Each Newton update doubles the correct-bit count, so the
    // target-specific update counts below cover 16, 32 and 64 bits.
    let mut inv = divisor;
    inv = inv.wrapping_mul(2_usize.wrapping_sub(divisor.wrapping_mul(inv)));
    inv = inv.wrapping_mul(2_usize.wrapping_sub(divisor.wrapping_mul(inv)));
    inv = inv.wrapping_mul(2_usize.wrapping_sub(divisor.wrapping_mul(inv)));
    #[cfg(not(target_pointer_width = "16"))]
    {
        inv = inv.wrapping_mul(2_usize.wrapping_sub(divisor.wrapping_mul(inv)));
    }
    #[cfg(not(any(target_pointer_width = "16", target_pointer_width = "32")))]
    {
        inv = inv.wrapping_mul(2_usize.wrapping_sub(divisor.wrapping_mul(inv)));
    }

    let limit = a_len.wrapping_sub(d_len);
    for idx in 0..=limit {
        // SAFETY: idx <= a_len - d_len < a_len
        let a_idx = unsafe { *a_limbs.get_unchecked(idx) };
        let q_i = a_idx.wrapping_mul(inv);

        // SAFETY: `idx <= a_len - d_len` gives `idx + d_len <= a_len`, so the
        // destination suffix and source each cover `d_len` initialized,
        // limb-aligned elements. Rust's simultaneous `&mut [Limb]` and
        // `&[Limb]` inputs guarantee the regions do not alias, satisfying every
        // selected architecture backend's pointer contract.
        let (carry, borrow) = unsafe {
            ArchKernels::sub_mul_limbs_unchecked(
                a_limbs.as_mut_ptr().add(idx),
                d_limbs.as_ptr(),
                d_len,
                q_i,
            )
        };
        // `q_i * d_0 = a_i (mod B)` by construction, so the processed low
        // limb is now exactly zero and never needs to be inspected again.
        debug_assert_eq!(
            // SAFETY: `idx <= limit < a_len` by the loop bounds.
            unsafe { *a_limbs.get_unchecked(idx) },
            0,
            "the modular quotient digit must cancel its low limb"
        );

        let mut carry_idx = idx.wrapping_add(d_len);
        // The multiply carry is at most `B - 2` and `borrow` is one bit, so
        // their sum is at most `B - 1`; this wrapping addition cannot overflow.
        let mut current_borrow = carry.wrapping_add(borrow);
        while current_borrow > 0 && carry_idx < a_len {
            // SAFETY: carry_idx < a_len by loop condition
            let a_k = unsafe { *a_limbs.get_unchecked(carry_idx) };
            let (sub, underflow) = a_k.overflowing_sub(current_borrow);
            // SAFETY: carry_idx < a_len by loop condition
            *unsafe { a_limbs.get_unchecked_mut(carry_idx) } = sub;
            current_borrow = Limb::from(underflow);
            carry_idx = carry_idx.wrapping_add(1);
        }
        if current_borrow > 0 {
            return false;
        }
    }

    // Limbs `0..=a_len-d_len` were proved zero one at a time above. The
    // untouched suffix is the only remaining part of the exact remainder.
    let start = a_len.wrapping_sub(d_len).wrapping_add(1);
    for &limb in a_limbs.iter().skip(start) {
        if limb != 0 {
            return false;
        }
    }
    true
}
