//! Exact quotients and divisibility predicates for `InternalMpUint`.
//!
//! Known exact division uses scalar cancellation, quotient truncation or the
//! Newton tower. Divisibility predicates use an equal-width modular-residue
//! check or low-to-high quotient cancellation for wider dividends. The scalar
//! and predicate kernels use the divisor's odd part after removing its complete
//! power-of-two factor.
//!
//! References:
//! - T. Jebelean, "An algorithm for exact division", Journal of Symbolic Computation,
//!   Vol. 15, No. 2, pp. 169–180, Feb. 1993. DOI: 10.1006/jsco.1993.1012.

#![expect(
    unsafe_code,
    reason = "exact divisibility and validated operand spans establish cancellation, shift, and quotient kernel contracts"
)]

use core::{
    cmp::Ordering,
    mem::MaybeUninit,
    num::NonZeroUsize,
    ops::Rem,
    ptr::copy_nonoverlapping,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use super::{
    ArchKernels, DIVISION_DIVISIBLE_THRESHOLD, DIVISION_STACK_LIMBS, DivScratch, Division,
    InternalMpUint, Limb, ScratchBuffer,
};

impl InternalMpUint {
    /// Returns whether `self` is an exact multiple of `rhs`.
    /// Division by zero follows the integer predicate convention: only zero
    /// is divisible by zero.
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
        // Remove common whole-limb factors while checking the necessary
        // valuation condition. The canonical divisor terminates this scan.
        let mut a = self_limbs;
        let mut d = rhs_limbs;
        let (a_low, d_low) = loop {
            // SAFETY: removing a zero low limb cannot exhaust the nonzero
            // canonical divisor. Equal removals preserve a.len() >= d.len().
            let (a_low, d_low) = unsafe { (*a.get_unchecked(0), *d.get_unchecked(0)) };
            if d_low != 0 {
                break (a_low, d_low);
            }
            if a_low != 0 {
                return false;
            }
            // SAFETY: both first limbs exist and d's nonzero top remains.
            unsafe {
                a = a.get_unchecked(1..);
                d = d.get_unchecked(1..);
            }
        };
        let shift = d_low.trailing_zeros();
        // SAFETY: d_low != 0 gives shift < Limb::BITS, so 1<<shift is positive.
        let mask = unsafe { (1_usize << shift).unchecked_sub(1) };
        if a_low & mask != 0 {
            return false;
        }
        if d.len() == 1 {
            let odd = d_low >> shift;
            if odd == 1 {
                return true;
            }
            if a.len() == 1 {
                // SAFETY: the zero low divisor limbs were removed above.
                return unsafe { Rem::rem(a_low, NonZeroUsize::new_unchecked(d_low)) } == 0;
            }
            return is_divisible_by_odd_limb(a, odd);
        }
        if let [low, high] = d
            && shift != 0
            && high >> shift == 0
        {
            // SAFETY: the branch and nonzero low divisor give 0 < shift < Limb::BITS.
            let complement = unsafe { Limb::BITS.unchecked_sub(shift) };
            let odd = (low >> shift) | (high << complement);
            return is_divisible_by_odd_limb(a, odd);
        }
        if n_orig == m_orig {
            // Equal positive high limbs imply A < 2D; divisibility is A=D.
            match a.last().cmp(&d.last()) {
                Ordering::Less => return false,
                Ordering::Equal => return a == d,
                Ordering::Greater => {}
            }
            return is_divisible_by_equal_length(a, d, shift);
        }
        // SAFETY: a < d and equal widths returned above, so n_orig > m_orig.
        let extra = unsafe { n_orig.unchecked_sub(m_orig) };
        if d.len() >= DIVISION_DIVISIBLE_THRESHOLD && extra >= DIVISION_DIVISIBLE_THRESHOLD {
            let mut remainder = Self::zero();
            Division::rem_into(self, rhs, &mut remainder, &mut DivScratch::default());
            return remainder.is_zero();
        }
        is_divisible_by_shifted(a, d, shift)
    }
}

impl Division {
    /// Divides by a known nonzero exact divisor into reusable output storage.
    ///
    /// A scalar divisor is made odd by shifting the input during its read.
    /// For limb base B, the recurrence `q_i = (a_i - carry)*d^-1 mod B`
    /// cancels each low limb. The high half of `q_i*d` plus the subtraction
    /// borrow is the next carry, bounded by d-1. Exactness makes the final
    /// carry zero. No normalization, reciprocal estimate or correction is needed.
    /// Wider divisors use exact prefix truncation or the general quotient tower.
    pub fn div_exact_into(
        numerator: &InternalMpUint,
        divisor: &InternalMpUint,
        quotient: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        debug_assert!(
            !divisor.is_zero(),
            "exact division requires a nonzero divisor"
        );
        if numerator.is_zero() {
            quotient.clear();
            return;
        }
        if let [scalar] = divisor.limbs() {
            if scalar.is_power_of_two() {
                let _ = Self::div_rem_1::<true>(numerator.limbs(), *scalar, quotient);
                return;
            }
            let input = numerator.limbs();
            let mut output = quotient.prepare_limb_write(input.len());
            // SAFETY: the nonzero numerator and scalar are disjoint from the
            // output guard, which reserves input.len() aligned writable limbs.
            // Exact divisibility is proved by the caller; the kernel initializes
            // every reserved limb before the length is committed.
            unsafe {
                div_exact_1_raw(input, *scalar, output.as_mut_ptr());
                let _ = output.commit();
            }
            quotient.normalize();
        } else {
            Self::div_into::<true, true>(numerator, divisor, quotient, scratch);
        }
    }
}

/// Tests exact divisibility of canonical equal-width operands.
///
/// The caller removes whole zero limbs and supplies `shift = v_2(d)`
/// after proving `v_2(a) >= shift`. Thus `d' = d >> shift` is odd.
/// Both operands have at least two limbs. Equal width gives `a / d < B`, where
/// `B = 2^LIMB_BITS`, so an exact quotient is one limb. Its residue is
/// uniquely recovered as `(a' mod B) * (d' mod B)^(-1) mod B`; multiplying
/// that candidate by the original divisor then decides exact equality
/// without division.
fn is_divisible_by_equal_length(a_limbs: &[Limb], d_limbs: &[Limb], shift: u32) -> bool {
    let d_len = d_limbs.len();
    let a_len = a_limbs.len();
    debug_assert!(
        a_len == d_len && d_len >= 2 && shift < Limb::BITS,
        "the predicate dispatcher supplies equal multi-limb widths and a sub-limb valuation"
    );
    // SAFETY: the dispatcher supplies equal initialized spans of at least
    // two limbs. Whole zero limbs were removed, so shift < Limb::BITS;
    // the nonzero branch has both complementary shifts strictly in range.
    let (a_prime_0, d_prime_0) = unsafe {
        let a0 = *a_limbs.get_unchecked(0);
        let d0 = *d_limbs.get_unchecked(0);
        if shift == 0 {
            (a0, d0)
        } else {
            let complement = Limb::BITS.unchecked_sub(shift);
            (
                (a0 >> shift) | (*a_limbs.get_unchecked(1) << complement),
                (d0 >> shift) | (*d_limbs.get_unchecked(1) << complement),
            )
        }
    };
    debug_assert_eq!(
        d_prime_0 & 1,
        1,
        "shifting a non-zero divisor by its 2-adic valuation produces an odd value"
    );

    let inv = Division::modular_inverse_limb(d_prime_0);
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
        // from the low-half sum therefore stays at most `B - 1`.
        // SAFETY: high_carry+sum_carry <= B-1 fits the native limb.
        carry = unsafe { high_carry.unchecked_add(Limb::from(sum_carry)) };
    }
    carry == 0
}

/// Removes a sub-limb power of two into bounded stack or pooled storage.
///
/// The caller supplies canonical `a > d`, with whole zero limbs removed,
/// `shift = v_2(d) < Limb::BITS`, and `v_2(a) >= shift`. The odd part of d
/// has at least two limbs. An already odd divisor is borrowed directly.
fn is_divisible_by_shifted(a: &[Limb], d: &[Limb], shift: u32) -> bool {
    debug_assert!(
        a.len() > d.len() && d.len() >= 2 && shift < Limb::BITS,
        "cancellation requires a wider dividend and a multi-limb odd part"
    );
    let mut a_stack = [MaybeUninit::<Limb>::uninit(); DIVISION_STACK_LIMBS];
    let mut d_stack = [MaybeUninit::<Limb>::uninit(); DIVISION_STACK_LIMBS];
    let mut a_heap;
    let mut d_heap;
    let a_ptr = if a.len() <= DIVISION_STACK_LIMBS {
        a_stack.as_mut_ptr().cast::<Limb>()
    } else {
        a_heap = ScratchBuffer::acquire(a.len());
        a_heap.as_mut_ptr()
    };
    if shift == 0 {
        // SAFETY: the selected stack or heap reserves a.len() limbs,
        // disjoint from the initialized immutable input. This initializes
        // the complete prefix before forming its mutable slice.
        unsafe {
            copy_nonoverlapping(a.as_ptr(), a_ptr, a.len());
        }
    } else {
        // SAFETY: both disjoint spans cover a.len() aligned limbs; the
        // selected destination is writable but need not be initialized.
        // Here 0 < shift < Limb::BITS, and the kernel initializes all limbs.
        let _ = unsafe { ArchKernels::rshift_into_unchecked(a_ptr, a.as_ptr(), a.len(), shift) };
    }
    // SAFETY: the preceding copy or shift initialized the reserved prefix.
    // Its unique mutable borrow ends before either owning buffer is dropped.
    let mut numerator = unsafe { from_raw_parts_mut(a_ptr, a.len()) };
    if numerator.last() == Some(&0) {
        // SAFETY: a is canonical and shift < Limb::BITS, so at most one
        // high limb becomes zero. a > d and d's odd part has two limbs.
        let length = unsafe { numerator.len().unchecked_sub(1) };
        numerator = numerator.split_at_mut(length).0;
    }
    let divisor = if shift == 0 {
        d
    } else {
        let d_ptr = if d.len() <= DIVISION_STACK_LIMBS {
            d_stack.as_mut_ptr().cast::<Limb>()
        } else {
            d_heap = ScratchBuffer::acquire(d.len());
            d_heap.as_mut_ptr()
        };
        // SAFETY: d.len() aligned destination limbs are reserved in an
        // independent buffer. The nonoverlapping source is initialized and
        // 0 < shift < Limb::BITS; the shift initializes the complete prefix.
        let _ = unsafe { ArchKernels::rshift_into_unchecked(d_ptr, d.as_ptr(), d.len(), shift) };
        // SAFETY: the shift initialized this prefix, whose owner remains
        // live and unmodified throughout the cancellation call below.
        let shifted = unsafe { from_raw_parts(d_ptr, d.len()) };
        if let Some((&0, prefix)) = shifted.split_last() {
            prefix
        } else {
            shifted
        }
    };
    is_divisible_by_shifted_loop(numerator, divisor)
}

/// Tests exact divisibility by cancelling quotient limbs from least to most
/// significant.
///
/// The sole caller provides `a_len >= d_len >= 2` and an odd divisor after
/// removing the divisor's complete power-of-two factor from both operands.
/// At step `i`, `q_i = a_i * d_0^(-1) mod B` makes limb `i` exactly zero.
/// Once every possible quotient limb has been cancelled, divisibility is
/// equivalent to every unprocessed high limb and the final borrow being
/// zero.
#[expect(
    clippy::inline_always,
    reason = "the measured predicate path otherwise materializes its normalized slice arguments for a second call; inlining keeps normalization and cancellation in one frame"
)]
#[inline(always)]
pub fn is_divisible_by_shifted_loop(a_limbs: &mut [Limb], d_limbs: &[Limb]) -> bool {
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

    let inv = Division::modular_inverse_limb(divisor);
    let sub_mul = ArchKernels::selected_sub_mul_limbs_unchecked();

    // A leading limb below D's leading limb proves A < D*B^(a_len-d_len).
    // The quotient then needs only a_len-d_len cancellations. Otherwise
    // retain the possible extra digit; equal leading limbs alone do not
    // establish the strict bound on the complete high window.
    let extra = usize::from(a_limbs.last() >= d_limbs.last());
    // SAFETY: d_len >= 2 and a_len bounds an allocated limb slice, so
    // a_len-d_len+extra is nonnegative and strictly below a_len.
    let digits = unsafe { a_len.unchecked_sub(d_len).unchecked_add(extra) };
    for idx in 0..digits {
        // SAFETY: idx <= a_len - d_len < a_len.
        let a_idx = unsafe { *a_limbs.get_unchecked(idx) };
        let q_i = a_idx.wrapping_mul(inv);
        if q_i == 0 {
            // Multiplication by zero changes neither the remaining
            // dividend nor its borrow; this low limb is already zero.
            continue;
        }

        // SAFETY: `idx <= a_len - d_len` gives `idx + d_len <= a_len`, so the
        // destination suffix and source each cover `d_len` initialized,
        // limb-aligned elements. Rust's simultaneous `&mut [Limb]` and
        // `&[Limb]` inputs guarantee the regions do not alias, satisfying every
        // selected architecture backend's pointer contract.
        let (carry, borrow) =
            unsafe { sub_mul(a_limbs.as_mut_ptr().add(idx), d_limbs.as_ptr(), d_len, q_i) };
        // `q_i * d_0 = a_i (mod B)` by construction, so the processed low
        // limb is now exactly zero and never needs to be inspected again.
        debug_assert_eq!(
            // SAFETY: `idx < digits < a_len` by the loop bounds.
            unsafe { *a_limbs.get_unchecked(idx) },
            0,
            "the modular quotient digit must cancel its low limb"
        );

        // SAFETY: idx < digits <= a_len-d_len+1 gives idx+d_len <= a_len.
        let mut carry_idx = unsafe { idx.unchecked_add(d_len) };
        // SAFETY: q_i*d < q_i*B^d_len bounds the product carry below
        // q_i <= B-1. Adding the binary borrow therefore fits a limb.
        let mut current_borrow = unsafe { carry.unchecked_add(borrow) };
        while current_borrow > 0 && carry_idx < a_len {
            // SAFETY: carry_idx < a_len by loop condition
            let a_k = unsafe { *a_limbs.get_unchecked(carry_idx) };
            let (sub, underflow) = a_k.overflowing_sub(current_borrow);
            // SAFETY: carry_idx < a_len by loop condition
            *unsafe { a_limbs.get_unchecked_mut(carry_idx) } = sub;
            current_borrow = Limb::from(underflow);
            // SAFETY: carry_idx < a_len and an allocated Limb slice has
            // strictly fewer than usize::MAX elements on every target.
            carry_idx = unsafe { carry_idx.unchecked_add(1) };
        }
        if current_borrow > 0 {
            return false;
        }
    }

    // Limbs `0..digits` were proved zero one at a time above. The
    // untouched suffix is the only remaining part of the exact remainder.
    for &limb in a_limbs.iter().skip(digits) {
        if limb != 0 {
            return false;
        }
    }
    true
}

/// Cancels a known scalar factor from low to high into raw output storage.
///
/// For odd d, `q_i = (a_i-carry)*d^-1 mod B` cancels one low limb.
/// The next carry is `high(q_i*d)+borrow < d`: equality of the product's
/// high half with d-1 forces its low half at most B-d, whereas a borrow
/// gives residue at least B-d+1. The final exact residue is a nonnegative
/// multiple of d below B, so its quotient needs no high product or carry.
///
/// # Safety
///
/// `input` is nonempty, scalar is nonzero and divides its value exactly.
/// `output` covers `input.len()` aligned writable limbs, disjoint from input.
/// The output need not be initialized; every limb receives its first write.
unsafe fn div_exact_1_raw(input: &[Limb], scalar: Limb, output: *mut Limb) {
    let shift = scalar.trailing_zeros();
    let odd = scalar >> shift;
    let inverse = Division::modular_inverse_limb(odd);
    // SAFETY: the caller supplies a nonempty initialized input span.
    let (&top, prefix) = unsafe { input.split_last().unwrap_unchecked() };
    let mut carry = 0;
    if shift == 0 {
        for (index, &limb) in prefix.iter().enumerate() {
            let (residue, borrow) = limb.overflowing_sub(carry);
            let digit = residue.wrapping_mul(inverse);
            let (_, high) = ArchKernels::mul_limb_lo_hi(digit, odd);
            // SAFETY: the cancellation bound above proves high+borrow<odd.
            // index < prefix.len() < input.len() bounds the writable slot.
            unsafe {
                carry = high.unchecked_add(Limb::from(borrow));
                output.add(index).write(digit);
            }
        }
    } else {
        // SAFETY: scalar>0 gives shift<Limb::BITS; this branch has shift>0.
        // The nonempty input bounds the suffix, whose length matches prefix.
        let (complement, following) =
            unsafe { (Limb::BITS.unchecked_sub(shift), input.get_unchecked(1..)) };
        for (index, (&low, &next)) in prefix.iter().zip(following).enumerate() {
            let limb = (low >> shift) | (next << complement);
            let (residue, borrow) = limb.overflowing_sub(carry);
            let digit = residue.wrapping_mul(inverse);
            let (_, high) = ArchKernels::mul_limb_lo_hi(digit, odd);
            // SAFETY: the cancellation bound proves high+borrow<odd.
            // Equal prefix and suffix lengths bound this writable slot.
            unsafe {
                carry = high.unchecked_add(Limb::from(borrow));
                output.add(index).write(digit);
            }
        }
    }
    // Exactness makes (top>>shift)-carry divisible by odd. A negative
    // value would lie strictly between -odd and zero, which excludes an
    // exact multiple. Its nonnegative quotient is below B.
    // SAFETY: scalar>0 bounds shift, and the exactness argument proves
    // top>>shift>=carry. prefix.len() is the final reserved output slot.
    unsafe {
        let residue = (top >> shift).unchecked_sub(carry);
        let digit = residue.wrapping_mul(inverse);
        debug_assert_eq!(
            digit.checked_mul(odd),
            Some(residue),
            "exactness bounds the final product below the limb base"
        );
        output.add(prefix.len()).write(digit);
    }
}

/// Tests divisibility by an odd limb after the caller handles zero, one,
/// and single-limb dividends. No canonical modular residue is required.
///
/// After k cancellations, `A_low = Q*d - carry*B^k`, with `0 <= carry < d`.
/// Since B is invertible modulo d, the final carry is zero exactly when
/// d divides A. The last limb needs no canonical residue adjustment.
pub fn is_divisible_by_odd_limb(numerator: &[Limb], divisor: Limb) -> bool {
    debug_assert!(
        numerator.len() >= 2 && divisor > 1 && divisor & 1 == 1,
        "the dispatcher handles scalar dividends, one, and the power-of-two factor"
    );
    let inverse = Division::modular_inverse_limb(divisor);
    let mut carry = 0;
    for &limb in numerator {
        let (residue, borrow) = limb.overflowing_sub(carry);
        let digit = residue.wrapping_mul(inverse);
        let (_, high) = ArchKernels::mul_limb_lo_hi(digit, divisor);
        // high <= d-1. Equality implies residue <= B-d, whereas a
        // borrow requires residue >= B-d+1, so high+borrow < d.
        // SAFETY: the preceding bound gives high+borrow < divisor <= Limb::MAX.
        carry = unsafe { high.unchecked_add(Limb::from(borrow)) };
    }
    carry == 0
}
