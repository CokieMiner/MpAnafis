//! Prepared leading pair, reciprocal and digit evaluation for division windows.
//!
//! Recursive division only removes low divisor limbs. Every leaf therefore
//! shares the same leading pair, reciprocal, and selected subtraction kernel.

#![expect(
    unsafe_code,
    reason = "normalized leading pairs, initialized windows and bounded carries establish reciprocal and digit-kernel contracts"
)]
#![expect(
    clippy::self_named_module_files,
    reason = "private divisor fields require implementation children beneath their type owner; mod.rs files remain structural registries"
)]

use core::num::NonZeroUsize;

use super::{Addition, ArchKernels, Division, DoubleLimb, LIMB_BITS, Limb, QuotientDivisionMode};

mod full;
mod short;

/// Invariant divisor state shared by the prepared divisor and its high suffixes.
#[derive(Clone, Copy)]
pub struct PreparedDivisor {
    high: NonZeroUsize,
    low: Limb,
    inverse: Limb,
    sub_mul: unsafe fn(*mut Limb, *const Limb, usize, Limb) -> (Limb, Limb),
}

impl PreparedDivisor {
    /// Prepares a normalized divisor of at least two limbs. Subsequent windows
    /// use this divisor or one of its high suffixes.
    pub fn new(divisor: &[Limb]) -> Self {
        debug_assert!(
            divisor.len() >= 2,
            "the reciprocal requires two divisor limbs"
        );
        // SAFETY: normalized callers supply at least two initialized limbs.
        // The fixed-size leading pair exists; its two loads prepare the
        // reciprocal without independent index or minimum-width assertions.
        let (_, &[low, high]) = unsafe { divisor.split_last_chunk::<2>().unwrap_unchecked() };
        debug_assert_ne!(high >> (Limb::BITS - 1), 0, "the divisor is normalized");
        Self {
            // SAFETY: normalization sets the high bit, so high is nonzero.
            high: unsafe { NonZeroUsize::new_unchecked(high) },
            low,
            inverse: Division::invert_pi1(high, low),
            sub_mul: ArchKernels::selected_sub_mul_limbs_unchecked(),
        }
    }

    /// Computes one quotient digit and its n-limb remainder.
    /// The reciprocal supplies the leading two remainder limbs; only the low
    /// divisor prefix needs multiply-subtract. Equality of the leading pairs
    /// gives digit B-1 and its leading remainder directly by addition.
    /// The low n-1 remainder limbs replace `window`; its high limb remains
    /// in the caller's recurrence until the complete remainder is requested.
    ///
    /// The divisor is the prepared divisor or a high suffix with n >= 3 limbs.
    /// The window has n initialized limbs, `u2` supplies its high guard, and
    /// the preceding remainder is below the divisor. The prepared leading
    /// pair, reciprocal and subtraction kernel are retained together.
    #[expect(
        clippy::inline_always,
        reason = "Pinned Algorithm D measurements and emitted assembly show that outlining this per-digit step reloads cached divisor state and adds calls inside both quotient loops."
    )]
    #[inline(always)]
    fn divide_digit(&self, window: &mut [Limb], divisor: &[Limb], u2: Limb) -> (Limb, Limb) {
        let (d1, d0, inverse, sub_mul) = (self.high.get(), self.low, self.inverse, self.sub_mul);
        // SAFETY: full and short division retain at least three initialized
        // divisor limbs. The fixed leading triple leaves n-3 low digits.
        let (below_three, _) = unsafe { divisor.split_last_chunk::<3>().unwrap_unchecked() };
        // SAFETY: adding the triple reconstructs the materialized width;
        // its first digit makes the multiply-subtract prefix n-2 nonempty.
        let (lower_width, n) = unsafe {
            (
                below_three.len().unchecked_add(1),
                below_three.len().unchecked_add(3),
            )
        };
        // SAFETY: the caller supplies n initialized window limbs and its high
        // guard; the two high window digits are within that complete span.
        let (u1, u0) = unsafe {
            (
                *window.get_unchecked(n.unchecked_sub(1)),
                *window.get_unchecked(n.unchecked_sub(2)),
            )
        };
        debug_assert!(
            (u2, u1) <= (d1, d0),
            "the preceding remainder is below the normalized divisor"
        );
        if (u2, u1) == (d1, d0) {
            // Let U = R*B+a with 0 <= a < B and R < D. Equal leading
            // pairs give delta = D-R < B^(n-2). Normalization proves
            // delta*B < B^(n-1) <= D, so U-(B-1)*D = D-delta*B+a > 0.
            // U < B*D therefore makes B-1 exact, with no add-back.
            // The leading triple minus (B-1)*(d1*B+d0) is d1*B+d0+u0.
            // Its high digit may be B before the low-prefix borrow, so both
            // high updates use reduction modulo B rather than nonoverflowing arithmetic.
            let (low, leading_carry) = d0.overflowing_add(u0);
            let high = d1.wrapping_add(Limb::from(leading_carry));
            // SAFETY: the disjoint initialized low prefixes have n-2>0
            // limbs. Product carry < B-1 plus binary borrow fits one limb.
            let correction = unsafe {
                let (carry, borrow) = sub_mul(
                    window.as_mut_ptr(),
                    divisor.as_ptr(),
                    lower_width,
                    Limb::MAX,
                );
                carry.unchecked_add(borrow)
            };
            let (next_low, borrow) = low.overflowing_sub(correction);
            let upper = high.wrapping_sub(Limb::from(borrow));
            // SAFETY: n>=3 and the n-limb window contains this initialized slot.
            unsafe {
                *window.get_unchecked_mut(n.unchecked_sub(2)) = next_low;
            }
            return (Limb::MAX, upper);
        }
        // Strict inequality proves the reciprocal quotient fits one limb.
        let (mut quotient, high, low) = Division::udiv_qr_3by2(u2, u1, u0, d1, d0, inverse);
        // U = H*B^(n-2) + L and D = T*B^(n-2) + K. The
        // reciprocal already computed H = quotient*T + (high*B + low).
        // Only L - quotient*K remains; recomputing quotient*T would repeat
        // two limb products and their carry propagation.
        // SAFETY: n > 2; the low n-2 initialized limbs of each disjoint
        // owner satisfy the selected kernel's pointer contract. A scalar
        // product carry is below quotient (or zero for quotient zero), and
        // borrow is binary, so their sum fits Limb on every target.
        let correction = unsafe {
            let (carry, borrow) =
                sub_mul(window.as_mut_ptr(), divisor.as_ptr(), lower_width, quotient);
            carry.unchecked_add(borrow)
        };
        let (next_low, low_borrow) = low.overflowing_sub(correction);
        let (mut upper, negative) = high.overflowing_sub(Limb::from(low_borrow));
        // SAFETY: n > 2 and window has n initialized limbs. This slot
        // follows the n-2 outputs written by sub_mul.
        unsafe {
            *window.get_unchecked_mut(n.unchecked_sub(2)) = next_low;
        }
        if negative {
            // SAFETY: a zero digit subtracts nothing and cannot underflow.
            // The negative residue therefore proves quotient > 0.
            quotient = unsafe { quotient.unchecked_sub(1) };
            // A normalized 3/2 estimate is at most one too large. Add-back
            // restores the low n-1 limbs and the retained upper limb modulo
            // B^n; the discarded carry cancels the negative high guard.
            // SAFETY: n >= 3 bounds both initialized n-1-limb prefixes.
            let carry = unsafe {
                let width = n.unchecked_sub(1);
                Addition::add_slice_in_place(
                    window.get_unchecked_mut(..width),
                    divisor.get_unchecked(..width),
                )
            };
            upper = upper.wrapping_add(d1).wrapping_add(carry);
        }
        (quotient, upper)
    }
}
