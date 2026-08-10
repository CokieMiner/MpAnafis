//! The Möller–Granlund 3-by-2 division primitive.
//!
//! Together these turn each Algorithm D quotient digit into a handful of
//! multiplications: [`Division::invert_pi1`] builds a reciprocal for the
//! normalized two-limb divisor top once per division, and
//! [`Division::udiv_qr_3by2`] consumes it
//! once per digit. See N. Möller and T. Granlund, "Improved Division by
//! Invariant Integers", IEEE TC 2011, Algorithms 4 and 6.

use super::{Division, DoubleLimb, LIMB_BITS, Limb};

impl Division {
    /// Computes the 3-by-2 reciprocal of the normalized two-limb divisor top
    /// `(d1, d0)`, where `d1`'s most-significant bit is set.
    ///
    /// The result is `floor((B^3 - 1) / (d1 * B + d0)) - B` (with `B` the limb
    /// radix), the value [`Division::udiv_qr_3by2`] consumes.
    #[allow(
        clippy::inline_always,
        reason = "Computed once per division on the hot path; inlining keeps the divisor top in registers."
    )]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Limb<->DoubleLimb casts are exact; truncating a value proven < B to a Limb is the intended reduction mod B and is branchless on every target width."
    )]
    #[allow(
        clippy::similar_names,
        reason = "d1/d0/t1/t0/p/v are the standard reciprocal-algorithm naming convention."
    )]
    #[inline(always)]
    pub const fn invert_pi1(d1: Limb, d0: Limb) -> Limb {
        // 2/1 inverse of d1: floor((B^2 - 1) / d1) - B. Because d1 is normalized
        // the quotient lies in [B, 2B), so truncating it to a Limb subtracts B
        // exactly.
        // SAFETY: d1 is the normalized divisor top limb (most-significant bit
        // set), hence non-zero, so the division cannot fault.
        let quot = unsafe {
            DoubleLimb::MAX
                .checked_div(d1 as DoubleLimb)
                .unwrap_unchecked()
        };
        let mut v = quot as Limb;

        let mut p = d1.wrapping_mul(v).wrapping_add(d0);
        if p < d0 {
            v = v.wrapping_sub(1);
            let mask = if p >= d1 { Limb::MAX } else { 0 };
            p = p.wrapping_sub(d1);
            v = v.wrapping_add(mask);
            p = p.wrapping_sub(mask & d1);
        }

        let prod = (d0 as DoubleLimb).wrapping_mul(v as DoubleLimb);
        let t1 = prod.wrapping_shr(LIMB_BITS as u32) as Limb;
        let t0 = prod as Limb;
        p = p.wrapping_add(t1);
        if p < t1 {
            v = v.wrapping_sub(1);
            if p >= d1 && (p > d1 || t0 >= d0) {
                v = v.wrapping_sub(1);
            }
        }
        v
    }

    /// Divides the three-limb numerator `(n2, n1, n0)` by the normalized
    /// two-limb divisor `(d1, d0)` using the precomputed reciprocal `dinv`,
    /// returning the one-limb quotient with the two-limb remainder `(r1, r0)`.
    ///
    /// # Preconditions
    ///
    /// `(n2, n1) < (d1, d0)` as a two-limb value (so the quotient fits in one
    /// limb) and `d1`'s most-significant bit is set. Uses only multiplications;
    /// no hardware division.
    #[allow(
        clippy::inline_always,
        reason = "Called once per quotient digit on the division hot path; inlining removes the call and exposes the multiplies to the surrounding loop."
    )]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Limb<->DoubleLimb casts are exact; extracting the high/low limb of a DoubleLimb product via shift-and-truncate is branchless on every target width."
    )]
    #[allow(
        clippy::similar_names,
        reason = "n2/n1/n0, d1/d0, r1/r0, t1/t0, q1/q0 are the standard 3-by-2 division naming convention."
    )]
    #[inline(always)]
    pub fn udiv_qr_3by2(
        n2: Limb,
        n1: Limb,
        n0: Limb,
        d1: Limb,
        d0: Limb,
        dinv: Limb,
    ) -> (Limb, Limb, Limb) {
        // q1:q0 = n2 * dinv + (n2:n1)  -> q1 is the initial quotient estimate.
        let prod = (n2 as DoubleLimb).wrapping_mul(dinv as DoubleLimb);
        let mut q1 = prod.wrapping_shr(LIMB_BITS as u32) as Limb;
        let (q0, carry_q) = (prod as Limb).overflowing_add(n1);
        q1 = q1.wrapping_add(n2).wrapping_add(Limb::from(carry_q));

        // (r1:r0) = (n1:n0) - q1 * (d1:d0), all with the estimate q1.
        let mut r1 = n1.wrapping_sub(q1.wrapping_mul(d1));
        let (mut r0, borrow_d) = n0.overflowing_sub(d0);
        r1 = r1.wrapping_sub(d1).wrapping_sub(Limb::from(borrow_d));
        let prod_d0 = (q1 as DoubleLimb).wrapping_mul(d0 as DoubleLimb);
        let t1 = prod_d0.wrapping_shr(LIMB_BITS as u32) as Limb;
        let t0 = prod_d0 as Limb;
        let (r0b, borrow_t) = r0.overflowing_sub(t0);
        r0 = r0b;
        r1 = r1.wrapping_sub(t1).wrapping_sub(Limb::from(borrow_t));

        q1 = q1.wrapping_add(1);
        // If r1 >= q0 the estimate was one too large: undo the increment and add
        // the divisor back into the remainder (mask is all-ones == -1 if true).
        let mask = if r1 >= q0 { Limb::MAX } else { 0 };
        q1 = q1.wrapping_add(mask);
        let (r0c, carry_r) = r0.overflowing_add(mask & d0);
        r0 = r0c;
        r1 = r1.wrapping_add(mask & d1).wrapping_add(Limb::from(carry_r));

        // Final conditional correction: at most one more subtraction.
        if r1 >= d1 && (r1 > d1 || r0 >= d0) {
            q1 = q1.wrapping_add(1);
            let (r0d, borrow_f) = r0.overflowing_sub(d0);
            r0 = r0d;
            r1 = r1.wrapping_sub(d1).wrapping_sub(Limb::from(borrow_f));
        }

        (q1, r1, r0)
    }
}
