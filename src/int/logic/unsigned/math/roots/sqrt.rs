//! Square root kernels: the basecases and Zimmermann's recursive Karatsuba
//! square root.

#![allow(
    unsafe_code,
    reason = "Low-level square root uses unchecked slices for peak performance"
)]

use core::{
    cmp::{Ordering, min},
    mem::swap,
};

use super::{Division, DoubleLimb, InternalArbiUint, LIMB_BITS, Limb, SqrtScratch};

/// Floor square root by Newton iteration, at doubling precision.
///
/// A full-width iteration from a one-bit estimate would run one full-precision
/// division per step. Instead the value is truncated to the leading half of the
/// operand, rooted recursively, and the result used as the seed for a single
/// correction at full width. Because Newton doubles the number of correct bits
/// each step, that seed is already accurate to within one unit, so one
/// correction suffices and the total cost is dominated by the final division
/// rather than by `log2(bits)` of them.
impl SqrtScratch {
    /// Computes the floor square root with Newton iteration.
    pub fn isqrt_basecase(&mut self, a: &InternalArbiUint) -> InternalArbiUint {
        if a.is_zero() || a.is_one() {
            return a.clone();
        }

        let limbs = a.limbs();
        if limbs.len() == 1 {
            // SAFETY: len == 1 so get_unchecked is safe
            let val = unsafe { *limbs.get_unchecked(0) };
            return InternalArbiUint::from_limb(val.isqrt());
        }

        if limbs.len() == 2 {
            // SAFETY: limbs length is exactly 2, so 0 and 1 are in bounds
            let lo = unsafe { *limbs.get_unchecked(0) };
            // SAFETY: limbs length is exactly 2, so 0 and 1 are in bounds
            let hi = unsafe { *limbs.get_unchecked(1) };
            #[allow(
                clippy::as_conversions,
                reason = "Limb is guaranteed to fit within DoubleLimb"
            )]
            let val = ((hi as DoubleLimb) << LIMB_BITS) | (lo as DoubleLimb);
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "isqrt fits in Limb"
            )]
            let root = val.isqrt() as Limb;
            return InternalArbiUint::from_limb(root);
        }

        let bits = a.significant_bits();
        let mut x = seed_estimate(a, bits, self);

        let mut quotient = self.get_temp();
        let mut rem = self.get_temp();
        let mut next = self.get_temp();

        let root = loop {
            Division::div_rem_into(a, &x, &mut quotient, &mut rem, &mut self.div_scratch);
            next.clone_from(&x);
            next.add_assign(&quotient);
            next.shr_assign(1);

            if next.cmp(&x) != Ordering::Less {
                break x;
            }
            swap(&mut x, &mut next);
        };

        self.return_temp(quotient);
        self.return_temp(rem);
        self.return_temp(next);
        root
    }

    /// Computes the floor square root and exact remainder.
    pub fn sqrt_rem_basecase(
        &mut self,
        a: &InternalArbiUint,
    ) -> (InternalArbiUint, InternalArbiUint) {
        let root = self.isqrt_basecase(a);
        let mut square = self.get_temp();
        square.assign_product_with_scratch(&root, &root, &mut self.mul_scratch);
        let mut rem = self.get_temp();
        rem.clone_from(a);
        rem.sub_assign(&square);
        self.return_temp(square);
        (root, rem)
    }
}

/// Produces a Newton seed accurate to roughly the full width.
///
/// `sqrt(a) = sqrt(a >> 2k) << k` up to one unit, so rooting the leading half
/// of `a` and shifting back costs one recursion at half precision instead of a
/// chain of corrections at full precision. Falls back to the classic one-bit
/// estimate when the operand is too narrow for the halving to pay for itself.
fn seed_estimate(a: &InternalArbiUint, bits: usize, scratch: &mut SqrtScratch) -> InternalArbiUint {
    // Below this the recursion costs more than the corrections it saves: the
    // two-limb basecase above already answers everything narrower.
    const HALVING_FLOOR_BITS: usize = 4 * LIMB_BITS;

    if bits < HALVING_FLOOR_BITS {
        let mut estimate = InternalArbiUint::one();
        estimate.shl_assign(bits.wrapping_add(1).wrapping_shr(1));
        return estimate;
    }

    // Halve the operand width, keeping the shift even so the root shifts back
    // by exactly half of it.
    let mut shift = bits >> 1;
    if shift & 1 == 1 {
        shift = shift.wrapping_sub(1);
    }

    let mut head = scratch.get_temp();
    head.clone_from(a);
    head.shr_assign(shift);
    let seed = scratch.isqrt_basecase(&head);
    scratch.return_temp(head);

    // Newton converges from above, so the seed has to be an overestimate for the
    // descending loop to terminate on the floor root. With `t` the root of the
    // truncated head, `a < (t + 1)^2 * 2^shift`, so `(t + 1) << (shift / 2)`
    // strictly exceeds the true root while staying within a relative `1 / t` of
    // it — close enough that the full-width loop needs one or two corrections.
    let mut estimate = seed;
    estimate.add_assign(&InternalArbiUint::one());
    estimate.shl_assign(shift >> 1);
    estimate
}

impl SqrtScratch {
    #[allow(
        clippy::similar_names,
        clippy::too_many_lines,
        clippy::many_single_char_names,
        reason = "Mathematical notation maps s, r, q, u, a, k to the standard Karatsuba square-root algorithm."
    )]
    pub fn sqrt_rem_recursive(
        &mut self,
        a: &InternalArbiUint,
        need_remainder: bool,
    ) -> (InternalArbiUint, InternalArbiUint) {
        let scratch = self;
        let len = a.limbs().len();
        if len <= 2 {
            return scratch.sqrt_rem_basecase(a);
        }

        // Zimmermann's split needs the recursive half `a >> 2k` to cover the top
        // *half* of the operand, which is what `floor(len / 4)` gives. Rounding up
        // instead leaves that half short whenever `len` is not a multiple of four —
        // at `len = 6`, `k = 2` recurses on only 2 of 6 limbs — and a seed that
        // coarse breaks the algorithm's guarantee that one correction suffices. The
        // `max(1)` keeps `k` non-zero for the three-limb case, where `len - 2k` is
        // still a full limb.
        let k = (len >> 2).max(1);
        let limbs = a.limbs();

        let mut n_high = scratch.get_temp();
        if limbs.len() > 2_usize.wrapping_mul(k) {
            // SAFETY: 2*k is within bounds of limbs
            let limbs_slice = unsafe { limbs.get_unchecked(2_usize.wrapping_mul(k)..) };
            n_high.clone_from_slice(limbs_slice);
        }

        let (s1, r1) = scratch.sqrt_rem_recursive(&n_high, true);
        scratch.return_temp(n_high);

        let mut a1 = scratch.get_temp();
        if limbs.len() > k {
            let end = min(2_usize.wrapping_mul(k), limbs.len());
            // SAFETY: k..end is within bounds of limbs
            let limbs_slice = unsafe { limbs.get_unchecked(k..end) };
            a1.clone_from_slice(limbs_slice);
        }

        let mut r1_b = r1;
        r1_b.shl_assign(k.wrapping_mul(LIMB_BITS));
        r1_b.add_assign(&a1);
        scratch.return_temp(a1);

        let mut two_s1 = s1.clone();
        two_s1.shl_assign(1);

        let mut q = scratch.get_temp();
        let mut u = scratch.get_temp();
        Division::div_rem_into(&r1_b, &two_s1, &mut q, &mut u, &mut scratch.div_scratch);

        let mut s = s1;
        s.shl_assign(k.wrapping_mul(LIMB_BITS));
        s.add_assign(&q);

        let mut a0 = scratch.get_temp();
        let end = min(k, limbs.len());
        // SAFETY: 0..end is within bounds of limbs
        let a0_slice = unsafe { limbs.get_unchecked(0..end) };
        a0.clone_from_slice(a0_slice);

        let mut u_b = u;
        u_b.shl_assign(k.wrapping_mul(LIMB_BITS));
        u_b.add_assign(&a0);
        scratch.return_temp(a0);

        let mut q_sq = scratch.get_temp();
        q_sq.assign_product_with_scratch(&q, &q, &mut scratch.mul_scratch);
        scratch.return_temp(q);

        let remainder = if u_b.cmp(&q_sq) == Ordering::Less {
            let remainder = need_remainder.then(|| {
                let r_val = q_sq.sub(&u_b);

                let mut two_s_minus_1 = s.clone();
                two_s_minus_1.shl_assign(1);
                two_s_minus_1.sub_assign(&InternalArbiUint::one());

                two_s_minus_1.sub(&r_val)
            });
            s.sub_assign(&InternalArbiUint::one());
            remainder
        } else if need_remainder {
            Some(u_b.sub(&q_sq))
        } else {
            None
        };
        scratch.return_temp(q_sq);

        (s, remainder.unwrap_or_else(InternalArbiUint::zero))
    }
}
