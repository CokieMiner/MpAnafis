//! Extended Euclid and modular inversion.
//!
//! The loop is Lehmer-accelerated: a 2x2 transition matrix is simulated on the
//! leading limbs and applied in bulk, so a full-width division runs only when
//! the simulation stalls. The Bezout cofactors are maintained alongside the
//! remainders by the same matrix.

use core::{
    cmp::{Ordering, max},
    mem::swap,
};

use alloc::vec::Vec;

use super::{ArchKernels, DivScratch, Division, Gcd, InternalArbiUint, Limb};

impl Division {
    /// Computes the modular inverse of `a` modulo `m` using the extended
    /// Euclidean algorithm.
    ///
    /// Returns `None` when the inverse does not exist (i.e. when
    /// `gcd(a, m) != 1`). `m` must be non-zero.
    pub fn mod_inverse(a: &InternalArbiUint, m: &InternalArbiUint) -> Option<InternalArbiUint> {
        debug_assert!(
            !m.is_zero(),
            "modular inversion requires a non-zero modulus"
        );
        if m.is_one() {
            return Some(InternalArbiUint::zero());
        }

        // The units 1 and -1 have themselves as inverses. The predecessor check
        // is limb-wise so the common `a = m - 1` case avoids extended-GCD state
        // and a temporary subtraction of the modulus.
        if a.is_one() {
            return Some(InternalArbiUint::one());
        }
        if is_modulus_predecessor(a, m) {
            return Some(a.clone());
        }

        let (gcd, s, _) = Self::extended_gcd(a, m);
        if !gcd.is_one() || s.is_zero() {
            return None;
        }
        Some(s)
    }

    /// Returns `(gcd(a, b), s, t)` with `a * s - b * t` equal to the gcd, up to
    /// the sign convention fixed by the final parity correction.
    #[allow(
        clippy::too_many_lines,
        reason = "Extended GCD involves many repetitive swap and update steps; keeping it inline improves performance; non-performance lint."
    )]
    pub fn extended_gcd(
        a: &InternalArbiUint,
        b: &InternalArbiUint,
    ) -> (InternalArbiUint, InternalArbiUint, InternalArbiUint) {
        if a.is_zero() {
            return (b.clone(), InternalArbiUint::zero(), InternalArbiUint::one());
        }
        if b.is_zero() || a.cmp(b) == Ordering::Equal {
            return (a.clone(), InternalArbiUint::one(), InternalArbiUint::zero());
        }

        let mut r0 = a.clone();
        let mut r1 = b.clone();

        // Absolute values of coefficients
        let mut s0 = InternalArbiUint::one();
        let mut s1 = InternalArbiUint::zero();
        let mut t0 = InternalArbiUint::zero();
        let mut t1 = InternalArbiUint::one();

        let mut q = InternalArbiUint::zero();
        let mut next_r = InternalArbiUint::zero();
        let mut temp = InternalArbiUint::zero();
        let mut saved_r0 = InternalArbiUint::zero();
        let mut saved_r1 = InternalArbiUint::zero();
        let mut u_backup = Vec::new();
        let mut v_backup = Vec::new();

        let mut scratch = DivScratch::default();
        let mut step: usize = 0;

        while !r1.is_zero() {
            if r1.limbs().len() < 4 {
                Self::div_rem_into(&r0, &r1, &mut q, &mut next_r, &mut scratch);
                swap(&mut r0, &mut r1);
                swap(&mut r1, &mut next_r);

                temp.assign_product_with_scratch(&q, &s1, &mut scratch.mul_scratch);
                let mut next_s = s0.clone();
                next_s.add_assign(&temp);
                swap(&mut s0, &mut s1);
                swap(&mut s1, &mut next_s);

                temp.assign_product_with_scratch(&q, &t1, &mut scratch.mul_scratch);
                let mut next_t = t0.clone();
                next_t.add_assign(&temp);
                swap(&mut t0, &mut t1);
                swap(&mut t1, &mut next_t);

                step = step.wrapping_add(1);
                continue;
            }

            let (u0, v0, u1, v1, even) = if r0.limbs().len() == r1.limbs().len()
                && r0.limbs().len() >= Gcd::WIDE_LEHMER_THRESHOLD
            {
                let (u_hat, v_hat) = Gcd::extract_top_two_limbs(r0.limbs(), r1.limbs());
                Gcd::lehmer_simulate_wide(u_hat, v_hat)
            } else {
                let (u_hat, v_hat) = Gcd::extract_top_limb(r0.limbs(), r1.limbs());
                Gcd::lehmer_simulate(u_hat, v_hat)
            };

            let is_identity = u0 == 1 && v0 == 0 && u1 == 0 && v1 == 1;
            if is_identity {
                Self::div_rem_into(&r0, &r1, &mut q, &mut next_r, &mut scratch);
                swap(&mut r0, &mut r1);
                swap(&mut r1, &mut next_r);

                temp.assign_product_with_scratch(&q, &s1, &mut scratch.mul_scratch);
                let mut next_s = s0.clone();
                next_s.add_assign(&temp);
                swap(&mut s0, &mut s1);
                swap(&mut s1, &mut next_s);

                temp.assign_product_with_scratch(&q, &t1, &mut scratch.mul_scratch);
                let mut next_t = t0.clone();
                next_t.add_assign(&temp);
                swap(&mut t0, &mut t1);
                swap(&mut t1, &mut next_t);

                step = step.wrapping_add(1);
            } else {
                saved_r0.clone_from(&r0);
                saved_r1.clone_from(&r1);
                let ok = Gcd::lehmer_update(
                    &mut r0,
                    &mut r1,
                    &mut u_backup,
                    &mut v_backup,
                    u0,
                    v0,
                    u1,
                    v1,
                    even,
                );
                if ok {
                    update_abs_coeffs(&mut s0, &mut s1, u0, v0, u1, v1);
                    update_abs_coeffs(&mut t0, &mut t1, u0, v0, u1, v1);
                    if !even {
                        step = step.wrapping_add(1);
                    }
                } else {
                    swap(&mut r0, &mut saved_r0);
                    swap(&mut r1, &mut saved_r1);

                    Self::div_rem_into(&r0, &r1, &mut q, &mut next_r, &mut scratch);
                    swap(&mut r0, &mut r1);
                    swap(&mut r1, &mut next_r);

                    temp.assign_product_with_scratch(&q, &s1, &mut scratch.mul_scratch);
                    let mut next_s = s0.clone();
                    next_s.add_assign(&temp);
                    swap(&mut s0, &mut s1);
                    swap(&mut s1, &mut next_s);

                    temp.assign_product_with_scratch(&q, &t1, &mut scratch.mul_scratch);
                    let mut next_t = t0.clone();
                    next_t.add_assign(&temp);
                    swap(&mut t0, &mut t1);
                    swap(&mut t1, &mut next_t);

                    step = step.wrapping_add(1);
                }
            }
        }

        let mut final_s = InternalArbiUint::zero();
        let mut final_t = InternalArbiUint::zero();
        let mut dump = InternalArbiUint::zero();

        if s0.cmp(b) == Ordering::Less {
            final_s.clone_from(&s0);
        } else {
            Self::div_rem_into(&s0, b, &mut dump, &mut final_s, &mut scratch);
        }

        if t0.cmp(a) == Ordering::Less {
            final_t.clone_from(&t0);
        } else {
            Self::div_rem_into(&t0, a, &mut dump, &mut final_t, &mut scratch);
        }

        if step & 1 == 0 {
            if !final_t.is_zero() {
                let mut t_diff = a.clone();
                t_diff.sub_assign(&final_t);
                final_t = t_diff;
            }
        } else if !final_s.is_zero() {
            let mut s_diff = b.clone();
            s_diff.sub_assign(&final_s);
            final_s = s_diff;
        }

        (r0, final_s, final_t)
    }
}

/// Returns whether `value + 1 == modulus` without constructing `value + 1`.
///
/// The equal-length case propagates one carry through the value. The only
/// valid length-changing case is `modulus = B^n` and `value = B^n - 1`, where
/// `B = 2^LIMB_BITS`; normalized representations make those limb tests
/// sufficient on every supported pointer width.
fn is_modulus_predecessor(value: &InternalArbiUint, modulus: &InternalArbiUint) -> bool {
    let value_limbs = value.limbs();
    let modulus_limbs = modulus.limbs();
    if value_limbs.is_empty() || modulus_limbs.is_empty() {
        return false;
    }

    if value_limbs.len() == modulus_limbs.len() {
        let mut carry = 1;
        for (&value_limb, &modulus_limb) in value_limbs.iter().zip(modulus_limbs) {
            let (sum, overflow) = value_limb.overflowing_add(carry);
            if sum != modulus_limb {
                return false;
            }
            carry = Limb::from(overflow);
        }
        return carry == 0;
    }

    if modulus_limbs.len() != value_limbs.len().wrapping_add(1)
        || modulus_limbs.last().copied() != Some(1)
    {
        return false;
    }

    modulus_limbs
        .iter()
        .take(value_limbs.len())
        .all(|&limb| limb == 0)
        && value_limbs.iter().all(|&limb| limb == Limb::MAX)
}

/// Applies the Lehmer transition matrix to a cofactor pair in place.
#[allow(
    unsafe_code,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "u0/v0/u1/v1 represent the Lehmer transition matrix; max_len+1 ensures slice access is bounded. Using 'as' avoids checked conversions and is branchless even on 16-bit targets."
)]
fn update_abs_coeffs(
    s0: &mut InternalArbiUint,
    s1: &mut InternalArbiUint,
    u0: Limb,
    v0: Limb,
    u1: Limb,
    v1: Limb,
) {
    let max_len = max(s0.limbs().len(), s1.limbs().len());
    // Each result is the sum of two products by single-limb coefficients:
    //
    //     s0' = u0*s0 + v0*s1,   s1' = u1*s0 + v1*s1.
    //
    // Either sum is strictly below 2*B^(max_len+1), so max_len+2 limbs are
    // sufficient.  The extra limb is necessary: adding the two products can
    // produce a carry in {0, 1} above the ordinary max_len+1 product width.
    s0.resize(max_len.wrapping_add(2));
    s1.resize(max_len.wrapping_add(2));

    let s0_limbs = s0.limbs_mut();
    let s1_limbs = s1.limbs_mut();

    let mut carry0 = (0, 0);
    let mut carry1 = (0, 0);

    for i in 0..max_len {
        // SAFETY: both slices were resized to `max_len + 2`, so `i < max_len` is in bounds.
        let x = unsafe { *s0_limbs.get_unchecked(i) };
        // SAFETY: `s1_limbs` has length `max_len + 2`, so `i` is within bounds.
        let y = unsafe { *s1_limbs.get_unchecked(i) };

        let (result0, next_carry0) = add_two_limb_products(x, u0, y, v0, carry0);
        let (result1, next_carry1) = add_two_limb_products(x, u1, y, v1, carry1);
        // SAFETY: i < max_len < max_len + 2 for both resized slices.
        unsafe {
            *s0_limbs.get_unchecked_mut(i) = result0;
            *s1_limbs.get_unchecked_mut(i) = result1;
        }
        carry0 = next_carry0;
        carry1 = next_carry1;
    }

    // SAFETY: both slices have max_len + 2 elements.  carry.N.0 is the limb at
    // B^max_len and carry.N.1 is its at-most-one-bit carry at B^(max_len+1).
    unsafe {
        *s0_limbs.get_unchecked_mut(max_len) = carry0.0;
        *s0_limbs.get_unchecked_mut(max_len.wrapping_add(1)) = carry0.1;
        *s1_limbs.get_unchecked_mut(max_len) = carry1.0;
        *s1_limbs.get_unchecked_mut(max_len.wrapping_add(1)) = carry1.1;
    }

    s0.normalize();
    s1.normalize();
}

/// Returns one base-`B` limb of `x*x_coeff + y*y_coeff + carry` and its carry.
///
/// The incoming and outgoing carries are represented as `(low, high)`, where
/// `high` is zero or one.  Two full limb products may sum to almost `2*B^2`,
/// one bit wider than [`DoubleLimb`], so retaining that high bit is required
/// for an exact Lehmer coefficient update.
///
/// [`DoubleLimb`]: super::DoubleLimb
fn add_two_limb_products(
    x: Limb,
    x_coeff: Limb,
    y: Limb,
    y_coeff: Limb,
    carry: (Limb, Limb),
) -> (Limb, (Limb, Limb)) {
    let (x_lo, x_hi) = ArchKernels::mul_limb_lo_hi(x, x_coeff);
    let (y_lo, y_hi) = ArchKernels::mul_limb_lo_hi(y, y_coeff);

    let (lo_sum, lo_xy_overflow) = x_lo.overflowing_add(y_lo);
    let (result, lo_carry_overflow) = lo_sum.overflowing_add(carry.0);
    let carry_from_low = Limb::from(lo_xy_overflow).wrapping_add(Limb::from(lo_carry_overflow));

    let (hi_sum, hi_xy_overflow) = x_hi.overflowing_add(y_hi);
    let (hi_with_low_carry, hi_low_overflow) = hi_sum.overflowing_add(carry_from_low);
    let (next_low, hi_input_overflow) = hi_with_low_carry.overflowing_add(carry.1);

    // The exact high sum is < 2*B, so at most one of these additions can
    // overflow.  OR therefore preserves its single carry bit without a branch.
    let next_high = Limb::from(hi_xy_overflow | hi_low_overflow | hi_input_overflow);
    (result, (next_low, next_high))
}
