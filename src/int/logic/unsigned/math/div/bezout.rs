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

use super::{ArchKernels, DivScratch, Division, DoubleLimb, Gcd, InternalMpUint, LIMB_BITS, Limb};

impl Division {
    /// Computes the modular inverse of `a` modulo `m` using the extended
    /// Euclidean algorithm.
    ///
    /// Returns `None` when the inverse does not exist (i.e. when
    /// `gcd(a, m) != 1`). `m` must be non-zero.
    pub fn mod_inverse(a: &InternalMpUint, m: &InternalMpUint) -> Option<InternalMpUint> {
        debug_assert!(
            !m.is_zero(),
            "modular inversion requires a non-zero modulus"
        );
        if m.is_one() {
            return Some(InternalMpUint::zero());
        }

        // The units 1 and -1 have themselves as inverses. The predecessor check
        // is limb-wise so the common `a = m - 1` case avoids extended-GCD state
        // and a temporary subtraction of the modulus.
        if a.is_one() {
            return Some(InternalMpUint::one());
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
        a: &InternalMpUint,
        b: &InternalMpUint,
    ) -> (InternalMpUint, InternalMpUint, InternalMpUint) {
        if a.is_zero() {
            return (b.clone(), InternalMpUint::zero(), InternalMpUint::one());
        }
        if b.is_zero() || a.cmp(b) == Ordering::Equal {
            return (a.clone(), InternalMpUint::one(), InternalMpUint::zero());
        }

        let a_limbs = a.limbs();
        let b_limbs = b.limbs();
        if a_limbs.len() == 1 && b_limbs.len() == 1 {
            // SAFETY: a_limbs.len() == 1 and b_limbs.len() == 1
            let (gcd_val, s_val, t_val) =
                unsafe { extended_gcd_1(*a_limbs.get_unchecked(0), *b_limbs.get_unchecked(0)) };
            return (
                InternalMpUint::from_limb(gcd_val),
                InternalMpUint::from_limb(s_val),
                InternalMpUint::from_limb(t_val),
            );
        }

        let mut r0 = a.clone();
        let mut r1 = b.clone();

        // Absolute value of coefficient for `a`
        let mut s0 = InternalMpUint::one();
        let mut s1 = InternalMpUint::zero();
        s0.reserve(a.limbs().len().wrapping_add(4));
        s1.reserve(a.limbs().len().wrapping_add(4));

        let mut q = InternalMpUint::zero();
        let mut next_r = InternalMpUint::zero();
        let mut temp = InternalMpUint::zero();
        let mut u_backup = Vec::new();
        let mut v_backup = Vec::new();

        let mut scratch = DivScratch::default();
        let mut step: usize = 0;

        while !r1.is_zero() {
            if r1.limbs().len() < 2 {
                Self::div_rem_into(&r0, &r1, &mut q, &mut next_r, &mut scratch);
                swap(&mut r0, &mut r1);
                swap(&mut r1, &mut next_r);

                temp.assign_product_with_scratch(&q, &s1, &mut scratch.mul_scratch);
                s0.add_assign(&temp);
                swap(&mut s0, &mut s1);

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
                s0.add_assign(&temp);
                swap(&mut s0, &mut s1);

                step = step.wrapping_add(1);
            } else {
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
                    if !even {
                        step = step.wrapping_add(1);
                    }
                } else {
                    Self::div_rem_into(&r0, &r1, &mut q, &mut next_r, &mut scratch);
                    swap(&mut r0, &mut r1);
                    swap(&mut r1, &mut next_r);

                    temp.assign_product_with_scratch(&q, &s1, &mut scratch.mul_scratch);
                    s0.add_assign(&temp);
                    swap(&mut s0, &mut s1);

                    step = step.wrapping_add(1);
                }
            }
        }

        let mut final_s = InternalMpUint::zero();
        let mut final_t = InternalMpUint::zero();
        let mut dump = InternalMpUint::zero();

        if step & 1 == 0 {
            if s0.cmp(b) == Ordering::Less {
                final_s.clone_from(&s0);
            } else {
                Self::div_rem_into(&s0, b, &mut dump, &mut final_s, &mut scratch);
            }

            let mut as0 = a.clone();
            as0.mul_assign(&s0);
            as0.sub_assign(&r0);
            let t0 = as0.div(b);

            if t0.cmp(a) == Ordering::Less {
                final_t = t0;
            } else {
                Self::div_rem_into(&t0, a, &mut dump, &mut final_t, &mut scratch);
            }

            if !final_t.is_zero() {
                let mut t_diff = a.clone();
                t_diff.sub_assign(&final_t);
                final_t = t_diff;
            }
        } else {
            let mut s_rem = InternalMpUint::zero();
            if s0.cmp(b) == Ordering::Less {
                s_rem.clone_from(&s0);
            } else {
                Self::div_rem_into(&s0, b, &mut dump, &mut s_rem, &mut scratch);
            }

            if !s_rem.is_zero() {
                let mut s_diff = b.clone();
                s_diff.sub_assign(&s_rem);
                final_s = s_diff;
            }

            let mut as0 = a.clone();
            as0.mul_assign(&s0);
            as0.add_assign(&r0);
            let t0 = as0.div(b);

            if t0.cmp(a) == Ordering::Less {
                final_t = t0;
            } else {
                Self::div_rem_into(&t0, a, &mut dump, &mut final_t, &mut scratch);
            }
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
fn is_modulus_predecessor(value: &InternalMpUint, modulus: &InternalMpUint) -> bool {
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
    s0: &mut InternalMpUint,
    s1: &mut InternalMpUint,
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
#[inline]
#[allow(
    clippy::similar_names,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "DoubleLimb hardware arithmetic for fused 2-product carry accumulation."
)]
fn add_two_limb_products(
    x: Limb,
    x_coeff: Limb,
    y: Limb,
    y_coeff: Limb,
    carry: (Limb, Limb),
) -> (Limb, (Limb, Limb)) {
    let p1 = (x as DoubleLimb).wrapping_mul(x_coeff as DoubleLimb);
    let p2 = (y as DoubleLimb).wrapping_mul(y_coeff as DoubleLimb);
    let (sum_p, c1) = p1.overflowing_add(p2);
    let carry_in = (carry.0 as DoubleLimb) | ((carry.1 as DoubleLimb) << LIMB_BITS);
    let (sum_all, c2) = sum_p.overflowing_add(carry_in);
    let result = sum_all as Limb;
    let next_low = (sum_all >> LIMB_BITS) as Limb;
    let next_high = Limb::from(c1).wrapping_add(Limb::from(c2));
    (result, (next_low, next_high))
}

/// Single-limb extended GCD computed entirely in CPU registers.
#[inline]
#[allow(
    unsafe_code,
    clippy::similar_names,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "DoubleLimb arithmetic on single-limb operands uses DoubleLimb / Limb primitive casts for in-register register math."
)]
fn extended_gcd_1(a_val: Limb, b_val: Limb) -> (Limb, Limb, Limb) {
    let mut r0 = a_val;
    let mut r1 = b_val;
    let mut s0: DoubleLimb = 1;
    let mut s1: DoubleLimb = 0;
    let mut step: usize = 0;

    while r1 != 0 {
        // SAFETY: r1 is non-zero in this loop; rem_hi is 0 < r1.
        let (q_val, r) = unsafe { ArchKernels::divrem_1_unchecked(r0, 0, r1) };
        r0 = r1;
        r1 = r;
        let next_s = s0.wrapping_add((q_val as DoubleLimb).wrapping_mul(s1));
        s0 = s1;
        s1 = next_s;
        step = step.wrapping_add(1);
    }

    let b_double = b_val as DoubleLimb;
    let a_double = a_val as DoubleLimb;

    let (final_s, final_t) = if step & 1 == 0 {
        let s_coeff = if s0 < b_double {
            s0
        } else {
            let s_lo = s0 as Limb;
            let s_hi = (s0 >> LIMB_BITS) as Limb;
            // SAFETY: b_val is non-zero; s0 < 2*b_double so s_hi < b_val.
            let (_, rem) = unsafe { ArchKernels::divrem_1_unchecked(s_lo, s_hi, b_val) };
            rem as DoubleLimb
        };
        let (as0_lo, as0_hi) = ArchKernels::mul_limb_lo_hi(a_val, s0 as Limb);
        let (num_lo, borrow) = as0_lo.overflowing_sub(r0);
        let num_hi = as0_hi.wrapping_sub(Limb::from(borrow));
        // SAFETY: (a*s0 - r0) < a*b, so num_hi < b_val and quotient fits in Limb.
        let (t0, _) = unsafe { ArchKernels::divrem_1_unchecked(num_lo, num_hi, b_val) };
        let t_val = if (t0 as DoubleLimb) < a_double {
            t0
        } else {
            // SAFETY: a_val is non-zero; rem_hi is 0 < a_val.
            let (_, rem) = unsafe { ArchKernels::divrem_1_unchecked(t0, 0, a_val) };
            rem
        };
        let final_t = if t_val != 0 {
            a_val.wrapping_sub(t_val)
        } else {
            0
        };
        (s_coeff as Limb, final_t)
    } else {
        let s_rem = if s0 < b_double {
            s0
        } else {
            let s_lo = s0 as Limb;
            let s_hi = (s0 >> LIMB_BITS) as Limb;
            // SAFETY: b_val is non-zero; s0 < 2*b_double so s_hi < b_val.
            let (_, rem) = unsafe { ArchKernels::divrem_1_unchecked(s_lo, s_hi, b_val) };
            rem as DoubleLimb
        };
        let final_s = if s_rem != 0 {
            b_val.wrapping_sub(s_rem as Limb)
        } else {
            0
        };
        let (as0_lo, as0_hi) = ArchKernels::mul_limb_lo_hi(a_val, s0 as Limb);
        let (num_lo, carry) = as0_lo.overflowing_add(r0);
        let num_hi = as0_hi.wrapping_add(Limb::from(carry));
        // SAFETY: (a*s0 + r0) <= a*b, so num_hi < b_val and quotient fits in Limb.
        let (t0, _) = unsafe { ArchKernels::divrem_1_unchecked(num_lo, num_hi, b_val) };
        let final_t = if (t0 as DoubleLimb) < a_double {
            t0
        } else {
            // SAFETY: a_val is non-zero; rem_hi is 0 < a_val.
            let (_, rem) = unsafe { ArchKernels::divrem_1_unchecked(t0, 0, a_val) };
            rem
        };
        (final_s, final_t)
    };

    (r0, final_s, final_t)
}
