//! Jacobi symbol with Lehmer-accelerated Euclidean steps.

use core::{cmp::Ordering, mem::swap};

use alloc::vec::Vec;

use super::{DivScratch, Division, Gcd, InternalArbiUint, Limb};

impl InternalArbiUint {
    /// Computes the Jacobi symbol `(self | other)`.
    ///
    /// `other` must be odd and positive. Returns `1` if `self` is a quadratic
    /// residue modulo `other`, `-1` if it is not, and `0` when the operands are
    /// not coprime.
    #[must_use]
    pub fn jacobi_symbol(&self, other: &Self) -> i8 {
        debug_assert!(
            !other.is_zero() && other.is_odd(),
            "Jacobi modulus must be non-zero and odd"
        );
        if other.is_one() {
            return 1;
        }

        // Reduce a modulo n.
        let mut a = Self::zero();
        let mut scratch = DivScratch::default();
        Division::rem_into(self, other, &mut a, &mut scratch);
        let mut n = other.clone();
        let mut t: i8 = 1;

        if a.is_zero() {
            return 0;
        }

        // Remove factors of 2 from a.
        if !a.is_odd() {
            let tz = a.trailing_zeros();
            a.shr_assign(tz);
            // Apply (2/n) factor.
            if tz & 1 == 1 {
                let n_mod_8 = n.limbs().first().copied().unwrap_or(0) & 7;
                if n_mod_8 == 3 || n_mod_8 == 5 {
                    t = t.wrapping_neg();
                }
            }
        }

        let mut u_backup = Vec::new();
        let mut v_backup = Vec::new();
        loop {
            if n.is_one() || a.is_one() {
                return t;
            }
            let cmp = a.cmp(&n);
            if cmp == Ordering::Equal {
                return 0;
            }
            if cmp == Ordering::Less {
                swap(&mut a, &mut n);
                // Quadratic reciprocity for two odd numbers.
                let a_mod_4 = a.limbs().first().copied().unwrap_or(0) & 3;
                let n_mod_4 = n.limbs().first().copied().unwrap_or(0) & 3;
                if a_mod_4 == 3 && n_mod_4 == 3 {
                    t = t.wrapping_neg();
                }
            }

            // Now a > n. Both are odd. Try Lehmer acceleration for large numbers.
            if n.limbs().len() >= Gcd::LEHMER_THRESHOLD {
                let (u_hat, v_hat) = Gcd::extract_top_limb(a.limbs(), n.limbs());
                let u_low = a.limbs().first().copied().unwrap_or(0);
                let v_low = n.limbs().first().copied().unwrap_or(0);
                let LehmerJacobiStep {
                    u0,
                    v0,
                    u1,
                    v1,
                    even,
                    t: step_t,
                } = lehmer_jacobi_simulate(u_hat, v_hat, u_low, v_low);

                let is_identity = u0 == 1 && v0 == 0 && u1 == 0 && v1 == 1;
                if !is_identity
                    && Gcd::lehmer_update(
                        &mut a,
                        &mut n,
                        &mut u_backup,
                        &mut v_backup,
                        u0,
                        v0,
                        u1,
                        v1,
                        even,
                    )
                {
                    t = t.wrapping_mul(step_t);
                    continue;
                }
            }

            // When both odd operands have the same normalized limb length,
            // subtracting once is an exact quotient-one Euclidean step. The
            // result is even, so the following trailing-zero removal absorbs
            // the next binary reduction without a full multi-limb division.
            // If the lengths differ, the quotient may be large; use exact
            // division there instead of repeating potentially many subtractions.
            if a.limbs().len() == n.limbs().len() {
                a.sub_assign(&n);
                if a.is_zero() {
                    return 0;
                }
            } else {
                let mut fb_r = Self::zero();
                Division::rem_into(&a, &n, &mut fb_r, &mut scratch);
                a = fb_r;
            }

            if a.is_zero() {
                return 0;
            }

            let tz = a.trailing_zeros();
            a.shr_assign(tz);

            // Apply (2/n) factor.
            if tz & 1 == 1 {
                let n_mod_8 = n.limbs().first().copied().unwrap_or(0) & 7;
                if n_mod_8 == 3 || n_mod_8 == 5 {
                    t = t.wrapping_neg();
                }
            }
        }
    }
}

struct LehmerJacobiStep {
    u0: Limb,
    v0: Limb,
    u1: Limb,
    v1: Limb,
    even: bool,
    t: i8,
}

#[allow(
    unsafe_code,
    clippy::similar_names,
    reason = "Integer division is required for Lehmer simulation; unwrap_unchecked keeps quotient estimation branchless. Similar names reflect standard Lehmer coefficient symbols."
)]
const fn lehmer_jacobi_simulate(
    mut u_hat: Limb,
    mut v_hat: Limb,
    mut u_low: Limb,
    mut v_low: Limb,
) -> LehmerJacobiStep {
    let mut u_0: Limb = 1;
    let mut v_0: Limb = 0;
    let mut u_1: Limb = 0;
    let mut v_1: Limb = 1;
    let mut even = true;
    let mut t: i8 = 1;

    loop {
        if v_hat == 0 {
            break;
        }

        let mut q: Limb = 0;
        let mut rem = u_hat;
        if rem >= v_hat {
            rem = rem.wrapping_sub(v_hat);
            q = q.wrapping_add(1);
            if rem >= v_hat {
                rem = rem.wrapping_sub(v_hat);
                q = q.wrapping_add(1);
                if rem >= v_hat {
                    rem = rem.wrapping_sub(v_hat);
                    q = q.wrapping_add(1);
                    if rem >= v_hat {
                        // SAFETY: v_hat is non-zero
                        let div = unsafe { rem.checked_div(v_hat).unwrap_unchecked() };
                        q = q.wrapping_add(div);
                        rem = rem.wrapping_sub(div.wrapping_mul(v_hat));
                    }
                }
            }
        }

        let update_u = u_0.wrapping_add(q.wrapping_mul(u_1));
        let update_v = v_0.wrapping_add(q.wrapping_mul(v_1));

        if even {
            if v_hat < v_1 || rem < update_u {
                break;
            }
        } else if v_hat < u_1 || rem < update_v {
            break;
        }

        // If quotient is odd, the new remainder will be even!
        // We stop the Lehmer round before creating an even remainder.
        if q & 1 == 1 {
            break;
        }

        // q is even, so the remainder is odd.
        let rem_low = u_low.wrapping_sub(q.wrapping_mul(v_low));

        // Both v_low (old denominator) and rem_low (new numerator) are odd.
        // Apply quadratic reciprocity when swapping them.
        if (rem_low & 3 == 3) && (v_low & 3 == 3) {
            t = t.wrapping_neg();
        }

        u_hat = v_hat;
        v_hat = rem;
        u_low = v_low;
        v_low = rem_low;

        u_0 = update_u;
        v_0 = update_v;

        let temp_u = u_0;
        u_0 = u_1;
        u_1 = temp_u;

        let temp_v = v_0;
        v_0 = v_1;
        v_1 = temp_v;

        even = !even;
    }

    LehmerJacobiStep {
        u0: u_0,
        v0: v_0,
        u1: u_1,
        v1: v_1,
        even,
        t,
    }
}
