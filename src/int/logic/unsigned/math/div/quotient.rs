//! Quotient-only division through operand truncation.
//!
//! The full division engine spends work proportional to the divisor length on
//! every quotient limb, because it maintains the running remainder. When only
//! the quotient is wanted and the quotient is far shorter than the divisor,
//! nearly all of that work is discarded: a 65536-bit dividend over a 65536-bit
//! divisor yields a one-limb quotient, yet the engine normalizes, updates and
//! denormalizes 1024 limbs of remainder to produce it.
//!
//! Dropping the low `k` limbs of *both* operands barely moves the quotient.
//! Write `u = u' B^k + a` and `v = v' B^k + b` with `0 <= a, b < B^k` and
//! `B = 2^LIMB_BITS`, and let `u' = q' v' + r'` be the truncated division. Then
//!
//! ```text
//! u = q' v + D,   with D = r' B^k + a - q' b
//! ```
//!
//! so `floor(u / v) = q' + floor(D / v)`. Retaining `qn + 2` divisor limbs,
//! where `qn` is the quotient limb count, forces `q' < B^qn` and
//! `v >= B^(qn + 1 + k) > q' B^k`, which bounds `D` into `(-v, v)`. The true
//! quotient is therefore `q'` or `q' - 1` — truncation never underestimates and
//! is never more than one unit high.
//!
//! Which of the two holds is the sign of `D`. `r' >= q'` implies
//! `D >= q' (B^k - b) + a >= 0` and hence `q = q'`; that test reads only the
//! small truncated operands and settles all but a `B^-2` fraction of inputs, so
//! the common case costs a division on `2 qn + 1` limbs and nothing else. The
//! remaining inputs — exact multiples above all, where `D` is identically zero —
//! are resolved by forming `q' v` and comparing against `u`: one multiplication,
//! still cheaper than the full-width division it replaces.

use core::cmp::Ordering;

use super::{ArchKernels, Division, InternalMpUint, LIMB_BITS, Limb};

/// Guard limbs retained in the truncated divisor beyond the quotient length.
///
/// Two limbs keep the truncated divisor at least one limb longer than the
/// truncated quotient, which is what bounds the truncation error to a single
/// unit in the last place.
const DIV_Q_GUARD_LIMBS: usize = 2;

/// Maximum quotient accepted by the equal-width scalar path.
///
/// The path is aimed at the common near-equal shape, where a scalar
/// multiply-subtract is linear in the divisor width and replaces Algorithm D's
/// full-width quotient loop. Larger estimates return to the general tower.
const SMALL_QUOTIENT_MAX: Limb = 8;

impl Division {
    /// Computes an equal-width division whose quotient fits a small scalar.
    ///
    /// Let `e` be the leading-limb ratio and `q` the exact quotient. Equal-width
    /// normalized operands give `q <= e`, while
    /// `num / den > e * d_top / (d_top + 1)`. Therefore
    /// `q >= floor(e * d_top / (d_top + 1)) >= floor(e / 2)`. For `e <= 8`,
    /// this proves `e - q <= ceil(e / 2) <= 4`. Estimates zero and one are
    /// resolved directly from the already-loaded leading limbs, with a full
    /// comparison needed only when those limbs are equal. Larger estimates
    /// form the exact remainder with one scalar multiply-subtract and at most
    /// four downward corrections. An estimate outside the supported range
    /// returns to the caller without changing either output.
    pub fn small_quotient_div_rem(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
    ) -> bool {
        let u_limbs = Self::significant_limbs(num_a.limbs());
        let v_limbs = Self::significant_limbs(den_b.limbs());
        let u_len = u_limbs.len();
        let v_len = v_limbs.len();
        if u_len != v_len || u_len == 0 {
            return false;
        }

        // SAFETY: u_len and v_len are equal and strictly positive.
        let numerator_top = unsafe { *u_limbs.get_unchecked(u_len.wrapping_sub(1)) };
        // SAFETY: u_len and v_len are equal and strictly positive.
        let denominator_top = unsafe { *v_limbs.get_unchecked(v_len.wrapping_sub(1)) };

        // SAFETY: denominator_top is the leading limb of a normalized non-zero integer, so it cannot be zero.
        let mut quotient = unsafe {
            numerator_top
                .checked_div(denominator_top)
                .unwrap_unchecked()
        };
        if !(2..=SMALL_QUOTIENT_MAX).contains(&quotient) {
            if quotient == 0 {
                // Equal active lengths plus `u_top < v_top` proves `num < den`;
                // lower limbs cannot change the ordering of unequal leading limbs.
                quotient_out.clear();
                rem_out.clone_from(num_a);
                return true;
            }
            if quotient == 1 {
                // `u_top / v_top == 1` implies `u_top < 2*v_top`. Consequently
                // `num < 2*den`, including when doubling `v_top` carries into a
                // new limb, and the exact quotient is zero or one. Unequal
                // leading limbs already prove `num > den`; only equality needs
                // a full comparison of the lower limbs.
                if numerator_top == denominator_top
                    && InternalMpUint::cmp_limbs(u_limbs, v_limbs) == Ordering::Less
                {
                    quotient_out.clear();
                    rem_out.clone_from(num_a);
                } else {
                    *quotient_out = InternalMpUint::one();
                    *rem_out = num_a.sub(den_b);
                }
                return true;
            }
            return false;
        }

        let mut work = InternalMpUint::with_capacity(u_limbs.len().wrapping_add(1));
        let mut corrections = 0_u8;
        loop {
            let work_len = u_limbs.len().wrapping_add(1);
            work.resize(work_len);
            let work_limbs = work.limbs_mut();
            // SAFETY: `work` was resized to `u_limbs.len() + 1`, so this prefix
            // has exactly the source length and every element is initialized.
            unsafe { work_limbs.get_unchecked_mut(..u_limbs.len()) }.copy_from_slice(u_limbs);
            // SAFETY: the resize above added the carry slot at `u_limbs.len()`.
            unsafe {
                *work_limbs.get_unchecked_mut(u_limbs.len()) = 0;
            }

            let borrow = Self::mul_sub_in_place(
                work_limbs,
                v_limbs,
                quotient,
                ArchKernels::selected_sub_mul_limbs_unchecked(),
            );
            if borrow != 0 {
                if corrections >= 4 || quotient < 2 {
                    return false;
                }
                quotient = quotient.wrapping_sub(1);
                corrections = corrections.wrapping_add(1);
                continue;
            }

            work.normalize();
            if InternalMpUint::cmp_limbs(work.limbs(), v_limbs) != Ordering::Less {
                if corrections >= 4 {
                    return false;
                }
                let correction_borrow = Self::sub_limbs_in_place(work.limbs_mut(), v_limbs);
                debug_assert_eq!(correction_borrow, 0, "remainder was below divisor");
                quotient = quotient.wrapping_add(1);
                corrections = corrections.wrapping_add(1);
                continue;
            }

            *quotient_out = InternalMpUint::from_limb(quotient);
            *rem_out = work;
            return true;
        }
    }

    /// Computes `num_a / den_b` from the leading limbs of both operands.
    ///
    /// Returns `false` when the operand shape makes truncation inapplicable or
    /// unprofitable, leaving `quotient_out` untouched; the caller must then run
    /// the full division engine. Returns `true` with the exact floor quotient
    /// written to `quotient_out` otherwise — never an approximation.
    pub fn truncated_quotient(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
    ) -> bool {
        let u_limbs = Self::significant_limbs(num_a.limbs());
        let v_limbs = Self::significant_limbs(den_b.limbs());
        let num_len = u_limbs.len();
        let den_len = v_limbs.len();
        if den_len == 0 {
            return false;
        }

        // Equal-width operands have a quotient of zero or one unless the
        // dividend is at least twice the divisor.  Resolve those two common
        // cases before building the truncated operands: the comparison reads
        // only the original limbs and performs no allocation or division.
        if num_len < den_len {
            quotient_out.clear();
            return true;
        }
        if num_len == den_len {
            match InternalMpUint::cmp_limbs(u_limbs, v_limbs) {
                Ordering::Less => {
                    quotient_out.clear();
                    return true;
                }
                Ordering::Equal => {
                    *quotient_out = InternalMpUint::one();
                    return true;
                }
                Ordering::Greater if Self::less_than_double(u_limbs, v_limbs) => {
                    *quotient_out = InternalMpUint::one();
                    return true;
                }
                Ordering::Greater => {}
            }
        }

        let quot_len = num_len.wrapping_sub(den_len).wrapping_add(1);
        let Some(den_prime_len) = quot_len.checked_add(DIV_Q_GUARD_LIMBS) else {
            return false;
        };
        // Truncation only repays its setup once it discards at least as many
        // limbs as it keeps; below that the full engine is the cheaper route.
        let Some(min_den_len) = den_prime_len.checked_mul(2) else {
            return false;
        };
        if den_len < min_den_len {
            return false;
        }
        let split = den_len.wrapping_sub(den_prime_len);
        // SAFETY: `den_len >= 2 * den_prime_len` above proves
        // `den_prime_len <= den_len`, hence `split <= den_len`. The earlier
        // short-dividend return proves `num_len >= den_len`, so `split` also
        // bounds `u_limbs`.
        let (num_head, den_head) = unsafe {
            (
                u_limbs.get_unchecked(split..),
                v_limbs.get_unchecked(split..),
            )
        };
        let mut num_prime = InternalMpUint::zero();
        num_prime.clone_from_slice(num_head);
        let mut den_prime = InternalMpUint::zero();
        den_prime.clone_from_slice(den_head);

        // `den_head` ends on the non-zero top limb of the divisor, so the
        // truncated divisor satisfies the internal division precondition.
        let (quot, rem) = num_prime.div_rem(&den_prime);

        if rem < quot {
            // Ambiguous: the true quotient is `quot` or `quot - 1`, and only the
            // full-width product separates the two.
            let product = quot.mul(den_b);
            let overshoots =
                InternalMpUint::cmp_limbs(Self::significant_limbs(product.limbs()), u_limbs)
                    == Ordering::Greater;
            *quotient_out = quot;
            if overshoots {
                quotient_out.decrement();
            }
            return true;
        }

        *quotient_out = quot;
        true
    }

    /// Returns `limbs` without its high zero limbs.
    pub fn significant_limbs(limbs: &[Limb]) -> &[Limb] {
        let mut view = limbs;
        while view.last() == Some(&0) {
            let shorter_len = view.len().wrapping_sub(1);
            // SAFETY: shorter_len is strictly less than view.len() due to non-empty check.
            view = unsafe { view.get_unchecked(..shorter_len) };
        }
        view
    }

    /// Returns whether two equal-width normalized values satisfy `num < 2 * den`.
    ///
    /// If the high divisor limb has its top bit set, doubling creates an extra
    /// limb and the result is immediately true because `num` still fits in the
    /// original width. Otherwise each doubled limb is
    /// `den[i] << 1 | (den[i - 1] >> (LIMB_BITS - 1))`; the carry into a limb
    /// comes only from its immediate lower limb, so the comparison can proceed
    /// from the most significant limb without materializing `2 * den`.
    pub fn less_than_double(num: &[Limb], den: &[Limb]) -> bool {
        debug_assert_eq!(
            num.len(),
            den.len(),
            "equal-width doubling comparison requires matching lengths"
        );
        let den_len = den.len();
        if den_len == 0 || num.len() != den_len {
            return false;
        }

        let top = den_len.wrapping_sub(1);
        // SAFETY: den_len > 0, so top is in bounds.
        let den_top = unsafe { *den.get_unchecked(top) };
        if den_top >= (Limb::MAX >> 1).wrapping_add(1) {
            return true;
        }

        let mut index = den_len;
        while index > 0 {
            let limb_index = index.wrapping_sub(1);
            let carry = if limb_index == 0 {
                0
            } else {
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "LIMB_BITS is at most 64, which fits in u32 without truncation"
                )]
                let shift = (LIMB_BITS - 1) as u32;
                // SAFETY: limb_index > 0, so limb_index - 1 is in bounds.
                unsafe { *den.get_unchecked(limb_index.wrapping_sub(1)) }.wrapping_shr(shift)
            };
            // SAFETY: limb_index < den_len, so limb_index is in bounds for den and num.
            let (den_limb, num_limb) = unsafe {
                (
                    *den.get_unchecked(limb_index),
                    *num.get_unchecked(limb_index),
                )
            };
            let doubled = den_limb.wrapping_shl(1) | carry;
            match num_limb.cmp(&doubled) {
                Ordering::Less => return true,
                Ordering::Greater => return false,
                Ordering::Equal => index = limb_index,
            }
        }
        false
    }
}
