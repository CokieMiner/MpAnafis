//! Leading-limb extraction and quotient simulation for Lehmer reduction.

#![expect(
    unsafe_code,
    reason = "ordered leading windows establish operand indices, positive divisors, and nonoverflowing Euclidean coefficient updates"
)]

use core::{
    cmp::Ordering,
    mem::swap,
    num::NonZeroUsize,
    ops::{Div, Rem},
};

use super::{ArchKernels, DoubleLimb, Gcd, LIMB_BITS, Limb, WIDE_LEHMER_THRESHOLD};

impl Gcd {
    /// Dispatches Lehmer quotient simulation, selecting double-limb simulation
    /// when operands are wide and equal in length, and single-limb simulation otherwise.
    ///
    /// `on_quotient` receives the accepted prefix in execution order. Its state
    /// is speculative until the caller accepts the full-width matrix update;
    /// an empty closure compiles out of ordinary and extended GCD.
    #[inline]
    pub fn simulate_step<const SMALL_QUOTIENTS: bool>(
        u_limbs: &[Limb],
        v_limbs: &[Limb],
        force_wide: Option<bool>,
        branchless: bool,
        on_quotient: impl FnMut(Limb),
    ) -> (Limb, Limb, Limb, Limb, bool) {
        let u_len = u_limbs.len();
        let v_len = v_limbs.len();
        let use_wide = force_wide.unwrap_or(v_len >= WIDE_LEHMER_THRESHOLD);
        if use_wide && u_len >= 3 && u_len == v_len {
            let (u_hat, v_hat) = Self::extract_top_two_limbs(u_limbs, v_limbs);
            lehmer_simulate_wide::<SMALL_QUOTIENTS>(u_hat, v_hat, on_quotient)
        } else {
            let (u_hat, v_hat) = Self::extract_top_limb(u_limbs, v_limbs);
            if branchless {
                Self::lehmer_simulate::<true>(u_hat, v_hat, on_quotient)
            } else {
                Self::lehmer_simulate::<false>(u_hat, v_hat, on_quotient)
            }
        }
    }

    /// Extracts the two leading normalized limbs used by wide Lehmer simulation.
    ///
    /// Both values are aligned with the leading limb of `u`, which is at least
    /// `v` and has two or more limbs; `v` has the same width or one limb fewer.
    /// Absent limbs read as zero, so a two-limb pair yields its exact values.
    /// Keeping the second limb makes the quotient sequence stable for more
    /// Euclidean steps while the transition coefficients remain single-limb
    /// values, the representation consumed by the Lehmer matrix update.
    pub fn extract_top_two_limbs(u_limbs: &[Limb], v_limbs: &[Limb]) -> (DoubleLimb, DoubleLimb) {
        let len = u_limbs.len();
        debug_assert!(
            len >= 2 && len.wrapping_sub(v_limbs.len()) <= 1,
            "wide Lehmer extraction requires ordered widths differing by at most one limb"
        );
        let limb = |limbs: &[Limb], from_top: usize| {
            len.checked_sub(from_top)
                .and_then(|index| limbs.get(index))
                .copied()
                .unwrap_or(0)
        };
        let top_u = limb(u_limbs, 1);
        let shift = top_u.leading_zeros();
        let u_hat = normalize_top_two(top_u, limb(u_limbs, 2), limb(u_limbs, 3), shift);
        let v_hat = normalize_top_two(limb(v_limbs, 1), limb(v_limbs, 2), limb(v_limbs, 3), shift);
        (u_hat, v_hat)
    }

    /// Extracts the leading normalized limb used by Lehmer simulation.
    ///
    /// Both inputs must contain at least two limbs and `u_limbs` must not be
    /// shorter than `v_limbs`.
    pub fn extract_top_limb(u_limbs: &[Limb], v_limbs: &[Limb]) -> (Limb, Limb) {
        let u_len = u_limbs.len();
        let v_len = v_limbs.len();
        debug_assert!(v_len >= 2, "Lehmer extraction requires two divisor limbs");
        debug_assert!(u_len >= v_len, "Lehmer operands must be ordered");

        // SAFETY: the preconditions above prove both indices are in `u_limbs`.
        let (top_u, next_u) = unsafe {
            (
                *u_limbs.get_unchecked(u_len.wrapping_sub(1)),
                *u_limbs.get_unchecked(u_len.wrapping_sub(2)),
            )
        };
        let lz = top_u.leading_zeros();

        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS fits in u32 even on 16-bit targets (where LIMB_BITS is 16): avoids range checks and compiles to branchless register truncation."
        )]
        let limb_bits_u32 = LIMB_BITS as u32;

        let u_hat = if lz == 0 {
            top_u
        } else {
            top_u.wrapping_shl(lz) | next_u.wrapping_shr(limb_bits_u32.wrapping_sub(lz))
        };

        // The ordering precondition proves this subtraction cannot wrap.
        let limb_diff = u_len.wrapping_sub(v_len);
        let v_hat = match limb_diff.cmp(&1) {
            Ordering::Greater => 0,
            Ordering::Equal => {
                // SAFETY: `v_len >= 2` proves the top index exists.
                let top_v = unsafe { *v_limbs.get_unchecked(v_len.wrapping_sub(1)) };
                if lz == 0 {
                    0
                } else {
                    top_v.wrapping_shr(limb_bits_u32.wrapping_sub(lz))
                }
            }
            Ordering::Less => {
                // SAFETY: `v_len >= 2` proves both indices exist.
                let (top_v, next_v) = unsafe {
                    (
                        *v_limbs.get_unchecked(v_len.wrapping_sub(1)),
                        *v_limbs.get_unchecked(v_len.wrapping_sub(2)),
                    )
                };
                if lz == 0 {
                    top_v
                } else {
                    top_v.wrapping_shl(lz) | next_v.wrapping_shr(limb_bits_u32.wrapping_sub(lz))
                }
            }
        };

        (u_hat, v_hat)
    }
}

/// Simulates Lehmer steps from two leading limbs.
///
/// A quotient of one uses a subtraction in both simulation phases.
/// `SMALL_QUOTIENTS` extends the wide phase to three exact subtractions;
/// HGCD leaves enable it to avoid division for quotients two and three.
///
/// The matrix coefficients are capped at `Limb::MAX`; once a coefficient would
/// need a wider representation, the valid prefix accumulated so far is
/// returned and the caller can resume with the full operands.
#[expect(
    clippy::as_conversions,
    reason = "the double- and single-limb phases retain one transition matrix and quotient callback; limb-bounded multiply-adds fit DoubleLimb and narrowing follows proved bounds"
)]
#[cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "DoubleLimb (u64) is narrowed to Limb (u32) on 32-bit targets after checking <= Limb::MAX."
    )
)]
pub fn lehmer_simulate_wide<const SMALL_QUOTIENTS: bool>(
    mut u_hat: DoubleLimb,
    mut v_hat: DoubleLimb,
    mut on_quotient: impl FnMut(Limb),
) -> (Limb, Limb, Limb, Limb, bool) {
    let (mut u_0, mut v_0, mut u_1, mut v_1) = (1_usize, 0_usize, 0_usize, 1_usize);
    let mut even = true;
    let half_bits = LIMB_BITS >> 1;
    // B = T^2 with T = 2^(LIMB_BITS/2). Once both windows are below
    // B*T, discarding half a limb leaves single-limb windows. The combined
    // exponent 3*LIMB_BITS/2 is below DoubleLimb's 2*LIMB_BITS on all
    // supported widths (16, 32, 64); neither shift discards a set bit.
    let narrow_limit: DoubleLimb = (1 << LIMB_BITS) << half_bits;

    loop {
        if v_hat == 0 {
            return (u_0, v_0, u_1, v_1, even);
        }
        if (u_hat | v_hat) < narrow_limit {
            break;
        }

        let mut q: DoubleLimb = 0;
        let mut rem = u_hat;
        if rem >= v_hat {
            rem = rem.wrapping_sub(v_hat);
            q = 1;
            if rem >= v_hat {
                if SMALL_QUOTIENTS {
                    rem = rem.wrapping_sub(v_hat);
                    q = 2;
                    if rem >= v_hat {
                        rem = rem.wrapping_sub(v_hat);
                        q = 3;
                    }
                }
                if rem >= v_hat {
                    let (residual_q, residual_rem) = Gcd::div_rem_wide(rem, v_hat);
                    q = q.wrapping_add(residual_q);
                    rem = residual_rem;
                }
            }
        }

        if q > Limb::MAX as DoubleLimb {
            return (u_0, v_0, u_1, v_1, even);
        }

        // Each coefficient and q is below B. The complete multiply-add is
        // at most B*(B-1), so DoubleLimb cannot overflow; only the final
        // single-limb coefficient bound can reject this step.
        // SAFETY: q <= B-1 is checked above and all coefficients are
        // Limb values. Each product is <= (B-1)^2 and each complete sum
        // is <= B*(B-1) < B^2, on 16-, 32-, and 64-bit targets alike.
        let (candidate_first, candidate_second) = unsafe {
            (
                q.unchecked_mul(u_1 as DoubleLimb)
                    .unchecked_add(u_0 as DoubleLimb),
                q.unchecked_mul(v_1 as DoubleLimb)
                    .unchecked_add(v_0 as DoubleLimb),
            )
        };
        if candidate_first > Limb::MAX as DoubleLimb || candidate_second > Limb::MAX as DoubleLimb {
            return (u_0, v_0, u_1, v_1, even);
        }

        let update_u = candidate_first as Limb;
        let update_v = candidate_second as Limb;
        // For R = c*U-d*V the adverse truncation coefficient is d.
        // The divisor inherits this bound from the preceding accepted
        // remainder (and starts with adverse coefficient zero), so only
        // the newly computed remainder needs an acceptance comparison.
        let adverse = if even { update_v } else { update_u };
        if rem < adverse as DoubleLimb {
            return (u_0, v_0, u_1, v_1, even);
        }

        on_quotient(q as Limb);
        (u_hat, v_hat) = (v_hat, rem);
        (u_0, u_1) = (u_1, update_u);
        (v_0, v_1) = (v_1, update_v);
        even = !even;
    }

    // Let P be the local transition after truncation. Keeping both new
    // remainders >= 2T bounds every entry of P by (B-1)/(2T) < T/2.
    // The lost half-limb contributes less than T/2 to each scaled error.
    // Before original-input truncation error, the remainders therefore
    // exceed 2B-B/2. Their nonnegative inverse matrix reconstructs the
    // initial windows, both below B^2, bounding every matrix entry below B.
    // Original-input truncation then contributes less than B/T = T. Thus
    // a simulated remainder >= 2T is strictly positive for the full input.
    // This fixed guard replaces per-step adverse-coefficient comparisons.
    // LIMB_BITS/2 + 1 < LIMB_BITS for every supported width.
    let minimum: Limb = 2 << half_bits;
    let mut narrow_u = (u_hat >> half_bits) as Limb;
    let mut narrow_v = (v_hat >> half_bits) as Limb;
    if narrow_v < minimum {
        return (u_0, v_0, u_1, v_1, even);
    }
    loop {
        let mut q: Limb = 0;
        let mut rem = narrow_u;
        if rem >= narrow_v {
            rem = rem.wrapping_sub(narrow_v);
            q = 1;
            if rem >= narrow_v {
                // The retained divisor exceeds half a limb, so a
                // narrower hardware divide cannot apply.
                // SAFETY: entry and every accepted remainder preserve
                // narrow_v >= 2^(LIMB_BITS/2+1) > 0. The numerator's high
                // limb is zero, so the residual quotient fits Limb.
                let (residual_q, residual_rem) =
                    unsafe { ArchKernels::divrem_1_unchecked(rem, 0, narrow_v) };
                q = q.wrapping_add(residual_q);
                rem = residual_rem;
            }
        }
        if rem < minimum {
            break;
        }
        // SAFETY: narrow_v and rem >= 2T bound the exact wide-window
        // remainders represented by this candidate by > 3B/2. The nonnegative
        // inverse transition reconstructs initial windows < B^2, so every
        // candidate coefficient is < B. Its product and sum are both
        // nonnegative and no larger than that coefficient, on all widths.
        let (update_u, update_v) = unsafe {
            (
                q.unchecked_mul(u_1).unchecked_add(u_0),
                q.unchecked_mul(v_1).unchecked_add(v_0),
            )
        };
        on_quotient(q);
        (narrow_u, narrow_v) = (narrow_v, rem);
        (u_0, u_1) = (u_1, update_u);
        (v_0, v_1) = (v_1, update_v);
        even = !even;
    }
    (u_0, v_0, u_1, v_1, even)
}

impl Gcd {
    pub fn lehmer_simulate<const BRANCHLESS: bool>(
        mut u_hat: Limb,
        v_hat: Limb,
        mut on_quotient: impl FnMut(Limb),
    ) -> (Limb, Limb, Limb, Limb, bool) {
        debug_assert!(u_hat >= v_hat, "leading windows preserve operand order");
        let mut u_0: Limb = 1;
        let mut v_0: Limb = 0;
        let mut u_1: Limb = 0;
        let mut v_1: Limb = 1;
        let mut even = true;
        let Some(mut divisor) = NonZeroUsize::new(v_hat) else {
            return (u_0, v_0, u_1, v_1, even);
        };

        // Ordered windows give q >= 1. Each accepted remainder is below
        // v_hat and at least its positive adverse coefficient, so subsequent
        // iterations retain both ordering and a nonzero divisor.
        loop {
            let denominator = divisor.get();
            let mut rem = u_hat.wrapping_sub(denominator);
            let mut q: Limb = 1;
            if BRANCHLESS {
                let second = Limb::from(rem >= denominator);
                rem = rem.wrapping_sub(denominator & 0_usize.wrapping_sub(second));
                let third = Limb::from(rem >= denominator);
                rem = rem.wrapping_sub(denominator & 0_usize.wrapping_sub(third));
                q = 1_usize.wrapping_add(second).wrapping_add(third);
            } else if rem >= denominator {
                rem = rem.wrapping_sub(denominator);
                q = 2;
                if rem >= denominator {
                    rem = rem.wrapping_sub(denominator);
                    q = 3;
                }
            }
            if rem >= denominator {
                let residual_q = Div::div(rem, divisor);
                let residual_rem = Rem::rem(rem, divisor);
                // SAFETY: the total quotient is floor(u_hat/denominator)<=
                // u_hat<=Limb::MAX. The partial quotients sum to that value.
                q = unsafe { q.unchecked_add(residual_q) };
                rem = residual_rem;
            }

            // SAFETY: this is exact Euclid on the original one-limb windows.
            // Every coefficient, including the zero-remainder row, is bounded
            // by an original window divided by their GCD, hence <= Limb::MAX.
            // Each nonnegative product is no greater than its complete sum.
            let (update_u, update_v) = unsafe {
                (
                    q.unchecked_mul(u_1).unchecked_add(u_0),
                    q.unchecked_mul(v_1).unchecked_add(v_0),
                )
            };

            // The previous accepted remainder already bounds this divisor's
            // adverse coefficient. Only the new negative term needs checking.
            let adverse = if even { update_v } else { update_u };
            if rem < adverse {
                break;
            }

            on_quotient(q);
            u_hat = denominator;
            // Every orientation's adverse coefficient is positive: it starts
            // at one and follows positive-quotient Euclidean column additions.
            // SAFETY: acceptance proves rem>=adverse>=1. Carry this invariant
            // in the divisor's type instead of an assumption at each division.
            divisor = unsafe { NonZeroUsize::new_unchecked(rem) };
            u_0 = update_u;
            v_0 = update_v;

            swap(&mut u_0, &mut u_1);
            swap(&mut v_0, &mut v_1);
            even = !even;
        }

        (u_0, v_0, u_1, v_1, even)
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "DoubleLimb is split into exact Limb-sized halves and recombined without loss"
)]
impl Gcd {
    /// Divides one double-limb value by another, returning quotient and remainder.
    ///
    /// When the true quotient exceeds one limb this is not an exact division:
    /// it returns the sentinel `(B, 0)` instead. Callers only invoke it where a
    /// multi-limb quotient is rejected outright, so the sentinel never escapes
    /// as an arithmetic result.
    pub fn div_rem_wide(numerator: DoubleLimb, divisor: DoubleLimb) -> (DoubleLimb, DoubleLimb) {
        debug_assert!(divisor != 0, "wide Lehmer divisor must be nonzero");
        let limb_shift = LIMB_BITS as u32;
        let numerator_high = numerator.wrapping_shr(limb_shift) as Limb;
        let numerator_low = numerator as Limb;
        let divisor_high = divisor.wrapping_shr(limb_shift) as Limb;
        let divisor_low = divisor as Limb;

        if divisor_high == 0 {
            if numerator_high >= divisor_low {
                return (DoubleLimb::from(1_u8) << LIMB_BITS, 0);
            }
            // SAFETY: divisor_low is nonzero because divisor is nonzero and its high
            // limb is zero; this branch proves numerator_high < divisor_low.
            let (quotient, remainder) = unsafe {
                ArchKernels::divrem_1_unchecked(numerator_low, numerator_high, divisor_low)
            };
            return (quotient as DoubleLimb, remainder as DoubleLimb);
        }

        // With N = n1*B+n0 and D = d1*B+d0, q0 = floor(n1/d1)
        // overestimates N/D. If q0 <= d1, the omitted product q0*d0
        // is strictly below d1*B <= D, so one correction is sufficient.
        // This case needs no normalization or three-limb numerator.
        // SAFETY: divisor_high is nonzero and the high dividend is zero.
        let (head_quotient, head_remainder) =
            unsafe { ArchKernels::divrem_1_unchecked(numerator_high, 0, divisor_high) };
        if head_quotient <= divisor_high {
            let prefix =
                ((head_remainder as DoubleLimb) << LIMB_BITS) | numerator_low as DoubleLimb;
            let product = (head_quotient as DoubleLimb).wrapping_mul(divisor_low as DoubleLimb);
            let correction = prefix < product;
            let remainder = prefix.wrapping_sub(product);
            return (
                head_quotient.wrapping_sub(Limb::from(correction)) as DoubleLimb,
                if correction {
                    remainder.wrapping_add(divisor)
                } else {
                    remainder
                },
            );
        }

        let shift = divisor_high.leading_zeros();
        let (n2, n1, n0, d1, d0) = if shift == 0 {
            (0, numerator_high, numerator_low, divisor_high, divisor_low)
        } else {
            let lower_shift = limb_shift.wrapping_sub(shift);
            (
                numerator_high.wrapping_shr(lower_shift),
                numerator_high.wrapping_shl(shift) | numerator_low.wrapping_shr(lower_shift),
                numerator_low.wrapping_shl(shift),
                divisor_high.wrapping_shl(shift) | divisor_low.wrapping_shr(lower_shift),
                divisor_low.wrapping_shl(shift),
            )
        };

        // `n2` contains only the bits shifted out of one numerator limb, while
        // normalization sets the top bit of `d1`; therefore `n2 < d1` on every
        // supported limb width and the quotient fits one limb.
        // SAFETY: normalization makes d1 nonzero and the bound above proves n2 < d1.
        let (mut quotient, mut quotient_remainder) =
            unsafe { ArchKernels::divrem_1_unchecked(n1, n2, d1) };
        let mut quotient_low_product = (quotient as DoubleLimb).wrapping_mul(d0 as DoubleLimb);
        let mut remainder_window =
            ((quotient_remainder as DoubleLimb) << LIMB_BITS) | n0 as DoubleLimb;
        if quotient_low_product > remainder_window {
            quotient = quotient.wrapping_sub(1);
            let (remainder, overflow) = quotient_remainder.overflowing_add(d1);
            quotient_remainder = remainder;
            quotient_low_product = quotient_low_product.wrapping_sub(d0 as DoubleLimb);
            remainder_window = ((quotient_remainder as DoubleLimb) << LIMB_BITS) | n0 as DoubleLimb;
            if !overflow && quotient_low_product > remainder_window {
                quotient = quotient.wrapping_sub(1);
                quotient_remainder = quotient_remainder.wrapping_add(d1);
                quotient_low_product = quotient_low_product.wrapping_sub(d0 as DoubleLimb);
                remainder_window =
                    ((quotient_remainder as DoubleLimb) << LIMB_BITS) | n0 as DoubleLimb;
            }
        }

        let normalized_remainder = remainder_window.wrapping_sub(quotient_low_product);
        (
            quotient as DoubleLimb,
            normalized_remainder.wrapping_shr(shift),
        )
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The normalized high and low limbs are each Limb-sized and are packed into DoubleLimb by construction."
)]
const fn normalize_top_two(top: Limb, next: Limb, third: Limb, shift: u32) -> DoubleLimb {
    if shift == 0 {
        return ((top as DoubleLimb) << LIMB_BITS) | next as DoubleLimb;
    }
    let lower_shift = (LIMB_BITS as u32).wrapping_sub(shift);
    let high = top.wrapping_shl(shift) | next.wrapping_shr(lower_shift);
    let low = next.wrapping_shl(shift) | third.wrapping_shr(lower_shift);
    ((high as DoubleLimb) << LIMB_BITS) | low as DoubleLimb
}
