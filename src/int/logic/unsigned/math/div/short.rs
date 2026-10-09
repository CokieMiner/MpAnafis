//! Exact quotient-only basecase with progressively shorter remainder updates.
//!
//! Let B be the limb base, D a normalized n-limb divisor, and j the quotient
//! position. Discard k = max(n - 2 - j, 0) low divisor limbs at that step.
//! Every omitted contribution is `q_j * (D mod B^k) * B^j < B^(n-1)`.
//! For `t = min(n-2, quotient_digits)`, their sum E therefore satisfies
//! 0 <= E < t*B^(n-1). A valid limb slice occupies at most `isize::MAX` bytes,
//! hence n < B/2 on every supported pointer width. Normalization gives
//! D >= B^n/2, so E < D independently of an empirical crossover.
//!
//! The shortened updates produce an upper quotient estimate Q and retain
//! R = U - Q*D + E >= 0. Dropping nonnegative low contributions cannot
//! underestimate the first quotient digit differing from exact division.
//! Thus Q is the exact quotient or one larger. The high remainder limb
//! certifies Q whenever R >= t*B^(n-1). Otherwise,
//! only the omitted triangular products are subtracted from the same buffer;
//! their borrow proves Q is one too large. Folding that carry into the
//! retained pair yields U-Q*D; a negative residue plus D is the remainder.
//! A leading-window overflow repairs the products omitted by the already
//! written digits, then finishes only the remaining full-width windows.
//!
//! For known divisibility U = q*D, R = (q-Q)*D + E >= 0 and E < D
//! exclude Q >= q+1. The upper estimate is therefore already q, and
//! quotient-only output needs no omitted triangular products.
//!
//! Two retained limbs bound the error below one divisor; one retained limb
//! would allow a complete divisor's error per step. This is the basecase
//! short-division approach discussed by Harvey and Zimmermann,
//! "Short Division of Long Integers" (2011).

#![expect(
    unsafe_code,
    reason = "normalized guard windows and the triangular error bound prove slice indices, carry sums, and kernel preconditions"
)]

use core::num::NonZeroUsize;

use super::{Addition, Division, DoubleLimb, LIMB_BITS, Limb, PreparedDivisor};

impl PreparedDivisor {
    /// Writes an exact quotient from normalized operands. With
    /// `WRITE_REMAINDER`, the triangular correction also finishes the
    /// n-limb remainder in the low numerator limbs so Algorithm D can
    /// denormalize it. Every division completes in the supplied workspace.
    ///
    /// The divisor has at least three limbs, its top bit set and the prepared
    /// leading pair. The dividend
    /// has a high guard below the divisor's leading limb. `quotient` contains
    /// exactly `numerator.len() - divisor.len()` initialized slots and is nonempty.
    /// With `TRUNCATED_REMAINDER`, it has exactly `divisor.len()-1` slots;
    /// A `true` result proves that the unnormalized remainder exceeds the
    /// quotient, allowing operand truncation to finish without writing it.
    /// `KNOWN_EXACT` requires U to be a multiple of D and quotient-only
    /// output. It skips triangular repair when all shortened windows fit.
    #[expect(
        clippy::as_conversions,
        reason = "DoubleLimb has twice LIMB_BITS and embeds every native Limb exactly on all supported pointer widths"
    )]
    #[cfg_attr(
        not(target_pointer_width = "16"),
        expect(
            clippy::cast_possible_truncation,
            reason = "DoubleLimb narrows to Limb on 32-bit and 64-bit targets"
        )
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "the full windows, shortened windows and triangular correction share one initialized numerator and quotient workspace"
    )]
    pub fn divide_quotient<
        const WRITE_REMAINDER: bool,
        const TRUNCATED_REMAINDER: bool,
        const KNOWN_EXACT: bool,
    >(
        &self,
        numerator_storage: &mut [Limb],
        divisor: &[Limb],
        quotient: &mut [Limb],
    ) -> bool {
        // SAFETY: normalized Algorithm D and guarded Burnikel prefixes
        // retain the three leading divisor digits. The triangular floor
        // contains the first of those digits and every digit below the triple.
        let (below_three, _) = unsafe { divisor.split_last_chunk::<3>().unwrap_unchecked() };
        // SAFETY: these sums reconstruct existing materialized divisor spans.
        let (n, floor, upper_index) = unsafe {
            (
                below_three.len().unchecked_add(3),
                below_three.len().unchecked_add(1),
                below_three.len().unchecked_add(2),
            )
        };
        let digits = quotient.len();
        debug_assert!(
            n >= 3 && numerator_storage.len() > n,
            "short division retains two divisor limbs and a numerator guard"
        );
        debug_assert!(
            !KNOWN_EXACT || (!WRITE_REMAINDER && !TRUNCATED_REMAINDER),
            "known divisibility supplies only the final quotient"
        );
        debug_assert_eq!(
            Some(digits),
            numerator_storage.len().checked_sub(n),
            "the quotient has one slot per descending division window"
        );
        debug_assert!(
            !TRUNCATED_REMAINDER || (WRITE_REMAINDER && digits.checked_add(1) == Some(n)),
            "truncated-prefix certification retains one divisor guard limb"
        );
        debug_assert_eq!(
            (
                divisor.last().copied(),
                divisor.iter().rev().nth(1).copied()
            ),
            (Some(self.high), Some(self.low)),
            "the prepared leading pair matches"
        );
        // SAFETY: every admitted window supplies at least one quotient digit.
        let quotient_width = unsafe { NonZeroUsize::new_unchecked(digits) };
        // SAFETY: the supplied n+digits initialized numerator span fits its
        // owner; defining that view preserves the quotient/window equality.
        let numerator =
            unsafe { numerator_storage.get_unchecked_mut(..n.unchecked_add(quotient_width.get())) };
        let (high, low, inverse, sub_mul) = (self.high, self.low, self.inverse, self.sub_mul);
        // SAFETY: numerator.len() = n+digits with n>=3 and digits>=1 bounds
        // its initialized high pair. Normalization supplied its high guard.
        let (mut upper, following) = unsafe {
            (
                *numerator.last().unwrap_unchecked(),
                *numerator.get_unchecked(numerator.len().unchecked_sub(2)),
            )
        };
        let active_digits = if digits > 1 && upper == 0 && following < high {
            // SAFETY: digits>1 bounds the initialized output's highest slot.
            // The high pair proves this window is below D, hence its digit
            // is zero and the next window uses following as its high guard.
            unsafe {
                let active = digits.unchecked_sub(1);
                *quotient.get_unchecked_mut(active) = 0;
                upper = following;
                active
            }
        } else {
            digits
        };
        // A zero high digit contributes no omitted product. Keep its output
        // slot for truncated certification, but omit both its division step
        // and its triangular repair from the active arithmetic.
        let terms = floor.min(active_digits);

        // Above the triangle, every low divisor limb contributes to a later
        // quotient digit and the full Algorithm D step is required. The high
        // remainder limb is retained across both descending loops.
        for j in (terms..active_digits).rev() {
            // SAFETY: j < digits and numerator.len() >= n + digits. The full
            // n-limb window is initialized and disjoint from the divisor;
            // upper supplies its normalized high guard.
            unsafe {
                let end = j.unchecked_add(n);
                let window = numerator.get_unchecked_mut(j..end);
                let digit;
                (digit, upper) =
                    Division::knuth_d_step(window, divisor, upper, high, low, inverse, sub_mul);
                *quotient.get_unchecked_mut(j) = digit;
            }
        }

        // Every shortened window starts at floor; each successive step drops
        // one more low divisor limb and consumes one high numerator limb.
        for j in (1..terms).rev() {
            // SAFETY: j < terms <= floor, so 1 <= drop <= floor. The active
            // divisor retains at least three limbs. Its initialized numerator
            // window ends within n + digits; upper retains the high guard.
            unsafe {
                let drop = floor.unchecked_sub(j);
                let guard = j.unchecked_add(n);
                let previous = guard.unchecked_sub(1);
                if (upper, *numerator.get_unchecked(previous)) >= (high, low) {
                    // No lower digit has been written. Repair the omitted
                    // high-digit products and finish this exact tail once.
                    // Its memory recurrence needs the retained high guard.
                    *numerator.get_unchecked_mut(guard) = upper;
                    self.complete_quotient(numerator, divisor, quotient, j.unchecked_add(1), terms);
                    return false;
                }
                let window = numerator.get_unchecked_mut(floor..guard);
                let head = divisor.get_unchecked(drop..);
                let digit;
                (digit, upper) =
                    Division::knuth_d_step(window, head, upper, high, low, inverse, sub_mul);
                *quotient.get_unchecked_mut(j) = digit;
            }
        }

        // The last digit has exactly the reciprocal's two-limb divisor,
        // so its remainder is final without another multiply-subtract.
        // SAFETY: n >= 3 and numerator has a high guard at n. The lower pair
        // is initialized and upper retains that guard. The comparison enforces the reciprocal's
        // strict quotient-fit precondition, including truncation overflow.
        // TRUNCATED_REMAINDER also proves quotient.len() = n-1, so its
        // initialized leading slot floor = n-2 exists for certification.
        unsafe {
            let guard = upper;
            let previous = *numerator.get_unchecked(upper_index);
            if (guard, previous) >= (high, low) {
                *numerator.get_unchecked_mut(n) = guard;
                self.complete_quotient(numerator, divisor, quotient, 1, terms);
                return false;
            }
            let (digit, remainder_high, remainder_low) = Division::udiv_qr_3by2(
                guard,
                previous,
                *numerator.get_unchecked(floor),
                high,
                low,
                inverse,
            );
            *quotient.get_unchecked_mut(0) = digit;
            if KNOWN_EXACT {
                // U=q*D and Q>=q give R=(q-Q)*D+E with 0<=E<D.
                // Every accepted window leaves R>=0, excluding Q>q.
                // No caller consumes the residual window in this mode.
                return false;
            }
            *numerator.get_unchecked_mut(floor) = remainder_low;
            *numerator.get_unchecked_mut(upper_index) = remainder_high;
            // E < terms*B^(n-1). With h=Q[n-2], B*Q < (h+1)*B^(n-1).
            // Hence remainder_high > terms+h proves U-Q*D > B*Q. Undoing any
            // normalization shift s < LIMB_BITS leaves remainder > Q.
            // A saturated threshold rejects an unrepresentable certificate.
            if TRUNCATED_REMAINDER
                && remainder_high > terms.saturating_add(*quotient.get_unchecked(floor))
            {
                return true;
            }
            // Ordinary quotient-only output needs only U-Q*D >= 0.
            if !WRITE_REMAINDER && remainder_high >= terms {
                return false;
            }
        }

        // Recover only the omitted triangle, highest quotient digit first.
        // Each term ends at floor. Accumulate its carry into a double limb;
        // the two retained numerator limbs are never rewritten by this loop.
        // Earlier high-quotient updates are already exact.
        let mut correction_sum: DoubleLimb = 0;
        for j in (0..terms).rev() {
            // SAFETY: 0 <= j < terms <= floor and drop = floor-j > 0.
            // The disjoint source/destination prefixes contain drop limbs;
            // quotient[j] was initialized in the preceding descending loop.
            // A scalar product carry is below the scalar (or zero for zero),
            // and the returned borrow is binary, so their sum fits Limb.
            let correction = unsafe {
                let drop = floor.unchecked_sub(j);
                let (carry, borrow) = sub_mul(
                    numerator.as_mut_ptr().add(j),
                    divisor.as_ptr(),
                    drop,
                    *quotient.get_unchecked(j),
                );
                carry.unchecked_add(borrow)
            };
            // SAFETY: each correction fits Limb, and terms <= n-2 < B/2
            // by the materialized-slice bound. Their sum is < B^2/2 and fits
            // DoubleLimb, whose width is exactly twice LIMB_BITS.
            unsafe {
                correction_sum = correction_sum.unchecked_add(correction as DoubleLimb);
            }
        }
        // SAFETY: floor = n-2 and numerator has at least n+1 initialized
        // limbs. Every correction kernel ends strictly below these slots.
        let retained = unsafe {
            ((*numerator.get_unchecked(upper_index) as DoubleLimb) << LIMB_BITS)
                | (*numerator.get_unchecked(floor) as DoubleLimb)
        };
        // The triangle wrote U-Q*D below the retained pair. Its collected
        // carry exceeds that pair exactly when Q is one high. Quotient-only
        // output needs just that comparison; remainder output also folds the
        // carry into the pair and restores D after underflow.
        let (folded, underflow) = retained.overflowing_sub(correction_sum);
        if WRITE_REMAINDER {
            // SAFETY: floor = n-2 and the numerator has at least n initialized
            // limbs. These two slots hold the retained pair just read above.
            unsafe {
                *numerator.get_unchecked_mut(floor) = folded as Limb;
                *numerator.get_unchecked_mut(upper_index) = (folded >> LIMB_BITS) as Limb;
            }
        }
        if underflow {
            let borrow = Addition::propagate_borrow(quotient, 1);
            debug_assert_eq!(borrow, 0, "an overestimated quotient is positive");
            if WRITE_REMAINDER {
                // SAFETY: normalization supplied at least n+1 initialized
                // numerator limbs; triangular repair preserves that length.
                let region = unsafe { numerator.get_unchecked_mut(..n) };
                let carry = Addition::add_slice_in_place(region, divisor);
                debug_assert_eq!(carry, 1, "a negative residue plus D is rem+B^n");
            }
        }
        false
    }

    /// Repairs only the omitted products of the completed high digits, then
    /// divides the remaining exact windows using the same reciprocal and kernel.
    ///
    /// `remaining > 0` low digits are unwritten. Every omitted contribution
    /// ends below limb `n-2`, and their sum E is below D. The completed prefix
    /// cannot exceed the exact prefix: an excess would subtract at least
    /// `D*B^remaining`, which E < D cannot make nonnegative. The repaired
    /// dividend therefore lies in `[0, D*B^remaining)`; subtraction cannot
    /// underflow and each subsequent full-width digit fits a limb.
    fn complete_quotient(
        &self,
        numerator: &mut [Limb],
        divisor: &[Limb],
        quotient: &mut [Limb],
        remaining: usize,
        terms: usize,
    ) {
        let n = divisor.len();
        // SAFETY: n >= 3, remaining <= terms <= n-2, and the original
        // numerator has n+quotient.len() limbs with quotient.len() >= terms.
        let (floor, active_len) = unsafe { (n.unchecked_sub(2), n.unchecked_add(remaining)) };
        for j in (remaining..terms).rev() {
            // SAFETY: remaining <= j < terms <= n-2. The initialized
            // disjoint product spans end at floor, within both operands.
            // Product carry plus its binary borrow fits a native limb.
            let correction = unsafe {
                let width = floor.unchecked_sub(j);
                let (carry, borrow) = (self.sub_mul)(
                    numerator.as_mut_ptr().add(j),
                    divisor.as_ptr(),
                    width,
                    *quotient.get_unchecked(j),
                );
                carry.unchecked_add(borrow)
            };
            // SAFETY: floor < active_len <= numerator.len(). The nonempty
            // tail was initialized by normalization and preceding steps.
            let (first, tail) = unsafe {
                numerator
                    .get_unchecked_mut(floor..active_len)
                    .split_first_mut()
                    .unwrap_unchecked()
            };
            let (difference, underflow) = first.overflowing_sub(correction);
            *first = difference;
            let borrow = Addition::propagate_borrow(tail, Limb::from(underflow));
            debug_assert_eq!(
                borrow, 0,
                "the completed quotient prefix cannot overestimate"
            );
        }
        // SAFETY: remaining >= 1 and active_len <= numerator.len(). The
        // repaired dividend initializes its high guard at active_len-1.
        let mut upper = unsafe { *numerator.get_unchecked(active_len.unchecked_sub(1)) };
        for j in (0..remaining).rev() {
            // SAFETY: j < remaining bounds the initialized n-limb window
            // by active_len. Its retained high guard is below D; every step
            // preserves this bound for the following quotient digit.
            unsafe {
                let end = j.unchecked_add(n);
                let window = numerator.get_unchecked_mut(j..end);
                let digit;
                (digit, upper) = Division::knuth_d_step(
                    window,
                    divisor,
                    upper,
                    self.high,
                    self.low,
                    self.inverse,
                    self.sub_mul,
                );
                *quotient.get_unchecked_mut(j) = digit;
            }
        }
        // SAFETY: n >= 3 and numerator.len() >= active_len > n. All lower
        // remainder limbs were written above; this publishes the high limb.
        unsafe {
            *numerator.get_unchecked_mut(n.unchecked_sub(1)) = upper;
        }
    }
}
