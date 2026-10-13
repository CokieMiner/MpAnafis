//! Three-limb division with normalization during input reads.
//!
//! Each digit updates `U = R*B + a`, where `0 <= R < D` and `0 <= a < B`.
//! The leading 3-by-2 division removes the two high divisor products; only
//! the product with the low divisor limb remains. The remainder stays in
//! three scalar limbs, so no normalized dividend workspace is required.
//!
//! For the final quotient-only digit, omitting that low product leaves
//! `R' = U - q*D + q*d0`. Since `q*d0 < B^2`, a nonzero high limb of R'
//! proves the estimate exact. Known divisibility proves the same result.
//! Otherwise one low product decides whether the digit needs a decrement.
//!
//! Reference: D. E. Knuth, *The Art of Computer Programming*, Volume 2,
//! 3rd ed., Section 4.3.1, Algorithm D (1997).

#![expect(
    unsafe_code,
    reason = "the fixed divisor width, nonzero high limb and admitted input length bound shifts, output writes and nonoverflowing corrections"
)]

use core::{num::NonZeroUsize, ptr::null_mut};

use super::{ArchKernels, CompleteDivisionMode, Division, InternalMpUint, Limb, LimbWriteGuard};

impl Division {
    /// Divides an initialized numerator of at least three limbs by a
    /// three-limb divisor whose high limb is nonzero. Zero input padding is
    /// permitted. Only requested outputs are changed; existing capacity is reused.
    #[expect(
        clippy::too_many_lines,
        reason = "initial quotient admission, streaming remainder and final low-product certificate share the same scalar state and output write guard"
    )]
    pub fn div_rem_3_unnormalized<Mode: CompleteDivisionMode>(
        numerator: &[Limb],
        divisor: (Limb, Limb, NonZeroUsize),
        quotient: &mut InternalMpUint,
        remainder: &mut InternalMpUint,
    ) {
        debug_assert!(
            numerator.len() >= 3,
            "division dispatch proves the input width"
        );
        let (divisor_low, divisor_middle, divisor_high) = divisor;
        let shift = divisor_high.leading_zeros();
        // SAFETY: a nonzero high limb gives shift < Limb::BITS. All
        // supported limbs have at least 16 bits, so both subtractions fit.
        let complement = unsafe { Limb::BITS.unchecked_sub(1).unchecked_sub(shift) };
        let d0 = divisor_low << shift;
        let d1 = (divisor_middle << shift) | ((divisor_low >> 1) >> complement);
        let d2 = (divisor_high.get() << shift) | ((divisor_middle >> 1) >> complement);
        // SAFETY: the dispatch boundary supplies at least three initialized
        // numerator limbs. The remaining prefix may be empty.
        let (prefix, &[bottom, middle, top]) =
            unsafe { numerator.split_last_chunk::<3>().unwrap_unchecked() };
        let upper = (top << shift) | ((middle >> 1) >> complement);
        let lower = (middle << shift) | ((bottom >> 1) >> complement);
        let guard = (top >> 1) >> complement;
        let (mut r0, mut r1, mut r2, remaining, high_digit) = if guard != 0 {
            // guard < 2^shift <= B/2 <= d2 proves the initial residue below D.
            (
                lower,
                upper,
                guard,
                // SAFETY: prefix.len()+1 = numerator.len()-2 fits usize.
                unsafe { prefix.len().unchecked_add(1) },
                0,
            )
        } else {
            // The leading three limbs are below B^3 <= 2*D, so subtracting D
            // at most once establishes the same initial residue bound.
            let leading =
                (bottom << shift) | ((prefix.last().copied().unwrap_or(0) >> 1) >> complement);
            (
                leading,
                lower,
                upper,
                prefix.len(),
                Limb::from((upper, lower, leading) >= (d2, d1, d0)),
            )
        };
        if high_digit != 0 {
            let (low_difference, low_borrow) = r0.overflowing_sub(d0);
            let (middle_difference, middle_borrow) = r1.overflowing_sub(d1);
            let (next_middle, carry_borrow) =
                middle_difference.overflowing_sub(Limb::from(low_borrow));
            r0 = low_difference;
            r1 = next_middle;
            // SAFETY: the full leading comparison proved R >= D. The
            // low-word borrow therefore fits in the high-word difference.
            r2 = unsafe {
                r2.unchecked_sub(d2)
                    .unchecked_sub(Limb::from(middle_borrow || carry_borrow))
            };
        }
        // SAFETY: a nonzero high digit occurs only with guard=0, where
        // remaining+1 = numerator.len()-2. Otherwise remaining itself fits.
        let digits = unsafe { remaining.unchecked_add(usize::from(high_digit != 0)) };
        let mut output = Mode::WRITE_QUOTIENT.then(|| quotient.prepare_limb_write(digits));
        let destination = output
            .as_mut()
            .map_or(null_mut(), LimbWriteGuard::as_mut_ptr);
        if Mode::WRITE_QUOTIENT && high_digit != 0 {
            // SAFETY: the exclusive output guard reserves digits=remaining+1
            // aligned limbs. This writes its leading slot before commit.
            unsafe {
                destination.add(remaining).write(high_digit);
            }
        }
        if remaining != 0 {
            let inverse = Self::invert_pi1(d2, d1);
            if remaining > 1 {
                // A normalized divisor requires neither an adjacent input
                // limb nor shift counts in the ordinary digit loop.
                (r0, r1, r2) = if shift == 0 {
                    divide_digits::<Mode, false>(
                        numerator,
                        remaining,
                        shift,
                        (r0, r1, r2),
                        (d0, d1, d2),
                        inverse,
                        destination,
                    )
                } else {
                    divide_digits::<Mode, true>(
                        numerator,
                        remaining,
                        shift,
                        (r0, r1, r2),
                        (d0, d1, d2),
                        inverse,
                        destination,
                    )
                };
            }
            // SAFETY: the admitted numerator contains at least three limbs.
            let incoming = unsafe { *numerator.get_unchecked(0) } << shift;
            if Mode::WRITE_REMAINDER {
                let (digit, next) = divide_step((r0, r1, r2), incoming, (d0, d1, d2), inverse);
                (r0, r1, r2) = next;
                if Mode::WRITE_QUOTIENT {
                    // SAFETY: remaining>0 gives digits>0. The exclusive
                    // output guard reserves this final uninitialized slot.
                    unsafe {
                        destination.write(digit);
                    }
                }
            } else if Mode::WRITE_QUOTIENT {
                let digit = if (r2, r1) == (d2, d1) {
                    // R<D and matching leading pairs give D-R < B.
                    // Thus U-(B-1)*D = D-(D-R)*B+incoming > 0.
                    Limb::MAX
                } else {
                    let (estimate, high, low) = Self::udiv_qr_3by2(r2, r1, r0, d2, d1, inverse);
                    if Mode::KNOWN_EXACT || high != 0 {
                        estimate
                    } else {
                        let (product_low, product_high) = ArchKernels::mul_limb_lo_hi(estimate, d0);
                        let (_, borrow) = incoming.overflowing_sub(product_low);
                        if low.borrowing_sub(product_high, borrow).1 {
                            // SAFETY: a zero estimate cannot subtract a
                            // nonzero product. Underflow proves estimate>0.
                            unsafe { estimate.unchecked_sub(1) }
                        } else {
                            estimate
                        }
                    }
                };
                // SAFETY: quotient-only modes request an output. remaining>0
                // gives digits>0, and the guard reserves this last slot.
                unsafe {
                    destination.write(digit);
                }
            }
        }
        if let Some(span) = output {
            // SAFETY: every slot below remaining was initialized by the
            // descending loop and final step. The optional leading digit
            // initialized the sole remaining slot. A zero width has no slots.
            let _ = unsafe { span.commit() };
            quotient.normalize();
        }
        if Mode::WRITE_REMAINDER {
            remainder.clone_from_slice(&[
                (r0 >> shift) | ((r1 << 1) << complement),
                (r1 >> shift) | ((r2 << 1) << complement),
                r2 >> shift,
            ]);
        }
    }
}

/// Consumes ordinary high digits, retaining the exact three-limb residue.
///
/// `SHIFTED` selects a positive normalization shift; otherwise `shift=0`.
/// `digits <= numerator.len()-2` and the initial residue is below the divisor.
/// Quotient-producing modes hold an exclusive write guard with at least
/// `digits` slots at `destination`; other modes never access that pointer.
#[expect(
    clippy::inline_always,
    reason = "inlining the once-selected digit loop preserves scalar remainder state and removes aggregate transfers across its boundary"
)]
#[inline(always)]
fn divide_digits<Mode: CompleteDivisionMode, const SHIFTED: bool>(
    numerator: &[Limb],
    digits: usize,
    shift: u32,
    mut residue: (Limb, Limb, Limb),
    divisor: (Limb, Limb, Limb),
    inverse: Limb,
    destination: *mut Limb,
) -> (Limb, Limb, Limb) {
    let complement = if SHIFTED {
        // SAFETY: dispatch selects this specialization only for
        // 0 < shift < Limb::BITS, so the complement is also in that range.
        unsafe { Limb::BITS.unchecked_sub(shift) }
    } else {
        0
    };
    for index in (1..digits).rev() {
        // SAFETY: 1 <= index < digits <= numerator.len()-2 bounds both
        // initialized input words. Each selected shift is below Limb::BITS.
        let incoming = unsafe {
            let current = *numerator.get_unchecked(index);
            if SHIFTED {
                (current << shift)
                    | (*numerator.get_unchecked(index.unchecked_sub(1)) >> complement)
            } else {
                current
            }
        };
        let (digit, next) = divide_step(residue, incoming, divisor, inverse);
        residue = next;
        if Mode::WRITE_QUOTIENT {
            // SAFETY: index < digits and the caller's exclusive guard
            // reserves at least digits aligned output slots until commit.
            unsafe {
                destination.add(index).write(digit);
            }
        }
    }
    residue
}

/// Computes `floor((R*B+a)/D)` and its three-limb remainder for R<D.
#[expect(
    clippy::inline_always,
    reason = "inlining removes per-digit aggregate arguments and return stores and keeps the streaming remainder and divisor limbs in scalar registers"
)]
#[inline(always)]
fn divide_step(
    residue: (Limb, Limb, Limb),
    incoming: Limb,
    divisor: (Limb, Limb, Limb),
    inverse: Limb,
) -> (Limb, (Limb, Limb, Limb)) {
    let (r0, r1, r2) = residue;
    let (d0, d1, d2) = divisor;
    if (r2, r1) == (d2, d1) {
        // R<D gives delta=d0-r0 in [1,B-1]. The exact digit is B-1,
        // with remainder D-delta*B+incoming in [0,D).
        let (low, carry) = incoming.overflowing_add(d0);
        // SAFETY: delta>=1 and carry<=1, so both subtractions fit.
        let gap = unsafe { d0.unchecked_sub(r0).unchecked_sub(Limb::from(carry)) };
        let (middle, borrow) = d1.overflowing_sub(gap);
        // SAFETY: normalization gives d2>=B/2, while borrow<=1.
        let high = unsafe { d2.unchecked_sub(Limb::from(borrow)) };
        return (Limb::MAX, (low, middle, high));
    }
    let (mut digit, high, middle) = Division::udiv_qr_3by2(r2, r1, r0, d2, d1, inverse);
    let (product_low, product_high) = ArchKernels::mul_limb_lo_hi(digit, d0);
    let (mut next_low, product_borrow) = incoming.overflowing_sub(product_low);
    let (mut next_middle, middle_borrow) = middle.borrowing_sub(product_high, product_borrow);
    let (mut next_high, negative) = high.overflowing_sub(Limb::from(middle_borrow));
    if negative {
        // SAFETY: subtracting a zero digit cannot underflow, so digit>0.
        digit = unsafe { digit.unchecked_sub(1) };
        let (low_sum, low_carry) = next_low.overflowing_add(d0);
        let (middle_sum, middle_carry) = next_middle.overflowing_add(d1);
        let (next_sum, extra_carry) = middle_sum.overflowing_add(Limb::from(low_carry));
        next_low = low_sum;
        next_middle = next_sum;
        // Adding D modulo B^3 restores a residue below D. Its outgoing
        // carry cancels the negative guard and is not part of the remainder.
        next_high = next_high
            .wrapping_add(d2)
            .wrapping_add(Limb::from(middle_carry || extra_carry));
    }
    (digit, (next_low, next_middle, next_high))
}
