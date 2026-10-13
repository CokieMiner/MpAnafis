//! Four-limb division with normalization during input reads.
//!
//! Each digit updates `U = R*B + a`, with `0 <= R < D`. Division by the
//! leading divisor pair removes its two high products. Two low products
//! complete the subtraction; their combined value is below B^3 <= D,
//! so a negative residue requires exactly one addback.
//!
//! The remainder stays in four scalar limbs. The final quotient-only digit
//! omits both low products when its high residue limb or known divisibility
//! certifies the estimate.
//!
//! Reference: D. E. Knuth, *The Art of Computer Programming*, Volume 2,
//! 3rd ed., Section 4.3.1, Algorithm D (1997).

#![expect(
    unsafe_code,
    reason = "the fixed divisor width, nonzero high limb and admitted input span bound normalization, output writes and exact corrections"
)]

use core::{num::NonZeroUsize, ptr::null_mut};

use super::{ArchKernels, CompleteDivisionMode, Division, InternalMpUint, Limb, LimbWriteGuard};

impl Division {
    /// Divides an initialized numerator of at least four limbs by a
    /// four-limb divisor with a nonzero high limb. Zero input padding is
    /// permitted. Only requested outputs change; existing capacity is reused.
    /// A known-exact mode requires divisibility of the complete input.
    #[expect(
        clippy::too_many_lines,
        reason = "initial quotient admission, retained scalar residue and final low-product certificate share one output write guard"
    )]
    pub fn div_rem_4_unnormalized<Mode: CompleteDivisionMode>(
        numerator: &[Limb],
        divisor: (Limb, Limb, Limb, NonZeroUsize),
        quotient: &mut InternalMpUint,
        remainder: &mut InternalMpUint,
    ) {
        debug_assert!(
            numerator.len() >= 4,
            "division dispatch proves the input width"
        );
        let (divisor_bottom, divisor_low, divisor_middle, divisor_high) = divisor;
        let shift = divisor_high.leading_zeros();
        // SAFETY: a nonzero high limb gives shift < Limb::BITS. Every
        // supported limb has at least 16 bits, so both subtractions fit.
        let complement = unsafe { Limb::BITS.unchecked_sub(1).unchecked_sub(shift) };
        let d0 = divisor_bottom << shift;
        let d1 = (divisor_low << shift) | ((divisor_bottom >> 1) >> complement);
        let d2 = (divisor_middle << shift) | ((divisor_low >> 1) >> complement);
        let d3 = (divisor_high.get() << shift) | ((divisor_middle >> 1) >> complement);
        // SAFETY: dispatch supplies at least four initialized numerator
        // limbs; the remaining prefix may be empty.
        let (prefix, &[bottom, lower_input, middle, top]) =
            unsafe { numerator.split_last_chunk::<4>().unwrap_unchecked() };
        let upper = (top << shift) | ((middle >> 1) >> complement);
        let center = (middle << shift) | ((lower_input >> 1) >> complement);
        let lower = (lower_input << shift) | ((bottom >> 1) >> complement);
        let guard = (top >> 1) >> complement;
        let (mut r0, mut r1, mut r2, mut r3, remaining, high_digit) = if guard != 0 {
            // guard < 2^shift <= B/2 <= d3 establishes R<D.
            (
                lower,
                center,
                upper,
                guard,
                // SAFETY: prefix.len()+1 = numerator.len()-3 fits usize.
                unsafe { prefix.len().unchecked_add(1) },
                0,
            )
        } else {
            // The leading four limbs are below B^4 <= 2*D. At most one
            // subtraction establishes the same initial residue bound.
            let leading =
                (bottom << shift) | ((prefix.last().copied().unwrap_or(0) >> 1) >> complement);
            (
                leading,
                lower,
                center,
                upper,
                prefix.len(),
                Limb::from((upper, center, lower, leading) >= (d3, d2, d1, d0)),
            )
        };
        if high_digit != 0 {
            let (bottom_difference, bottom_borrow) = r0.overflowing_sub(d0);
            let (low_difference, low_borrow) = r1.borrowing_sub(d1, bottom_borrow);
            let (middle_difference, middle_borrow) = r2.borrowing_sub(d2, low_borrow);
            r0 = bottom_difference;
            r1 = low_difference;
            r2 = middle_difference;
            // SAFETY: the full comparison proved R>=D, so the lower-word
            // borrow fits in the high-word difference.
            r3 = unsafe {
                r3.unchecked_sub(d3)
                    .unchecked_sub(Limb::from(middle_borrow))
            };
        }
        // SAFETY: high_digit!=0 requires guard=0, where remaining+1 equals
        // numerator.len()-3. Otherwise remaining itself fits usize.
        let digits = unsafe { remaining.unchecked_add(usize::from(high_digit != 0)) };
        let mut output = Mode::WRITE_QUOTIENT.then(|| quotient.prepare_limb_write(digits));
        let destination = output
            .as_mut()
            .map_or(null_mut(), LimbWriteGuard::as_mut_ptr);
        if Mode::WRITE_QUOTIENT && high_digit != 0 {
            // SAFETY: the exclusive guard reserves digits=remaining+1
            // aligned slots; this initializes its sole leading slot.
            unsafe {
                destination.add(remaining).write(high_digit);
            }
        }
        if remaining != 0 {
            let inverse = Self::invert_pi1(d3, d2);
            if remaining > 1 {
                (r0, r1, r2, r3) = if shift == 0 {
                    divide_digits::<Mode, false>(
                        numerator,
                        remaining,
                        shift,
                        (r0, r1, r2, r3),
                        (d0, d1, d2, d3),
                        inverse,
                        destination,
                    )
                } else {
                    divide_digits::<Mode, true>(
                        numerator,
                        remaining,
                        shift,
                        (r0, r1, r2, r3),
                        (d0, d1, d2, d3),
                        inverse,
                        destination,
                    )
                };
            }
            // SAFETY: the admitted initialized input contains at least four limbs.
            let incoming = unsafe { *numerator.get_unchecked(0) } << shift;
            if Mode::WRITE_REMAINDER {
                let (digit, next) =
                    divide_step((r0, r1, r2, r3), incoming, (d0, d1, d2, d3), inverse);
                (r0, r1, r2, r3) = next;
                if Mode::WRITE_QUOTIENT {
                    // SAFETY: remaining>0 gives digits>0, and the exclusive
                    // guard reserves this final uninitialized output slot.
                    unsafe {
                        destination.write(digit);
                    }
                }
            } else if Mode::WRITE_QUOTIENT {
                let digit = if (r3, r2) == (d3, d2) {
                    // Matching leading pairs give 1<=D-R<B^2. Thus
                    // D-(D-R)*B+incoming is in [0,D), with digit B-1.
                    Limb::MAX
                } else {
                    let (estimate, high, low) = Self::udiv_qr_3by2(r3, r2, r1, d3, d2, inverse);
                    if Mode::KNOWN_EXACT || high != 0 {
                        estimate
                    } else {
                        let (product_low, product_high) = ArchKernels::mul_limb_lo_hi(estimate, d0);
                        let (product_middle, product_top) = estimate.carrying_mul(d1, product_high);
                        let (_, bottom_borrow) = incoming.overflowing_sub(product_low);
                        let (_, low_borrow) = r0.borrowing_sub(product_middle, bottom_borrow);
                        if low.borrowing_sub(product_top, low_borrow).1 {
                            // SAFETY: subtracting a zero estimate cannot
                            // underflow; this borrow proves estimate>0.
                            unsafe { estimate.unchecked_sub(1) }
                        } else {
                            estimate
                        }
                    }
                };
                // SAFETY: this mode requests the quotient; remaining>0
                // reserves the final slot under the exclusive output guard.
                unsafe {
                    destination.write(digit);
                }
            }
        }
        if let Some(span) = output {
            // SAFETY: the descending loop and final step initialize every
            // slot below remaining. The optional high digit initializes the
            // sole leading slot. A zero-width guard has no slots.
            let _ = unsafe { span.commit() };
            quotient.normalize();
        }
        if Mode::WRITE_REMAINDER {
            remainder.clone_from_slice(&[
                (r0 >> shift) | ((r1 << 1) << complement),
                (r1 >> shift) | ((r2 << 1) << complement),
                (r2 >> shift) | ((r3 << 1) << complement),
                r3 >> shift,
            ]);
        }
    }
}

/// Consumes ordinary digits, retaining the exact four-limb residue below D.
///
/// `SHIFTED` selects a positive normalization shift; otherwise shift=0.
/// `digits <= numerator.len()-3`. Quotient-producing modes hold an exclusive
/// guard with at least digits slots; other modes never access destination.
#[expect(
    clippy::inline_always,
    reason = "inlining the once-selected loop removes aggregate transfers and exposes its scalar remainder state to the driver"
)]
#[inline(always)]
fn divide_digits<Mode: CompleteDivisionMode, const SHIFTED: bool>(
    numerator: &[Limb],
    digits: usize,
    shift: u32,
    mut residue: (Limb, Limb, Limb, Limb),
    divisor: (Limb, Limb, Limb, Limb),
    inverse: Limb,
    destination: *mut Limb,
) -> (Limb, Limb, Limb, Limb) {
    let complement = if SHIFTED {
        // SAFETY: dispatch selects this variant only for 0<shift<Limb::BITS.
        unsafe { Limb::BITS.unchecked_sub(shift) }
    } else {
        0
    };
    for index in (1..digits).rev() {
        // SAFETY: 1<=index<digits<=numerator.len()-3 bounds both initialized
        // input words. Each selected shift is below Limb::BITS.
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
            // SAFETY: index<digits lies inside the caller's exclusive guard;
            // each descending iteration initializes one distinct aligned slot.
            unsafe {
                destination.add(index).write(digit);
            }
        }
    }
    residue
}

/// Computes `floor((R*B+a)/D)` and its four-limb remainder for R<D.
#[expect(
    clippy::inline_always,
    reason = "inlining removes per-digit tuple transfers and keeps the low products and borrow chain in the surrounding scalar loop"
)]
#[inline(always)]
fn divide_step(
    residue: (Limb, Limb, Limb, Limb),
    incoming: Limb,
    divisor: (Limb, Limb, Limb, Limb),
    inverse: Limb,
) -> (Limb, (Limb, Limb, Limb, Limb)) {
    let (r0, r1, r2, r3) = residue;
    let (d0, d1, d2, d3) = divisor;
    if (r3, r2) == (d3, d2) {
        // Matching leading pairs and R<D give delta=D-R in [1,B^2-1].
        // The exact digit is B-1, with remainder D-delta*B+incoming.
        let (delta_low, delta_borrow) = d0.overflowing_sub(r0);
        // SAFETY: the strict lower-pair comparison proves the high-word
        // difference can absorb delta_borrow without underflow.
        let delta_high = unsafe { d1.unchecked_sub(r1).unchecked_sub(Limb::from(delta_borrow)) };
        let (bottom, carry) = incoming.overflowing_add(d0);
        let (gap_low, gap_borrow) = delta_low.overflowing_sub(Limb::from(carry));
        // SAFETY: delta>=1 and carry<=1 give delta-carry>=0. If its low
        // subtraction borrows, delta_high is therefore at least one.
        let gap_high = unsafe { delta_high.unchecked_sub(Limb::from(gap_borrow)) };
        let (low, low_borrow) = d1.overflowing_sub(gap_low);
        let (middle, middle_borrow) = d2.borrowing_sub(gap_high, low_borrow);
        // SAFETY: normalization gives d3>=B/2; middle_borrow<=1.
        let high = unsafe { d3.unchecked_sub(Limb::from(middle_borrow)) };
        return (Limb::MAX, (bottom, low, middle, high));
    }
    let (mut digit, high, middle) = Division::udiv_qr_3by2(r3, r2, r1, d3, d2, inverse);
    let (product_low, product_high) = ArchKernels::mul_limb_lo_hi(digit, d0);
    // q*d1+high(q*d0) < B^2; these are the remaining two product limbs.
    let (product_middle, product_top) = digit.carrying_mul(d1, product_high);
    let (mut next_bottom, bottom_borrow) = incoming.overflowing_sub(product_low);
    let (mut next_low, low_borrow) = r0.borrowing_sub(product_middle, bottom_borrow);
    let (mut next_middle, middle_borrow) = middle.borrowing_sub(product_top, low_borrow);
    let (mut next_high, negative) = high.overflowing_sub(Limb::from(middle_borrow));
    if negative {
        // SAFETY: a zero digit cannot subtract a nonzero product; underflow
        // therefore proves digit>0. The omitted product is below D.
        digit = unsafe { digit.unchecked_sub(1) };
        let (bottom_sum, bottom_carry) = next_bottom.overflowing_add(d0);
        let (low_sum, low_carry) = next_low.carrying_add(d1, bottom_carry);
        let (middle_sum, middle_carry) = next_middle.carrying_add(d2, low_carry);
        next_bottom = bottom_sum;
        next_low = low_sum;
        next_middle = middle_sum;
        // Adding D modulo B^4 cancels the negative guard and restores R<D.
        next_high = next_high
            .wrapping_add(d3)
            .wrapping_add(Limb::from(middle_carry));
    }
    (digit, (next_bottom, next_low, next_middle, next_high))
}
