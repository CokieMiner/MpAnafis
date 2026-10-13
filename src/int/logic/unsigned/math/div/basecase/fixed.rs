//! Streamed division by five- through eight-limb divisors.
//!
//! Each digit evaluates `U = R*B + a` with `R < D`. Division by the
//! leading pair omits its two products; the remaining product is below
//! `B^(n-1) <= D`, so one addback completes an overestimated digit.
//! Normalization occurs during input reads. Only the fixed-width residue
//! and requested outputs are retained.
//!
//! References: Knuth, *The Art of Computer Programming*, Volume 2,
//! Section 4.3.1, Algorithm D (1997); Möller and Granlund, "Improved Division
//! by Invariant Integers", Algorithms 5 and 6 (2011).

#![expect(
    unsafe_code,
    reason = "admitted fixed widths, a nonzero divisor high limb and initialized input spans bound normalization and output writes"
)]

use core::{num::NonZeroUsize, ptr::null_mut};

use super::{CompleteDivisionMode, Division, InternalMpUint, Limb, LimbWriteGuard};

impl Division {
    /// Divides by a canonical five- through eight-limb divisor.
    ///
    /// The numerator has at least as many initialized limbs as the divisor.
    /// Zero input padding is permitted. Only requested outputs change;
    /// a known-exact mode requires divisibility of the complete input.
    pub fn div_rem_fixed_unnormalized<Mode: CompleteDivisionMode>(
        numerator: &[Limb],
        divisor: &[Limb],
        quotient: &mut InternalMpUint,
        remainder: &mut InternalMpUint,
    ) {
        debug_assert!(
            (5..=8).contains(&divisor.len()),
            "fixed divisor width is five to eight"
        );
        if let &[d0, d1, d2, d3, high] = divisor {
            // SAFETY: the admitted canonical divisor has a nonzero high limb.
            let nonzero_high = unsafe { NonZeroUsize::new_unchecked(high) };
            divide_fixed::<Mode, 4>(
                numerator,
                ([d0, d1, d2, d3], nonzero_high),
                quotient,
                remainder,
            );
        } else if let &[d0, d1, d2, d3, d4, high] = divisor {
            // SAFETY: the admitted canonical divisor has a nonzero high limb.
            let nonzero_high = unsafe { NonZeroUsize::new_unchecked(high) };
            divide_fixed::<Mode, 5>(
                numerator,
                ([d0, d1, d2, d3, d4], nonzero_high),
                quotient,
                remainder,
            );
        } else if let &[d0, d1, d2, d3, d4, d5, high] = divisor {
            // SAFETY: the admitted canonical divisor has a nonzero high limb.
            let nonzero_high = unsafe { NonZeroUsize::new_unchecked(high) };
            divide_fixed::<Mode, 6>(
                numerator,
                ([d0, d1, d2, d3, d4, d5], nonzero_high),
                quotient,
                remainder,
            );
        } else {
            // SAFETY: admission permits lengths five through eight. The
            // preceding patterns exclude five through seven, leaving eight
            // initialized limbs with a nonzero canonical high limb.
            let (&[d0, d1, d2, d3, d4, d5, d6, _], nonzero_high) = unsafe {
                let words = divisor.first_chunk::<8>().unwrap_unchecked();
                (
                    words,
                    NonZeroUsize::new_unchecked(*words.last().unwrap_unchecked()),
                )
            };
            divide_fixed::<Mode, 7>(
                numerator,
                ([d0, d1, d2, d3, d4, d5, d6], nonzero_high),
                quotient,
                remainder,
            );
        }
    }
}

/// Normalizes the fixed divisor and establishes the initial residue below D.
#[expect(
    clippy::too_many_lines,
    reason = "initial residue admission, scalar high state and the last quotient certificate share one exclusive output guard"
)]
fn divide_fixed<Mode: CompleteDivisionMode, const LOW: usize>(
    numerator: &[Limb],
    divisor: ([Limb; LOW], NonZeroUsize),
    quotient: &mut InternalMpUint,
    remainder: &mut InternalMpUint,
) {
    debug_assert!(
        (4..=7).contains(&LOW),
        "fixed division width is five to eight"
    );
    // SAFETY: LOW is at most seven, so the complete divisor width fits usize.
    let width = unsafe { LOW.unchecked_add(1) };
    debug_assert!(
        numerator.len() >= width,
        "division dispatch proves input width"
    );
    let (mut d_lower, divisor_high) = divisor;
    let shift = divisor_high.leading_zeros();
    // SAFETY: the nonzero high limb gives shift<Limb::BITS, and every
    // supported limb has at least 16 bits. Both subtractions are exact.
    let complement = unsafe { Limb::BITS.unchecked_sub(1).unchecked_sub(shift) };
    let mut previous = 0;
    for word in &mut d_lower {
        let current = *word;
        *word = (current << shift) | ((previous >> 1) >> complement);
        previous = current;
    }
    let d_high = (divisor_high.get() << shift) | ((previous >> 1) >> complement);
    // SAFETY: numerator.len()>=LOW+1>=5 bounds the two high input limbs
    // and the subtraction defining the unshifted leading window.
    let (top, next, first) = unsafe {
        let last = numerator.len().unchecked_sub(1);
        (
            *numerator.get_unchecked(last),
            *numerator.get_unchecked(last.unchecked_sub(1)),
            numerator.len().unchecked_sub(width),
        )
    };
    let guard = (top >> 1) >> complement;
    // SAFETY: first+1=numerator.len()-LOW fits usize. A nonzero guard
    // makes this the first retained low-word position and digit count.
    let remaining = unsafe { first.unchecked_add(usize::from(guard != 0)) };
    let mut r_lower = [0; LOW];
    for (offset, word) in r_lower.iter_mut().enumerate() {
        // SAFETY: offset<LOW and remaining<=numerator.len()-LOW bound
        // the current initialized limb. The predecessor is read only
        // for a positive index; both selected shifts are below Limb::BITS.
        *word = unsafe {
            let index = remaining.unchecked_add(offset);
            let current = *numerator.get_unchecked(index);
            let preceding = if index == 0 {
                0
            } else {
                *numerator.get_unchecked(index.unchecked_sub(1))
            };
            (current << shift) | ((preceding >> 1) >> complement)
        };
    }
    let mut r_high = if guard == 0 {
        (top << shift) | ((next >> 1) >> complement)
    } else {
        guard
    };
    // A positive guard is below d_high. Otherwise R<B^width<=2D;
    // one comparison and at most one subtraction establish R<D.
    let high_digit = Limb::from(
        guard == 0
            && (r_high > d_high
                || (r_high == d_high && r_lower.iter().rev().cmp(d_lower.iter().rev()).is_ge())),
    );
    if high_digit != 0 {
        let mut borrow = false;
        for (word, &divisor_word) in r_lower.iter_mut().zip(&d_lower) {
            let (difference, next_borrow) = word.borrowing_sub(divisor_word, borrow);
            *word = difference;
            borrow = next_borrow;
        }
        // SAFETY: the complete comparison proved R>=D, so its low-word
        // borrow fits in the high-word difference.
        r_high = unsafe {
            r_high
                .unchecked_sub(d_high)
                .unchecked_sub(Limb::from(borrow))
        };
    }
    // SAFETY: high_digit!=0 requires guard=0, so remaining+1 equals
    // numerator.len()-LOW. Otherwise remaining itself fits usize.
    let digits = unsafe { remaining.unchecked_add(usize::from(high_digit != 0)) };
    let mut output = Mode::WRITE_QUOTIENT.then(|| quotient.prepare_limb_write(digits));
    let destination = output
        .as_mut()
        .map_or(null_mut(), LimbWriteGuard::as_mut_ptr);
    if Mode::WRITE_QUOTIENT && high_digit != 0 {
        // SAFETY: the exclusive guard reserves digits=remaining+1
        // aligned slots. This initializes the sole leading slot.
        unsafe {
            destination.add(remaining).write(high_digit);
        }
    }
    if remaining != 0 {
        // SAFETY: LOW>=4 gives both initialized leading low-state words.
        let leading = unsafe { *d_lower.last().unwrap_unchecked() };
        let inverse = Division::invert_pi1(d_high, leading);
        if remaining > 1 {
            if shift == 0 {
                divide_digits::<Mode, LOW, false>(
                    numerator,
                    remaining,
                    shift,
                    (&mut r_lower, &mut r_high),
                    (&d_lower, d_high),
                    inverse,
                    destination,
                );
            } else {
                divide_digits::<Mode, LOW, true>(
                    numerator,
                    remaining,
                    shift,
                    (&mut r_lower, &mut r_high),
                    (&d_lower, d_high),
                    inverse,
                    destination,
                );
            }
        }
        // SAFETY: the admitted input contains at least five initialized limbs.
        let incoming = unsafe { *numerator.first().unwrap_unchecked() } << shift;
        if Mode::WRITE_REMAINDER {
            let digit = divide_step(
                (&mut r_lower, &mut r_high),
                incoming,
                (&d_lower, d_high),
                inverse,
            );
            if Mode::WRITE_QUOTIENT {
                // SAFETY: remaining>0 gives an exclusive initialized-span
                // guard with a final slot; no earlier iteration writes it.
                unsafe {
                    destination.write(digit);
                }
            }
        } else if Mode::WRITE_QUOTIENT {
            // SAFETY: LOW>=4 bounds the two retained leading low words.
            let (middle, lower) = unsafe {
                (
                    *r_lower.last().unwrap_unchecked(),
                    *r_lower.get_unchecked(LOW.unchecked_sub(2)),
                )
            };
            let digit = if (r_high, middle) == (d_high, leading) {
                Limb::MAX
            } else {
                let (estimate, high, low) =
                    Division::udiv_qr_3by2(r_high, middle, lower, d_high, leading, inverse);
                if Mode::KNOWN_EXACT || high != 0 {
                    estimate
                } else {
                    // The high residue is zero. Only the low-product
                    // borrow is required; no final remainder is written.
                    let mut carry = 0;
                    let mut borrow = false;
                    let mut input = incoming;
                    // SAFETY: LOW>=4, so the omitted prefix has LOW-1 words.
                    let prefix = unsafe { LOW.unchecked_sub(1) };
                    for (index, &word) in d_lower.iter().take(prefix).enumerate() {
                        let (product, next_carry) = estimate.carrying_mul(word, carry);
                        borrow = input.borrowing_sub(product, borrow).1;
                        carry = next_carry;
                        // SAFETY: index<prefix<LOW bounds initialized state.
                        input = unsafe { *r_lower.get_unchecked(index) };
                    }
                    if low.borrowing_sub(carry, borrow).1 {
                        // SAFETY: subtracting a zero estimate cannot borrow.
                        unsafe { estimate.unchecked_sub(1) }
                    } else {
                        estimate
                    }
                }
            };
            // SAFETY: remaining>0 reserves this final aligned quotient slot.
            unsafe {
                destination.write(digit);
            }
        }
    }
    if let Some(span) = output {
        // SAFETY: the descending loop and final digit initialize every
        // lower slot. The optional high digit initializes the leading slot.
        let _ = unsafe { span.commit() };
        quotient.normalize();
    }
    if Mode::WRITE_REMAINDER {
        write_remainder(&mut r_lower, r_high, shift, complement, remainder);
    }
}

/// Consumes ordinary digits with a retained residue below the normalized divisor.
#[expect(
    clippy::inline_always,
    reason = "inlining exposes the borrowed fixed residue to the selected loop and removes per-digit calls"
)]
#[inline(always)]
fn divide_digits<Mode: CompleteDivisionMode, const LOW: usize, const SHIFTED: bool>(
    numerator: &[Limb],
    digits: usize,
    shift: u32,
    residue: (&mut [Limb; LOW], &mut Limb),
    divisor: (&[Limb; LOW], Limb),
    inverse: Limb,
    destination: *mut Limb,
) {
    let (r_lower, r_high) = residue;
    let complement = if SHIFTED {
        // SAFETY: the caller selects this variant only for 0<shift<Limb::BITS.
        unsafe { Limb::BITS.unchecked_sub(shift) }
    } else {
        0
    };
    for index in (1..digits).rev() {
        // SAFETY: 1<=index<digits<=numerator.len()-LOW bounds the current
        // and preceding initialized input limbs and the exact subtraction.
        let incoming = unsafe {
            let current = *numerator.get_unchecked(index);
            if SHIFTED {
                (current << shift)
                    | (*numerator.get_unchecked(index.unchecked_sub(1)) >> complement)
            } else {
                current
            }
        };
        let digit = divide_step((&mut *r_lower, &mut *r_high), incoming, divisor, inverse);
        if Mode::WRITE_QUOTIENT {
            // SAFETY: the caller's exclusive guard has at least digits
            // slots. Each descending iteration initializes one distinct slot.
            unsafe {
                destination.add(index).write(digit);
            }
        }
    }
}

/// Computes one exact digit and residue from `R*B+a`, with R<D.
#[expect(
    clippy::inline_always,
    reason = "inlining keeps the fixed product and borrow chain in the surrounding loop and exposes its residue updates"
)]
#[inline(always)]
fn divide_step<const LOW: usize>(
    residue: (&mut [Limb; LOW], &mut Limb),
    incoming: Limb,
    divisor: (&[Limb; LOW], Limb),
    inverse: Limb,
) -> Limb {
    let (r_lower, r_high) = residue;
    let (d_lower, d_high) = divisor;
    // SAFETY: the selected widths give LOW>=4; both arrays are initialized.
    let (leading, middle, lower, prefix) = unsafe {
        (
            *d_lower.last().unwrap_unchecked(),
            *r_lower.last().unwrap_unchecked(),
            *r_lower.get_unchecked(LOW.unchecked_sub(2)),
            LOW.unchecked_sub(1),
        )
    };
    if (*r_high, middle) == (d_high, leading) {
        // delta=D-R lies in [1,B^(LOW-1)). Reconstruct
        // D-delta*B+incoming with additions and subtractions only.
        // SAFETY: LOW>=4 supplies the initialized bottom divisor limb.
        let (bottom, carry) =
            incoming.overflowing_add(unsafe { *d_lower.first().unwrap_unchecked() });
        // Compute delta-carry directly. delta>=1 and carry<=1 make
        // this difference nonnegative across the complete low prefix.
        let mut borrow = carry;
        for (word, &d) in r_lower.iter_mut().zip(d_lower).take(prefix) {
            let (difference, next_borrow) = d.borrowing_sub(*word, borrow);
            *word = difference;
            borrow = next_borrow;
        }
        // SAFETY: LOW>=4 supplies the initialized bottom gap word. Save it
        // before overwriting the same slot with the bottom result.
        let mut gap = unsafe { *r_lower.first().unwrap_unchecked() };
        // SAFETY: LOW>=4 gives an initialized residue bottom word.
        unsafe {
            *r_lower.first_mut().unwrap_unchecked() = bottom;
        }
        borrow = false;
        for (word, &d) in r_lower.iter_mut().skip(1).zip(d_lower.iter().skip(1)) {
            let next_gap = *word;
            let (difference, next_borrow) = d.borrowing_sub(gap, borrow);
            *word = difference;
            borrow = next_borrow;
            gap = next_gap;
        }
        // SAFETY: d_high>=B/2 and borrow<=1, so the subtraction is exact.
        *r_high = unsafe { d_high.unchecked_sub(Limb::from(borrow)) };
        return Limb::MAX;
    }
    let (mut digit, high, low) =
        Division::udiv_qr_3by2(*r_high, middle, lower, d_high, leading, inverse);
    let mut input = incoming;
    let mut carry = 0;
    let mut borrow = false;
    for (word, &d) in r_lower.iter_mut().zip(d_lower).take(prefix) {
        // Preserve the next input before replacing this state word.
        let next_input = *word;
        let (product, next_carry) = digit.carrying_mul(d, carry);
        let (difference, next_borrow) = input.borrowing_sub(product, borrow);
        *word = difference;
        carry = next_carry;
        borrow = next_borrow;
        input = next_input;
    }
    let (next_middle, middle_borrow) = low.borrowing_sub(carry, borrow);
    // SAFETY: LOW>=4 supplies this last initialized low-state slot.
    unsafe {
        *r_lower.last_mut().unwrap_unchecked() = next_middle;
    }
    let (mut next_high, negative) = high.overflowing_sub(Limb::from(middle_borrow));
    if negative {
        // SAFETY: a zero digit cannot subtract a nonzero product. Its borrow
        // proves digit>0; the omitted product is below the normalized divisor.
        digit = unsafe { digit.unchecked_sub(1) };
        let mut addback_carry = false;
        for (word, &d) in r_lower.iter_mut().zip(d_lower) {
            let (sum, next_carry) = word.carrying_add(d, addback_carry);
            *word = sum;
            addback_carry = next_carry;
        }
        // Adding D modulo B^(LOW+1) removes the negative guard and restores R<D.
        next_high = next_high
            .wrapping_add(d_high)
            .wrapping_add(Limb::from(addback_carry));
    }
    *r_high = next_high;
    digit
}

/// Denormalizes the residue and writes only its significant magnitude.
fn write_remainder<const LOW: usize>(
    r_lower: &mut [Limb; LOW],
    mut r_high: Limb,
    shift: u32,
    complement: u32,
    remainder: &mut InternalMpUint,
) {
    // SAFETY: LOW>=4, so the last low-word index is nonnegative.
    let last = unsafe { LOW.unchecked_sub(1) };
    for index in 0..last {
        // SAFETY: index<LOW-1 bounds both initialized words. Load them
        // before writing the lower slot under the exclusive array borrow;
        // ascending writes preserve every still-required higher word.
        unsafe {
            let current = *r_lower.get_unchecked(index);
            let higher = *r_lower.get_unchecked(index.unchecked_add(1));
            *r_lower.get_unchecked_mut(index) = (current >> shift) | ((higher << 1) << complement);
        }
    }
    // SAFETY: LOW>=4 supplies the last initialized low word. The high
    // residue is still normalized, and both shift counts are below Limb::BITS.
    unsafe {
        let word = r_lower.last_mut().unwrap_unchecked();
        *word = (*word >> shift) | ((r_high << 1) << complement);
    }
    r_high >>= shift;
    let length = if r_high != 0 {
        // SAFETY: LOW<=7 makes the complete significant width exact.
        unsafe { LOW.unchecked_add(1) }
    } else {
        r_lower
            .iter()
            .rposition(|&word| word != 0)
            .map_or(0, |index| {
                // SAFETY: index<LOW<=7 makes the significant prefix length exact.
                unsafe { index.unchecked_add(1) }
            })
    };
    // Reserve the significant magnitude, preserving inline storage when
    // the denormalized remainder has at most four limbs.
    let mut span = remainder.prepare_limb_write(length);
    let destination = span.as_mut_ptr();
    // SAFETY: min(length,LOW) bounds initialized source words and the
    // exclusive writable destination. The local array and owner storage
    // are disjoint. A nonzero high limb reserves length=LOW+1.
    unsafe {
        destination.copy_from_nonoverlapping(r_lower.as_ptr(), length.min(LOW));
        if r_high != 0 {
            destination.add(LOW).write(r_high);
        }
    }
    // SAFETY: every reserved slot is initialized. Its high limb is nonzero
    // by length selection, or the empty prefix represents canonical zero.
    let _ = unsafe { span.commit() };
}
