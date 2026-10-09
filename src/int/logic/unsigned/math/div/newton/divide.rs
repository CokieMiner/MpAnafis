//! Block-wise Barrett division using a Newton reciprocal.
//!
//! Each block consumes the preceding remainder and a low dividend window.
//! Quotient blocks are written directly to the output.
//!
//! For normalized D, write α=D/B^n in [1/2,1). The block driver supplies
//! U<D*B^t. With H=floor(U/B^n), L=U-H*B^n and the lower reciprocal
//! V=B^(n+t)/D-ε, 0<=ε<2, use Q0=floor(H*V/B^t):
//!
//! ```text
//! U/D - H*V/B^t = L/D + H*ε/B^t
//!              < 1/α + 2α <= 3.
//! ```
//!
//! The first inequality uses L<B^n and H/B^t<α. The last is equivalent
//! to (2α-1)*(α-1)<=0. Flooring adds less than one, so Q-Q0 is in {0,1,2,3}.
//! An ordinary block multiplies only t dividend limbs and t inverse limbs.
//!
//! The final quotient-only block retains one additional dividend and inverse
//! limb to avoid forming a residue. Upward rounding costs <1/α² and Newton
//! costs <2, so ε<2+1/α². For its scaled estimate A, the real error is
//! <2/α+2α<=5; thus floor(B*U/D)-A<=5. Its low digit certifies rounding
//! unless adding five can carry into the quotient. At most one correction
//! is required when this guard is ambiguous.

#![expect(
    unsafe_code,
    reason = "normalized Newton blocks and checked scratch reservations establish quotient-pointer and product-buffer bounds"
)]

use core::{
    cmp::Ordering,
    mem::replace,
    num::NonZeroUsize,
    ops::Div,
    ptr::{copy_nonoverlapping, null_mut},
};

use super::{
    Addition, ArchKernels, DivScratch, Division, InternalMpUint, Limb,
    NEWTON_SMALL_QUOTIENT_BLOCK_RATIO, ScratchBuffer,
};

impl Division {
    /// Divides `num_a` by `den_b` using a reciprocal shared by all quotient blocks.
    ///
    /// `WRITE_REMAINDER` selects whether the exact residue is denormalized into
    /// `rem_out`. Each intermediate block reconstructs its remainder
    /// in place for consumption by the following block.
    /// Quotient-only division skips the final remainder copy and denormalization.
    /// Remainder-only output consumes uncorrected estimates only in residue products.
    /// With `EXACT`, known divisibility supplies quotient-only output using two correction bits.
    /// Ordinary quotient-only output reuses a reciprocal guard, when present,
    /// to avoid the final residue product unless rounding is ambiguous.
    pub fn newton<const WRITE_QUOTIENT: bool, const WRITE_REMAINDER: bool, const EXACT: bool>(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        debug_assert!(!den_b.is_zero(), "division requires a non-zero divisor");
        debug_assert!(
            !EXACT || (WRITE_QUOTIENT && !WRITE_REMAINDER),
            "invalid exact outputs"
        );
        if num_a < den_b {
            if WRITE_QUOTIENT {
                quotient_out.clear();
            }
            if WRITE_REMAINDER {
                rem_out.clone_from(num_a);
            } else {
                rem_out.clear();
            }
            return;
        }
        let v_limbs = den_b.limbs();
        let u_limbs = num_a.limbs();
        // SAFETY: the normalized nonzero divisor has a most-significant limb.
        let shift = unsafe { v_limbs.last().unwrap_unchecked() }.leading_zeros();
        // Keep normalization storage independent; borrow an already normalized divisor.
        let mut den_storage = replace(&mut scratch.newton_v_norm, ScratchBuffer::acquire(0));
        let divisor = if shift == 0 {
            v_limbs
        } else {
            Self::shift_limbs_left::<false>(v_limbs, shift, &mut den_storage);
            den_storage.as_slice()
        };
        let mut dividend = replace(&mut scratch.newton_u_norm, ScratchBuffer::acquire(0));
        Self::shift_limbs_left::<true>(u_limbs, shift, &mut dividend);
        let n = divisor.len();
        // SAFETY: num >= den proves len >= n; the one-limb extension of a
        // materialized Limb slice fits usize on 16-, 32-, and 64-bit targets.
        let q_len = unsafe { dividend.len().unchecked_sub(n) };
        if WRITE_QUOTIENT {
            let mut quotient = quotient_out.prepare_limb_write(q_len);
            // SAFETY: q_len reserved digits are disjoint from dividend and scratch;
            // the driver initializes each digit before commit.
            unsafe {
                Self::newton_div_blocks::<true, WRITE_REMAINDER, EXACT>(
                    &mut dividend,
                    divisor,
                    quotient.as_mut_ptr(),
                    q_len,
                    scratch,
                );
                let _ = quotient.commit();
            }
            quotient_out.normalize();
        } else {
            // SAFETY: the false specialization never uses its null quotient
            // pointer. Normalization and the zero guard establish the same
            // dividend geometry and high-window bound as the writing path.
            unsafe {
                Self::newton_div_blocks::<false, true, false>(
                    &mut dividend,
                    divisor,
                    null_mut(),
                    q_len,
                    scratch,
                );
            }
        }

        if WRITE_REMAINDER {
            // SAFETY: the remainder-producing driver leaves n initialized
            // residue limbs in the dividend, disjoint from both public outputs.
            let remainder = unsafe { dividend.get_unchecked(..n) };
            if shift == 0 {
                rem_out.clone_from_slice(remainder);
            } else {
                let mut output = rem_out.prepare_limb_write(n);
                // SAFETY: 0<shift<LIMB_BITS. The disjoint source supplies n
                // initialized residue limbs; preparation reserves n writable
                // output limbs, all initialized by the kernel before commit.
                unsafe {
                    let _ = ArchKernels::rshift_into_unchecked(
                        output.as_mut_ptr(),
                        remainder.as_ptr(),
                        n,
                        shift,
                    );
                    let _ = output.commit();
                }
                rem_out.normalize();
            }
        } else {
            rem_out.clear();
        }
        scratch.newton_v_norm = den_storage;
        scratch.newton_u_norm = dividend;
    }

    /// Divides a normalized dividend in place, retaining its remainder when requested.
    /// The divisor is nonempty and normalized, `quo_len > 0`, and
    /// `num.len() == den.len() + quo_len`. The high `den.len()` limbs are below
    /// `den`, so each quotient block fits its destination without an extra digit.
    ///
    /// # Safety
    /// With `WRITE_QUOTIENT`, `quo` provides `quo_len` aligned writable limbs
    /// disjoint from inputs and scratch. They may be uninitialized; the driver
    /// initializes the complete span. Otherwise `quo` is never used and may be null.
    /// At least one output mode is enabled. `EXACT` requires divisibility and
    /// quotient-only output; the last residue is unused.
    pub unsafe fn newton_div_blocks<
        const WRITE_QUOTIENT: bool,
        const WRITE_REMAINDER: bool,
        const EXACT: bool,
    >(
        dividend: &mut [Limb],
        den: &[Limb],
        quo: *mut Limb,
        quo_len: usize,
        scratch: &mut DivScratch,
    ) {
        debug_assert!(WRITE_QUOTIENT || WRITE_REMAINDER, "empty output policy");
        // SAFETY: normalized dispatch supplies a nonempty divisor. Its
        // positive width also supplies the denominator for block partitioning.
        let divisor_width = unsafe { NonZeroUsize::new_unchecked(den.len()) };
        let n = divisor_width.get();
        debug_assert!(quo_len > 0, "empty quotient span");
        debug_assert_eq!(
            dividend.len().checked_sub(n),
            Some(quo_len),
            "invalid block lengths"
        );
        // SAFETY: normalized dispatch supplies a nonempty divisor and at
        // least one quotient slot. Both positive widths fit their limb owners.
        let quotient_width = unsafe { NonZeroUsize::new_unchecked(quo_len) };
        // SAFETY: the caller supplies exactly n+quo_len initialized limbs.
        // Deriving the active view from those widths preserves that equality
        // throughout the block recurrence; the materialized span bounds the sum.
        let num = unsafe { dividend.get_unchecked_mut(..n.unchecked_add(quotient_width.get())) };
        let mut end = quotient_width.get();
        if num.last() == Some(&0) {
            // A zero guard bounds the leading n-limb window below B^n <= 2D.
            // Its quotient digit is zero or one. Remove that digit before sizing
            // the reciprocal, leaving the subsequent high window strictly below D.
            // SAFETY: end > 0 and num.len() = n + end, so the n-limb window
            // ending before the guard and the corresponding quotient digit exist.
            unsafe {
                end = end.unchecked_sub(1);
                let high = num.get_unchecked_mut(end..end.unchecked_add(n));
                let digit = Limb::from(InternalMpUint::cmp_limbs(high, den) != Ordering::Less);
                if digit != 0 {
                    let borrow = Addition::sub_slice_in_place(high, den);
                    debug_assert_eq!(borrow, 0, "the leading window is at least D");
                }
                if WRITE_QUOTIENT {
                    *quo.add(end) = digit;
                }
            }
            if end == 0 {
                return;
            }
        }
        // Equal widths over ceil(end/n) blocks reduce reciprocal precision
        // and the largest products. Below the configured ratio, one block
        // avoids a second divisor-width residue; intermediate shapes use two.
        let block = if end > n {
            // SAFETY: normalized D has n>0 limbs and end>n. Both ceil quotients
            // lie in [1,end]; materialized limb widths bound each increment.
            unsafe {
                let last = end.unchecked_sub(1);
                let count = Div::div(last, divisor_width).unchecked_add(1);
                Div::div(last, NonZeroUsize::new_unchecked(count)).unchecked_add(1)
            }
        } else if end <= n.div_euclid(NEWTON_SMALL_QUOTIENT_BLOCK_RATIO) {
            end
        } else {
            end.div_ceil(2)
        };
        let (reciprocal, discarded) = if WRITE_REMAINDER || EXACT {
            Self::newton_block_reciprocal::<false>(den, block, scratch)
        } else {
            Self::newton_block_reciprocal::<true>(den, block, scratch)
        };
        // SAFETY: the reciprocal builder reserves the discarded guard limb and
        // returns a nonzero high part. The owner outlives every block call.
        let inverse = unsafe { reciprocal.limbs().get_unchecked(discarded..) };
        while end != 0 {
            let start = end.saturating_sub(block);
            // SAFETY: start < end <= quo_len and num.len() = n + quo_len.
            // The output and mutable dividend have disjoint owners. The previous
            // step leaves its remainder at end..end+n, exactly this window's high part.
            unsafe {
                let window = num.get_unchecked_mut(start..end.unchecked_add(n));
                let destination = if WRITE_QUOTIENT {
                    quo.add(start)
                } else {
                    null_mut()
                };
                if EXACT && start == 0 {
                    // Earlier blocks subtract multiples of D; the final exact
                    // window has no remainder consumer.
                    newton_div_block::<true, true, false>(
                        window,
                        den,
                        inverse,
                        destination,
                        scratch,
                    );
                } else if !WRITE_REMAINDER && start == 0 && discarded == 1 {
                    // The final quotient block consumes its guard and discards the residue.
                    newton_div_block::<true, false, true>(
                        window,
                        den,
                        reciprocal.limbs(),
                        destination,
                        scratch,
                    );
                } else {
                    newton_div_block::<WRITE_QUOTIENT, false, false>(
                        window,
                        den,
                        inverse,
                        destination,
                        scratch,
                    );
                }
            }
            end = start;
        }
        scratch.dummy_rem = reciprocal;
    }
}

/// Computes one quotient block and stores its exact remainder when required.
/// All reads of the dividend finish before its remainder is stored.
///
/// # Safety
/// Set `quo_len=num.len()-den.len()` and `block=reciprocal.len()-1`, both positive.
/// `quo_len<=block`. With `WRITE_QUOTIENT`, `quo` supplies
/// `quo_len` writable limbs disjoint from all inputs and scratch, and the complete
/// span is initialized. Otherwise `quo` is never used and may be null.
/// `EXACT` requires divisibility, quotient output and no consumer of the residue.
/// `GUARDED` requires quotient output and the full reciprocal with its guard:
/// its error is below `2+1/α²` for normalized `α=D/B^n`, and
/// `quo_len < block`. Its final residue is unused.
unsafe fn newton_div_block<const WRITE_QUOTIENT: bool, const EXACT: bool, const GUARDED: bool>(
    num: &mut [Limb],
    den: &[Limb],
    reciprocal: &[Limb],
    quo: *mut Limb,
    scratch: &mut DivScratch,
) {
    // SAFETY: the normalized driver supplies a nonempty divisor and
    // num.len()>den.len(); their difference is the positive quotient width.
    let (divisor_width, quotient_width) = unsafe {
        (
            NonZeroUsize::new_unchecked(den.len()),
            NonZeroUsize::new_unchecked(num.len().unchecked_sub(den.len())),
        )
    };
    let n = divisor_width.get();
    let quo_len = quotient_width.get();
    // Discarding d low inverse limbs gives error <1+epsilon/B^d.
    // For d>0 this remains below two, or below six in guarded mode;
    // the full-width case retains its preceding bound. No unused inverse
    // digits enter the last block's product.
    // SAFETY: quo_len<=block; guarded mode has quo_len<block. Thus the
    // requested leading one, quotient span and optional guard all exist.
    let inverse = unsafe {
        let width = quo_len.unchecked_add(if GUARDED { 2 } else { 1 });
        reciprocal.get_unchecked(reciprocal.len().unchecked_sub(width)..)
    };
    // Ordinary blocks use H=floor(U/B^n) and a t-limb inverse low part.
    // The final guarded block scales both prefixes by B for certification.
    // The lower reciprocal bound makes both estimates no greater than Q.
    // SAFETY: n > 0 and num.len() >= n + 1 by the block driver's contract.
    let high = unsafe { num.get_unchecked(n.unchecked_sub(usize::from(GUARDED))..) };
    let guard = Division::newton_quotient_estimate(high, inverse, scratch);
    // SAFETY: the estimator's exact suffix ends with quo_len digits and
    // one initialized high guard, a total of quo_len+1 limbs.
    // Any retained low product or certification guards precede that suffix.
    let estimate_start = unsafe {
        scratch
            .v_padded
            .len()
            .unchecked_sub(quo_len.unchecked_add(1))
    };
    // A certified guard makes Q0 exact; ambiguity requires residue correction.
    let write_remainder = !EXACT && !GUARDED;
    if EXACT {
        Division::newton_exact_correction(num, den, estimate_start, scratch);
    } else if write_remainder || guard > Limb::MAX - 5 {
        // The block reciprocal gives ε<2+1/α² and H/B^k<α. Its
        // scaled error is below 2/α+2α<=5. Integer floors differ
        // by at most five, so a guard <=B-6 cannot carry into Q.
        let remainder = Division::newton_remainder::<GUARDED>(num, den, estimate_start, scratch);

        if !remainder.is_empty() {
            // Both constructions give R<4D<B^(n+1); every limb above
            // position n is zero. Retain the sole high digit in a register.
            // SAFETY: a nonempty residue has at least n initialized limbs;
            // the cyclic zero is the only shorter encoding.
            let (digits, higher) = unsafe { remainder.split_at_mut_unchecked(n) };
            // Cyclic width n without a wrap has no high digit; its value
            // is zero. Linear products always retain their n+1 guard.
            let mut upper = higher.first().copied().unwrap_or(0);
            // SAFETY: all output modes retain quo_len initialized estimate
            // digits and their high guard, disjoint from the remainder.
            // Establish this fixed span before any correction mutates Q.
            let quotient_digits = unsafe {
                let end = estimate_start.unchecked_add(quo_len).unchecked_add(1);
                scratch.v_padded.get_unchecked_mut(estimate_start..end)
            };
            // Ordinary estimates require at most three iterations; the
            // decreasing residue establishes termination without a counter.
            while upper != 0 || InternalMpUint::cmp_limbs(digits, den) != Ordering::Less {
                if WRITE_QUOTIENT {
                    let carry = Addition::propagate_carry(quotient_digits, 1);
                    debug_assert_eq!(carry, 0, "the corrected quotient fits its block");
                }
                if GUARDED {
                    // The guard limits Q-Q0 to one. Its final residue is
                    // unused, so correcting Q needs no divisor subtraction.
                    break;
                }
                let borrow = Addition::sub_slice_in_place(digits, den);
                // SAFETY: R>=D proves upper>=borrow; the subtraction of
                // D cannot underflow the complete nonnegative residue.
                upper = unsafe { upper.unchecked_sub(borrow) };
            }
            // Only ordinary blocks consume the corrected n-limb residue
            // in a following block or in remainder publication.
            debug_assert!(
                GUARDED || (upper == 0 && InternalMpUint::cmp_limbs(digits, den) == Ordering::Less),
                "the reciprocal precision bounds the required corrections"
            );
        }
    }

    if WRITE_QUOTIENT {
        // SAFETY: v_padded retains the initialized quo_len output digits
        // after estimate_start. The estimator and correction bound Q by
        // the exact block quotient, so its extra high guard is zero.
        // The caller supplies quo_len aligned writable limbs disjoint from
        // scratch; copying the complete fixed span also writes leading zeros.
        unsafe {
            copy_nonoverlapping(scratch.v_padded.as_ptr().add(estimate_start), quo, quo_len);
        }
    }
}
