//! Knuth's Algorithm D, the basecase of the division tower.
//!
//! Recursive division terminates in these normalized limb kernels. A 3-by-2
//! reciprocal estimates each quotient digit using multiplication; its
//! construction performs one hardware division per prepared divisor.
//!
//! References:
//! - D. E. Knuth, *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*,
//!   3rd ed., Addison-Wesley, 1997, Section 4.3.1, Algorithm D (Division of nonnegative integers).

#![expect(
    unsafe_code,
    reason = "normalized division windows and write guards establish the bounds and initialization required by limb kernels"
)]

use core::{
    cmp::min,
    mem::MaybeUninit,
    num::NonZeroUsize,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use super::{
    ArchKernels, DIVISION_STACK_LIMBS, DivScratch, Division, DivisionMode, InternalMpUint, Limb,
    PreparedDivisor,
};

impl Division {
    /// Evaluates canonical limb slices with a nonzero divisor and a numerator
    /// at least as wide as the divisor. Ordinary modes return `()` after
    /// writing every requested output. `TruncatedPrefix` requires a dividend
    /// of `2*denominator.len()-2` limbs and returns a remainder certificate.
    /// `QuotientOnly<true>` requires divisibility of the complete slices;
    /// truncated prefixes do not inherit exactness.
    pub fn algorithm_d<Mode: DivisionMode>(
        u_limbs: &[Limb],
        v_limbs: &[Limb],
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) -> Mode::Outcome {
        if v_limbs.len() <= 2
            || (!Mode::CERTIFY_REMAINDER && v_limbs.len() <= 8)
            || u_limbs.len() < DIVISION_STACK_LIMBS
        {
            return Self::algorithm_d_stack::<Mode>(u_limbs, v_limbs, quotient_out, rem_out);
        }

        // SAFETY: the canonical nonzero divisor has a most-significant limb.
        let shift = unsafe { v_limbs.last().unwrap_unchecked() }.leading_zeros();
        Self::shift_limbs_left::<true>(u_limbs, shift, &mut scratch.u_norm);
        let v_norm = if shift == 0 {
            v_limbs
        } else {
            Self::shift_limbs_left::<false>(v_limbs, shift, &mut scratch.v_norm);
            scratch.v_norm.as_slice()
        };
        knuth_d_divide::<Mode>(
            scratch.u_norm.as_mut_slice(),
            v_norm,
            shift,
            quotient_out,
            rem_out,
        )
    }

    /// Completes basecase division using bounded stack normalization storage.
    ///
    /// The caller proves `u_limbs.len() < DIVISION_STACK_LIMBS`, unless the divisor
    /// has at most two limbs, or at most eight limbs in a complete output mode.
    /// These widths normalize during input reads. Other
    /// preconditions and the remainder certificate are those of `algorithm_d`.
    #[expect(
        clippy::too_many_lines,
        reason = "fixed-width admission and bounded normalization share the complete output and prefix-certificate contracts"
    )]
    pub fn algorithm_d_stack<Mode: DivisionMode>(
        u_limbs: &[Limb],
        v_limbs: &[Limb],
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
    ) -> Mode::Outcome {
        debug_assert!(
            v_limbs.last().is_some_and(|&limb| limb != 0),
            "internal division requires a canonical non-zero divisor"
        );
        debug_assert!(
            !Mode::CERTIFY_REMAINDER
                || (v_limbs
                    .len()
                    .checked_sub(1)
                    .and_then(|length| length.checked_mul(2))
                    == Some(u_limbs.len())),
            "truncated-prefix certification requires one divisor guard limb"
        );

        if v_limbs.len() == 1 {
            // SAFETY: v_limbs.len() == 1
            let v_single = unsafe { *v_limbs.get_unchecked(0) };
            let rem = if Mode::WRITE_QUOTIENT {
                Self::div_rem_1::<true>(u_limbs, v_single, quotient_out)
            } else {
                Self::div_rem_1::<false>(u_limbs, v_single, quotient_out)
            };
            if Mode::WRITE_REMAINDER {
                if rem == 0 {
                    rem_out.clear();
                } else {
                    *rem_out = InternalMpUint::from_limb(rem);
                }
            }
            return Mode::WRITTEN;
        }

        if let &[low, high] = v_limbs {
            Self::div_rem_2_unnormalized::<Mode::Primitive>(
                u_limbs,
                low,
                high,
                quotient_out,
                rem_out,
            );
            return Mode::WRITTEN;
        }

        if !Mode::CERTIFY_REMAINDER
            && let &[low, middle, high] = v_limbs
        {
            // SAFETY: the canonical three-limb divisor has a nonzero high limb.
            let divisor = (low, middle, unsafe { NonZeroUsize::new_unchecked(high) });
            Self::div_rem_3_unnormalized::<Mode::Primitive>(
                u_limbs,
                divisor,
                quotient_out,
                rem_out,
            );
            return Mode::WRITTEN;
        }

        if !Mode::CERTIFY_REMAINDER
            && let &[bottom, low, middle, high] = v_limbs
        {
            // SAFETY: the canonical four-limb divisor has a nonzero high limb.
            let divisor = (bottom, low, middle, unsafe {
                NonZeroUsize::new_unchecked(high)
            });
            Self::div_rem_4_unnormalized::<Mode::Primitive>(
                u_limbs,
                divisor,
                quotient_out,
                rem_out,
            );
            return Mode::WRITTEN;
        }

        if !Mode::CERTIFY_REMAINDER && (5..=8).contains(&v_limbs.len()) {
            Self::div_rem_fixed_unnormalized::<Mode::Primitive>(
                u_limbs,
                v_limbs,
                quotient_out,
                rem_out,
            );
            return Mode::WRITTEN;
        }

        let n_len = v_limbs.len();
        debug_assert!(
            u_limbs.len() >= n_len,
            "trivial division resolves shorter numerators"
        );
        // SAFETY: the nonzero divisor has a most-significant limb.
        let shift = unsafe { v_limbs.last().unwrap_unchecked() }.leading_zeros();
        // SAFETY: a materialized Limb slice has room in usize for a guard limb.
        let u_norm_len = unsafe { u_limbs.len().unchecked_add(1) };

        // Admission precedes the kernel; there is no partial scratch failure.
        debug_assert!(
            u_norm_len <= DIVISION_STACK_LIMBS,
            "stack division width exceeds its workspace"
        );
        let mut v_stack: [MaybeUninit<Limb>; DIVISION_STACK_LIMBS] =
            [MaybeUninit::uninit(); DIVISION_STACK_LIMBS];
        let mut u_stack: [MaybeUninit<Limb>; DIVISION_STACK_LIMBS] =
            [MaybeUninit::uninit(); DIVISION_STACK_LIMBS];
        let u_in_len = u_limbs.len();

        // SAFETY: `u_in_len < u_norm_len <= DIVISION_STACK_LIMBS`.
        // The `MaybeUninit` slice is in bounds, and the shift helper
        // initializes this entire prefix before it is read as `Limb`.
        let u_dst = unsafe { u_stack.get_unchecked_mut(..u_in_len) };
        let u_carry = Self::shift_limbs_left_uninit(u_dst, u_limbs, shift);
        // SAFETY: `u_norm_len = u_in_len + 1 <= DIVISION_STACK_LIMBS`; this writes
        // the sole limb not initialized by `shift_limbs_left_uninit`.
        unsafe {
            let _ = u_stack.get_unchecked_mut(u_in_len).write(u_carry);
        }

        // SAFETY: the shift and carry write initialized u_norm_len limbs
        // within u_stack; the slice does not outlive its owning array.
        let u_norm = unsafe { from_raw_parts_mut(u_stack.as_mut_ptr().cast::<Limb>(), u_norm_len) };
        let v_norm = if shift == 0 {
            v_limbs
        } else {
            // SAFETY: n_len <= u_norm_len <= DIVISION_STACK_LIMBS. The helper
            // initializes every element before a typed Limb slice is formed.
            let v_dst = unsafe { v_stack.get_unchecked_mut(..n_len) };
            let _ = Self::shift_limbs_left_uninit(v_dst, v_limbs, shift);
            // SAFETY: the preceding shift initialized all n_len limbs.
            unsafe { from_raw_parts(v_stack.as_ptr().cast::<Limb>(), n_len) }
        };
        knuth_d_divide::<Mode>(u_norm, v_norm, shift, quotient_out, rem_out)
    }

    /// Divides a normalized numerator in place, writing its requested outputs.
    ///
    /// `u_norm` is the normalized dividend with a high guard below the divisor's
    /// leading limb; `v_norm` has at least two limbs and its top bit set.
    /// The dividend is at least one limb longer than the divisor. Writes the
    /// complete quotient when `quo_out` has `u_norm.len()-v_norm.len()` limbs;
    /// an empty slice omits quotient stores. Writes up to `v_norm.len()`
    /// remainder limbs.
    pub fn knuth_d_divide_slice(
        u_norm: &mut [Limb],
        v_norm: &[Limb],
        quo_out: &mut [Limb],
        rem_out: &mut [Limb],
    ) {
        if let &[low, high] = v_norm {
            // A two-limb divisor uses only the reciprocal remainder; selecting
            // a multiply-subtract kernel would prepare unused architecture state.
            let inverse = Self::invert_pi1(high, low);
            // SAFETY: the normalized numerator contains at least three limbs.
            let last_digit = unsafe { u_norm.len().unchecked_sub(3) };
            Self::div_rem_2(u_norm, quo_out, last_digit, high, low, inverse);
        } else {
            PreparedDivisor::new(v_norm).divide(u_norm, v_norm, quo_out);
        }

        if !rem_out.is_empty() {
            let rem_len = min(rem_out.len(), v_norm.len());
            // SAFETY: rem_len <= rem_out.len() and rem_len <= v_norm.len() < u_norm.len().
            unsafe {
                rem_out
                    .get_unchecked_mut(..rem_len)
                    .copy_from_slice(u_norm.get_unchecked(..rem_len));
            }
        }
    }
}

/// Writes requested integer outputs from a normalized Algorithm D workspace.
#[expect(
    clippy::inline_always,
    reason = "Single hot call site per division; inlining exposes the quotient loop to register allocation and branch pruning."
)]
#[inline(always)]
fn knuth_d_divide<Mode: DivisionMode>(
    u_norm: &mut [Limb],
    v_norm: &[Limb],
    shift: u32,
    quotient_out: &mut InternalMpUint,
    rem_out: &mut InternalMpUint,
) -> Mode::Outcome {
    let n_len = v_norm.len();
    if Mode::WRITE_QUOTIENT {
        // SAFETY: u_norm contains at least n_len+1 initialized limbs.
        let maximum_digits = unsafe { u_norm.len().unchecked_sub(n_len) };
        // A zero high guard and a smaller following limb prove that the
        // leading quotient digit is zero. Exclude it before reserving the
        // output, so a fitting inline quotient never spills for that guard.
        // Truncation certification retains its dedicated input geometry.
        let leading_zero = !Mode::CERTIFY_REMAINDER && n_len >= 3 && maximum_digits > 1 && {
            // SAFETY: u_norm.len() >= n_len+1 >= 4 bounds its high pair;
            // the nonempty normalized divisor supplies its leading limb.
            unsafe {
                let high = u_norm.len().unchecked_sub(1);
                *u_norm.get_unchecked(high) == 0
                    && *u_norm.get_unchecked(high.unchecked_sub(1))
                        < *v_norm.last().unwrap_unchecked()
            }
        };
        // SAFETY: a discarded digit requires maximum_digits > 1, so the
        // shortened dividend remains at least one limb wider than v_norm.
        let (active_len, digits) = unsafe {
            (
                u_norm.len().unchecked_sub(usize::from(leading_zero)),
                maximum_digits.unchecked_sub(usize::from(leading_zero)),
            )
        };
        let quo_slice = quotient_out.ensure_capacity_set_len_get_limbs(digits);
        // SAFETY: active_len <= u_norm.len(). Removing a zero guard
        // required the next limb below the divisor's high limb, so
        // the shortened high divisor-width window remains below D.
        let active = unsafe { u_norm.get_unchecked_mut(..active_len) };
        if n_len >= 3 && (!Mode::WRITE_REMAINDER || Mode::CERTIFY_REMAINDER) {
            let prepared = PreparedDivisor::new(v_norm);
            let certificate = prepared.divide_quotient::<Mode::Quotient>(active, v_norm, quo_slice);
            if Mode::CERTIFY_REMAINDER && certificate == Mode::CERTIFIED {
                quotient_out.normalize();
                return Mode::CERTIFIED;
            }
        } else {
            Division::knuth_d_divide_slice(active, v_norm, quo_slice, &mut []);
        }
        quotient_out.normalize();
        if !Mode::WRITE_REMAINDER {
            return Mode::WRITTEN;
        }
    } else {
        Division::knuth_d_divide_slice(u_norm, v_norm, &mut [], &mut []);
    }

    if Mode::WRITE_REMAINDER {
        // SAFETY: `0..n_len` is within bounds of the normalized dividend `u_norm`.
        let rem_norm = unsafe { u_norm.get_unchecked(0..n_len) };
        if shift == 0 {
            rem_out.clone_from_slice(rem_norm);
        } else {
            let mut output = rem_out.prepare_limb_write(n_len);
            // SAFETY: 0<shift<LIMB_BITS, and the disjoint source contains
            // n_len initialized limbs. Preparation reserves that output
            // span; the kernel initializes every limb before commit.
            unsafe {
                let _ = ArchKernels::rshift_into_unchecked(
                    output.as_mut_ptr(),
                    rem_norm.as_ptr(),
                    n_len,
                    shift,
                );
                let _ = output.commit();
            }
            rem_out.normalize();
        }
    }
    Mode::WRITTEN
}
