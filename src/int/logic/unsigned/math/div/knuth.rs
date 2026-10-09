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
    slice::{from_raw_parts, from_raw_parts_mut},
};

use super::{
    Addition, ArchKernels, BURNIKEL_LONG_QUOTIENT_THRESHOLD, BURNIKEL_QUOTIENT_THRESHOLD,
    BURNIKEL_ZIEGLER_THRESHOLD, DIVISION_BASECASE_QUOTIENT_MAX_LIMBS, DIVISION_STACK_LIMBS,
    DivScratch, Division, InternalMpUint, Limb, PreparedDivisor,
};

impl Division {
    /// Attempts Algorithm D without constructing heap-backed division scratch.
    ///
    /// Returns `false` when the configured dispatch or normalization size
    /// requires the reusable scratch path. The divisor is nonzero. With
    /// `CHECK_TRIVIAL = false`, the numerator must have at least as many
    /// active limbs as the divisor; the quotient may still be zero or one.
    pub fn try_algorithm_d_unscratched<
        const WRITE_QUOTIENT: bool,
        const WRITE_REMAINDER: bool,
        const CHECK_TRIVIAL: bool,
    >(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
    ) -> bool {
        debug_assert!(
            !den_b.is_zero(),
            "internal division requires a non-zero divisor"
        );
        if CHECK_TRIVIAL
            && Self::trivial::<WRITE_QUOTIENT, WRITE_REMAINDER>(num_a, den_b, quotient_out, rem_out)
        {
            return true;
        }
        if CHECK_TRIVIAL
            && Self::power_of_two::<WRITE_QUOTIENT, WRITE_REMAINDER>(
                num_a,
                den_b,
                quotient_out,
                rem_out,
            )
        {
            return true;
        }
        let v_limbs = den_b.limbs();
        let u_limbs = num_a.limbs();
        // SAFETY: trivial division returned above, or CHECK_TRIVIAL=false
        // requires numerator width >= divisor width at the dispatch boundary.
        let extra = unsafe { u_limbs.len().unchecked_sub(v_limbs.len()) };
        let recursive_cutoff = if WRITE_REMAINDER {
            BURNIKEL_ZIEGLER_THRESHOLD
        } else if extra <= v_limbs.len() {
            BURNIKEL_QUOTIENT_THRESHOLD
        } else {
            BURNIKEL_LONG_QUOTIENT_THRESHOLD
        };
        if (v_limbs.len() >= recursive_cutoff && extra >= DIVISION_BASECASE_QUOTIENT_MAX_LIMBS)
            || (v_limbs.len() > 2 && u_limbs.len() >= DIVISION_STACK_LIMBS)
        {
            return false;
        }
        let _ = Self::algorithm_d_stack::<WRITE_QUOTIENT, WRITE_REMAINDER, false, false>(
            u_limbs,
            v_limbs,
            quotient_out,
            rem_out,
        );
        true
    }

    /// Evaluates canonical limb slices with a nonzero divisor and a numerator
    /// at least as wide as the divisor. `TRUNCATED_REMAINDER` requires
    /// both outputs and a dividend of `2*denominator.len()-2` limbs; it may
    /// return `true` without writing the remainder when the exact remainder
    /// exceeds the quotient. Otherwise it returns `false` with every requested
    /// output written. Division always completes; the return value is solely
    /// a mathematical certificate, used only by truncation.
    /// `KNOWN_EXACT` requires divisibility of these complete slices and
    /// quotient-only output; truncated prefixes do not inherit exactness.
    pub fn algorithm_d<
        const WRITE_QUOTIENT: bool,
        const WRITE_REMAINDER: bool,
        const TRUNCATED_REMAINDER: bool,
        const KNOWN_EXACT: bool,
    >(
        u_limbs: &[Limb],
        v_limbs: &[Limb],
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) -> bool {
        if v_limbs.len() <= 2 || u_limbs.len() < DIVISION_STACK_LIMBS {
            return Self::algorithm_d_stack::<
                WRITE_QUOTIENT,
                WRITE_REMAINDER,
                TRUNCATED_REMAINDER,
                KNOWN_EXACT,
            >(u_limbs, v_limbs, quotient_out, rem_out);
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
        knuth_d_divide::<WRITE_QUOTIENT, WRITE_REMAINDER, TRUNCATED_REMAINDER, KNOWN_EXACT>(
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
    /// has at most two limbs and normalizes during input reads. Other
    /// preconditions and the remainder certificate are those of `algorithm_d`.
    pub fn algorithm_d_stack<
        const WRITE_QUOTIENT: bool,
        const WRITE_REMAINDER: bool,
        const TRUNCATED_REMAINDER: bool,
        const KNOWN_EXACT: bool,
    >(
        u_limbs: &[Limb],
        v_limbs: &[Limb],
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
    ) -> bool {
        debug_assert!(
            v_limbs.last().is_some_and(|&limb| limb != 0),
            "internal division requires a canonical non-zero divisor"
        );
        debug_assert!(
            !KNOWN_EXACT || (WRITE_QUOTIENT && !WRITE_REMAINDER && !TRUNCATED_REMAINDER),
            "known divisibility supplies only the final quotient"
        );
        debug_assert!(
            !TRUNCATED_REMAINDER
                || (WRITE_QUOTIENT
                    && WRITE_REMAINDER
                    && v_limbs
                        .len()
                        .checked_sub(1)
                        .and_then(|length| length.checked_mul(2))
                        == Some(u_limbs.len())),
            "truncated-prefix certification requires one divisor guard limb"
        );

        if v_limbs.len() == 1 {
            // SAFETY: v_limbs.len() == 1
            let v_single = unsafe { *v_limbs.get_unchecked(0) };
            let rem = Self::div_rem_1::<WRITE_QUOTIENT>(u_limbs, v_single, quotient_out);
            if WRITE_REMAINDER {
                if rem == 0 {
                    rem_out.clear();
                } else {
                    *rem_out = InternalMpUint::from_limb(rem);
                }
            }
            return false;
        }

        if let &[low, high] = v_limbs {
            Self::div_rem_2_unnormalized::<WRITE_QUOTIENT, WRITE_REMAINDER>(
                u_limbs,
                low,
                high,
                quotient_out,
                rem_out,
            );
            return false;
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
        knuth_d_divide::<WRITE_QUOTIENT, WRITE_REMAINDER, TRUNCATED_REMAINDER, KNOWN_EXACT>(
            u_norm,
            v_norm,
            shift,
            quotient_out,
            rem_out,
        )
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
fn knuth_d_divide<
    const WRITE_QUOTIENT: bool,
    const WRITE_REMAINDER: bool,
    const TRUNCATED_REMAINDER: bool,
    const KNOWN_EXACT: bool,
>(
    u_norm: &mut [Limb],
    v_norm: &[Limb],
    shift: u32,
    quotient_out: &mut InternalMpUint,
    rem_out: &mut InternalMpUint,
) -> bool {
    let n_len = v_norm.len();
    if WRITE_QUOTIENT {
        // SAFETY: u_norm contains at least n_len+1 initialized limbs.
        let maximum_digits = unsafe { u_norm.len().unchecked_sub(n_len) };
        // A zero high guard and a smaller following limb prove that the
        // leading quotient digit is zero. Exclude it before reserving the
        // output, so a fitting inline quotient never spills for that guard.
        // Truncation certification retains its dedicated input geometry.
        let leading_zero = !TRUNCATED_REMAINDER && n_len >= 3 && maximum_digits > 1 && {
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
        if n_len >= 3 && (!WRITE_REMAINDER || TRUNCATED_REMAINDER) {
            let prepared = PreparedDivisor::new(v_norm);
            if prepared.divide_quotient::<WRITE_REMAINDER, TRUNCATED_REMAINDER, KNOWN_EXACT>(
                active, v_norm, quo_slice,
            ) {
                quotient_out.normalize();
                return true;
            }
        } else {
            Division::knuth_d_divide_slice(active, v_norm, quo_slice, &mut []);
        }
        quotient_out.normalize();
        if !WRITE_REMAINDER {
            return false;
        }
    } else {
        Division::knuth_d_divide_slice(u_norm, v_norm, &mut [], &mut []);
    }

    if WRITE_REMAINDER {
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
    false
}

impl Division {
    /// Computes one quotient digit and its n-limb remainder.
    /// The reciprocal supplies the leading two remainder limbs; only the low
    /// divisor prefix needs multiply-subtract. Equality of the leading pairs
    /// gives digit B-1 and its leading remainder directly by addition.
    /// The low n-1 remainder limbs replace `window`; its high limb remains
    /// in the caller's recurrence until the complete remainder is requested.
    /// The normalized divisor has n >= 3 limbs. The window has n initialized
    /// limbs, `u2` supplies its high guard, and the preceding remainder is
    /// below the divisor. The leading pair and reciprocal match this divisor.
    #[expect(
        clippy::inline_always,
        reason = "Pinned Algorithm D measurements and emitted assembly show that outlining this per-digit step reloads cached divisor state and adds calls inside both quotient loops."
    )]
    #[inline(always)]
    pub fn knuth_d_step(
        window: &mut [Limb],
        divisor: &[Limb],
        u2: Limb,
        d1: Limb,
        d0: Limb,
        inverse: Limb,
        sub_mul: unsafe fn(*mut Limb, *const Limb, usize, Limb) -> (Limb, Limb),
    ) -> (Limb, Limb) {
        // SAFETY: full and short division retain at least three initialized
        // divisor limbs. The fixed leading triple leaves n-3 low digits.
        let (below_three, _) = unsafe { divisor.split_last_chunk::<3>().unwrap_unchecked() };
        // SAFETY: adding the triple reconstructs the materialized width;
        // its first digit makes the multiply-subtract prefix n-2 nonempty.
        let (lower_width, n) = unsafe {
            (
                below_three.len().unchecked_add(1),
                below_three.len().unchecked_add(3),
            )
        };
        // SAFETY: the caller supplies n initialized window limbs and its high
        // guard; the two high window digits are within that complete span.
        let (u1, u0) = unsafe {
            (
                *window.get_unchecked(n.unchecked_sub(1)),
                *window.get_unchecked(n.unchecked_sub(2)),
            )
        };
        debug_assert!(
            (u2, u1) <= (d1, d0),
            "the preceding remainder is below the normalized divisor"
        );
        if (u2, u1) == (d1, d0) {
            // Let U = R*B+a with 0 <= a < B and R < D. Equal leading
            // pairs give delta = D-R < B^(n-2). Normalization proves
            // delta*B < B^(n-1) <= D, so U-(B-1)*D = D-delta*B+a > 0.
            // U < B*D therefore makes B-1 exact, with no add-back.
            // The leading triple minus (B-1)*(d1*B+d0) is d1*B+d0+u0.
            // Its high digit may be B before the low-prefix borrow, so both
            // high updates use reduction modulo B rather than nonoverflowing arithmetic.
            let (low, leading_carry) = d0.overflowing_add(u0);
            let high = d1.wrapping_add(Limb::from(leading_carry));
            // SAFETY: the disjoint initialized low prefixes have n-2>0
            // limbs. Product carry < B-1 plus binary borrow fits one limb.
            let correction = unsafe {
                let (carry, borrow) = sub_mul(
                    window.as_mut_ptr(),
                    divisor.as_ptr(),
                    lower_width,
                    Limb::MAX,
                );
                carry.unchecked_add(borrow)
            };
            let (next_low, borrow) = low.overflowing_sub(correction);
            let upper = high.wrapping_sub(Limb::from(borrow));
            // SAFETY: n>=3 and the n-limb window contains this initialized slot.
            unsafe {
                *window.get_unchecked_mut(n.unchecked_sub(2)) = next_low;
            }
            return (Limb::MAX, upper);
        }
        // Strict inequality proves the reciprocal quotient fits one limb.
        let (mut quotient, high, low) = Self::udiv_qr_3by2(u2, u1, u0, d1, d0, inverse);
        // U = H*B^(n-2) + L and D = T*B^(n-2) + K. The
        // reciprocal already computed H = quotient*T + (high*B + low).
        // Only L - quotient*K remains; recomputing quotient*T would repeat
        // two limb products and their carry propagation.
        // SAFETY: n > 2; the low n-2 initialized limbs of each disjoint
        // owner satisfy the selected kernel's pointer contract. A scalar
        // product carry is below quotient (or zero for quotient zero), and
        // borrow is binary, so their sum fits Limb on every target.
        let correction = unsafe {
            let (carry, borrow) =
                sub_mul(window.as_mut_ptr(), divisor.as_ptr(), lower_width, quotient);
            carry.unchecked_add(borrow)
        };
        let (next_low, low_borrow) = low.overflowing_sub(correction);
        let (mut upper, negative) = high.overflowing_sub(Limb::from(low_borrow));
        // SAFETY: n > 2 and window has n initialized limbs. This slot
        // follows the n-2 outputs written by sub_mul.
        unsafe {
            *window.get_unchecked_mut(n.unchecked_sub(2)) = next_low;
        }
        if negative {
            // SAFETY: a zero digit subtracts nothing and cannot underflow.
            // The negative residue therefore proves quotient > 0.
            quotient = unsafe { quotient.unchecked_sub(1) };
            // A normalized 3/2 estimate is at most one too large. Add-back
            // restores the low n-1 limbs and the retained upper limb modulo
            // B^n; the discarded carry cancels the negative high guard.
            // SAFETY: n >= 3 bounds both initialized n-1-limb prefixes.
            let carry = unsafe {
                let width = n.unchecked_sub(1);
                Addition::add_slice_in_place(
                    window.get_unchecked_mut(..width),
                    divisor.get_unchecked(..width),
                )
            };
            upper = upper.wrapping_add(d1).wrapping_add(carry);
        }
        (quotient, upper)
    }
}
