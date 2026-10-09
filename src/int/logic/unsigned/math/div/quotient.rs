//! Quotient-only division through operand truncation.
//!
//! A short quotient permits low operand limbs to be omitted while computing
//! its estimate. A remainder certificate or one bounded product comparison
//! determines whether the estimate requires a decrement.
//!
//! Write `u = u' B^k + a` and `v = v' B^k + b` with `0 <= a, b < B^k` and
//! `B = 2^LIMB_BITS`, and let `u' = q' v' + r'` be the truncated division. Then
//!
//! ```text
//! u = q' v + D,   with D = r' B^k + a - q' b
//! ```
//!
//! so `floor(u / v) = q' + floor(D / v)`. Retaining `qn + 1` divisor limbs,
//! where `qn` is the quotient limb count, forces `q' < B^qn` and
//! `v >= B^(qn + k) > q' B^k`, which bounds `D` into `(-v, v)`. The true
//! quotient is therefore `q'` or `q' - 1`.
//!
//! The sign of `D` decides which quotient holds. `r' >= q'` certifies
//! `D >= q'*(B^k-b)+a >= 0`. The short-division kernel can prove this from
//! its retained high remainder limbs before subtracting the omitted product
//! triangle; it then returns a certificate without materializing `r'`.
//! An ambiguous certificate finishes the exact prefix remainder in the same
//! workspace. A remainder below `q'` compares only `q'*b` against
//! `r'*B^k+a`, retaining the one-unit correction bound for exact multiples
//! and adjacent boundary values. Larger prefixes use a normalized low quotient
//! guard to avoid constructing that remainder; [`Division::guarded_quotient`]
//! proves its certificate and omitted-product fallback.
//!
//! References:
//! - A. H. Karp and P. Markstein, "High-Precision Division and Square Root", ACM Transactions
//!   on Mathematical Software, Vol. 23, No. 4, pp. 561–589, Dec. 1997. DOI: 10.1145/279232.279237.
//! - R. P. Brent and P. Zimmermann, *Modern Computer Arithmetic*, Cambridge University Press,
//!   2011, Section 1.4.6 "Only quotient or remainder wanted".
//!   DOI: 10.1017/CBO9780511921698.

#![expect(
    unsafe_code,
    reason = "truncation admission bounds source slices and exact quotient estimates validate scalar correction kernels"
)]

use core::{cmp::Ordering, num::NonZeroUsize};

use super::{
    BURNIKEL_LONG_QUOTIENT_THRESHOLD, BURNIKEL_QUOTIENT_THRESHOLD,
    DIVISION_BASECASE_QUOTIENT_MAX_LIMBS, DIVISION_STACK_LIMBS, DIVISION_TRUNCATION_RATIO,
    DivScratch, Division, DoubleLimb, InternalMpUint, LIMB_BITS, Limb, Multiplication,
    NEWTON_QUOTIENT_THRESHOLD,
};

impl Division {
    /// Computes only `num_a / den_b` into reusable output storage.
    ///
    /// The divisor must be nonzero. With `CHECK_TRUNCATED = false`, the caller
    /// must already reject trivial and truncated forms.
    /// Algorithm D omits remainder output. Recursive tiers retain exact
    /// intermediate remainders; quotient guards certify the final block.
    /// `KNOWN_EXACT`
    /// requires `num_a` to be a multiple of `den_b`; only the final quotient
    /// may omit the residue products justified by that invariant.
    pub fn div_into<const CHECK_TRUNCATED: bool, const KNOWN_EXACT: bool>(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        debug_assert!(!den_b.is_zero(), "division requires a nonzero divisor");
        if CHECK_TRUNCATED
            && truncated_quotient_into::<KNOWN_EXACT>(num_a, den_b, quotient_out, scratch)
        {
            return;
        }
        let denominator_len = den_b.limbs().len();
        // SAFETY: CHECK_TRUNCATED rejects trivial division above; otherwise
        // its caller already did so. The canonical numerator is therefore
        // at least as wide as the nonzero divisor.
        let extra_limbs = unsafe { num_a.limbs().len().unchecked_sub(denominator_len) };
        // A balanced short quotient omits ~n²/2 low product terms. Its
        // crossover differs from combined division, which needs the full
        // remainder. The size ratio below selects only this output policy.
        let recursive_cutoff = if extra_limbs <= denominator_len {
            BURNIKEL_QUOTIENT_THRESHOLD
        } else {
            BURNIKEL_LONG_QUOTIENT_THRESHOLD
        };
        if extra_limbs < DIVISION_BASECASE_QUOTIENT_MAX_LIMBS
            || denominator_len < recursive_cutoff
            || denominator_len <= 2
        {
            let mut unused_rem = InternalMpUint::zero();
            let _ = Self::algorithm_d::<true, false, false, KNOWN_EXACT>(
                num_a.limbs(),
                den_b.limbs(),
                quotient_out,
                &mut unused_rem,
                scratch,
            );
            return;
        }
        // Newton retains its reciprocal in dummy_rem. An unused public output
        // must not remove that reusable allocation or overwrite its new owner.
        let mut unused_rem = InternalMpUint::zero();
        if denominator_len >= NEWTON_QUOTIENT_THRESHOLD {
            Self::newton::<true, false, KNOWN_EXACT>(
                num_a,
                den_b,
                quotient_out,
                &mut unused_rem,
                scratch,
            );
        } else {
            Self::guarded_quotient::<KNOWN_EXACT>(
                num_a.limbs(),
                den_b.limbs(),
                quotient_out,
                0,
                scratch,
            );
        }
    }

    /// Computes `num_a / den_b` from the leading limbs of both operands.
    ///
    /// Returns `false` when the operand shape makes truncation inapplicable or
    /// unprofitable, leaving `quotient_out` untouched; the caller must then run
    /// the full division engine. Returns `true` with the exact floor quotient
    /// written to `quotient_out` otherwise. The divisor must be nonzero.
    /// With `EXACT`, the caller proves divisibility. Writing `a = q*d` and
    /// truncating at radix power S gives `a' = q*d' + floor(q*(d mod S)/S)`.
    /// The last term is below q. Keeping one guard limb makes d' > q, so
    /// truncated division gives q exactly, without a full-width product check.
    /// Prefixes that fit the stack need no division scratch. Larger prefixes
    /// and ambiguous residual products acquire their workspace before entry.
    pub fn truncated_quotient<const EXACT: bool>(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
    ) -> bool {
        debug_assert!(
            !den_b.is_zero(),
            "the public division boundary validates its divisor"
        );
        let mut rem = InternalMpUint::zero();
        if Self::trivial::<true, false>(num_a, den_b, quotient_out, &mut rem) {
            return true;
        }
        if Self::power_of_two::<true, false>(num_a, den_b, quotient_out, &mut rem) {
            return true;
        }
        let u_limbs = num_a.limbs();
        let v_limbs = den_b.limbs();
        let Some(split) = truncation_split(u_limbs.len(), v_limbs.len()) else {
            return false;
        };
        // SAFETY: truncation_split admits only 0 < split < v_limbs.len(),
        // with v_limbs.len() <= u_limbs.len(). Both suffixes are initialized
        // immutable views, and the retained divisor is nonempty.
        let (num_head, den_head) = unsafe {
            (
                u_limbs.get_unchecked(split.get()..),
                v_limbs.get_unchecked(split.get()..),
            )
        };
        if num_head.len() < DIVISION_STACK_LIMBS && den_head.len() < BURNIKEL_QUOTIENT_THRESHOLD {
            if EXACT {
                // Exactness belongs to the complete operands, not their prefixes.
                let _ = Self::algorithm_d_stack::<true, false, false, false>(
                    num_head,
                    den_head,
                    quotient_out,
                    &mut rem,
                );
                return true;
            }
            if Self::algorithm_d_stack::<true, true, true, false>(
                num_head,
                den_head,
                quotient_out,
                &mut rem,
            ) || rem >= *quotient_out
            {
                return true;
            }
            if quotient_out.limbs().len() == 1 {
                correct_truncated_scalar_quotient(u_limbs, v_limbs, quotient_out, &rem, split);
            } else {
                correct_truncated_quotient(
                    u_limbs,
                    v_limbs,
                    quotient_out,
                    &rem,
                    split.get(),
                    &mut DivScratch::default(),
                );
            }
        } else {
            divide_truncated::<EXACT>(
                u_limbs,
                v_limbs,
                quotient_out,
                split,
                &mut DivScratch::default(),
            );
        }
        true
    }
}

/// Writes an exact floor quotient through operand truncation with reusable scratch.
///
/// The divisor is nonzero. Rejected shapes leave the quotient untouched;
/// admitted prefixes retain one divisor guard limb. With `EXACT`, the
/// caller proves divisibility of the complete operands.
pub fn truncated_quotient_into<const EXACT: bool>(
    num_a: &InternalMpUint,
    den_b: &InternalMpUint,
    quotient_out: &mut InternalMpUint,
    scratch: &mut DivScratch,
) -> bool {
    debug_assert!(
        !den_b.is_zero(),
        "the quotient dispatcher validates its divisor"
    );
    let mut unused_rem = InternalMpUint::zero();
    if Division::trivial::<true, false>(num_a, den_b, quotient_out, &mut unused_rem) {
        return true;
    }
    if Division::power_of_two::<true, false>(num_a, den_b, quotient_out, &mut unused_rem) {
        return true;
    }
    let u_limbs = num_a.limbs();
    let v_limbs = den_b.limbs();
    let Some(split) = truncation_split(u_limbs.len(), v_limbs.len()) else {
        return false;
    };
    divide_truncated::<EXACT>(u_limbs, v_limbs, quotient_out, split, scratch);
    true
}

/// Selects a profitable prefix after trivial quotients have been resolved.
/// `num_len>=den_len>0` are the lengths of materialized limb slices.
const fn truncation_split(num_len: usize, den_len: usize) -> Option<NonZeroUsize> {
    // One guard is sufficient for both exact and general quotients:
    // den_prime >= B^(extra_limbs+1) > quot bounds the omitted low product.
    // SAFETY: both callers rejected trivial division, proving num_len>=den_len.
    // A materialized Limb slice has len<=isize::MAX/size_of::<Limb>(), with
    // at least two bytes per limb, so extra_limbs+2 fits every pointer width.
    let (extra_limbs, den_prime_len) = unsafe {
        let extra_limbs = num_len.unchecked_sub(den_len);
        (extra_limbs, extra_limbs.unchecked_add(2))
    };
    if den_prime_len >= den_len {
        return None;
    }
    // A Newton prefix can admit the half-width quotient boundary before
    // adding the guard. Where the full division recurses, truncation
    // replaces its leading block's repair product, against the discarded
    // divisor limbs, by the rare ambiguous-residual product, so any
    // shorter prefix pays. Below that crossover, short division already
    // skips the lower triangle. Its empirical admission keeps at most
    // half the divisor to amortize normalization and certification.
    let admitted = if den_prime_len >= NEWTON_QUOTIENT_THRESHOLD {
        extra_limbs <= den_len.div_euclid(DIVISION_TRUNCATION_RATIO)
    } else if den_len >= BURNIKEL_LONG_QUOTIENT_THRESHOLD {
        true
    } else {
        den_prime_len <= den_len.div_euclid(DIVISION_TRUNCATION_RATIO)
    };
    if !admitted {
        return None;
    }
    // SAFETY: the shape gate established den_prime_len < den_len.
    NonZeroUsize::new(unsafe { den_len.unchecked_sub(den_prime_len) })
}

/// Completes an admitted truncation with all reusable storage supplied.
/// Admission proves `0 < split < divisor.len() <= numerator.len()` and
/// retains one divisor guard beyond the maximum quotient width.
fn divide_truncated<const EXACT: bool>(
    numerator: &[Limb],
    divisor: &[Limb],
    quotient_out: &mut InternalMpUint,
    split: NonZeroUsize,
    scratch: &mut DivScratch,
) {
    // SAFETY: the caller obtained split from truncation_split, proving
    // 0 < split < divisor.len() <= numerator.len(). The immutable
    // suffixes retain each operand's initialized nonzero leading limb.
    let (num_head, den_head) = unsafe {
        (
            numerator.get_unchecked(split.get()..),
            divisor.get_unchecked(split.get()..),
        )
    };
    let den_prime_len = den_head.len();
    if den_prime_len >= BURNIKEL_QUOTIENT_THRESHOLD {
        Division::guarded_quotient::<EXACT>(numerator, divisor, quotient_out, split.get(), scratch);
        return;
    }
    let mut rem = InternalMpUint::zero();
    // The prefix has 2*quot_len numerator limbs and quot_len+1 divisor
    // limbs. The short kernel certifies rem >= quot from its retained
    // high pair or completes the same rem. Divisibility of the original
    // operands gives an exact quotient, without requiring an exact prefix.
    if EXACT {
        let _ = Division::algorithm_d::<true, false, false, false>(
            num_head,
            den_head,
            quotient_out,
            &mut rem,
            scratch,
        );
    } else if Division::algorithm_d::<true, true, true, false>(
        num_head,
        den_head,
        quotient_out,
        &mut rem,
        scratch,
    ) {
        return;
    }
    if !EXACT && rem < *quotient_out {
        if quotient_out.limbs().len() == 1 {
            correct_truncated_scalar_quotient(numerator, divisor, quotient_out, &rem, split);
        } else {
            correct_truncated_quotient(
                numerator,
                divisor,
                quotient_out,
                &rem,
                split.get(),
                scratch,
            );
        }
    }
}

/// Compares the omitted product with `rem*B^split + numerator_low`.
/// The prefix remainder is below the quotient, whose overestimate is at
/// most one; a strict product excess therefore requires one decrement.
fn correct_truncated_quotient(
    u_limbs: &[Limb],
    v_limbs: &[Limb],
    quotient_out: &mut InternalMpUint,
    rem: &InternalMpUint,
    split: usize,
    scratch: &mut DivScratch,
) {
    // U-Q*D = rem*B^split + U_low - Q*D_low. The retained
    // high divisor product is already accounted for by rem. Multiply
    // the borrowed low prefix directly into reusable product storage.
    // SAFETY: split < den_len <= num_len proves this initialized
    // divisor prefix exists and both materialized lengths fit their sum.
    let (low, product_len) = unsafe {
        (
            v_limbs.get_unchecked(..split),
            quotient_out.limbs().len().unchecked_add(split),
        )
    };
    scratch.q_den_low.reset_with_capacity(product_len);
    // SAFETY: U>=D gives a nonempty quotient, and split>0 gives a nonempty
    // divisor prefix. Their owners and the reserved output are disjoint.
    // The kernel initializes the complete product before set_len exposes it.
    unsafe {
        let _ = Multiplication::mul_nonempty_distinct_into_uninit(
            quotient_out.limbs(),
            low,
            scratch
                .q_den_low
                .spare_capacity_mut()
                .get_unchecked_mut(..product_len),
            &mut scratch.mul_scratch,
        );
        scratch.q_den_low.set_len(product_len);
    }
    let mut product_limbs = scratch.q_den_low.as_slice();
    while let Some((&0, prefix)) = product_limbs.split_last() {
        product_limbs = prefix;
    }
    let overlap = product_limbs.len().min(split);
    // SAFETY: overlap <= product.len(); if shorter than split, the
    // product has no high part. Otherwise overlap == split.
    let product_high = unsafe { product_limbs.get_unchecked(overlap..) };
    let overshoots = match InternalMpUint::cmp_limbs(product_high, rem.limbs()) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => {
            // SAFETY: overlap <= split < num_len and overlap <=
            // product.len(). Absent product limbs are zero. After
            // their input counterparts are checked, equal-width
            // reverse iteration compares the two low prefixes.
            unsafe {
                u_limbs
                    .get_unchecked(overlap..split)
                    .iter()
                    .all(|&limb| limb == 0)
                    && product_limbs
                        .get_unchecked(..overlap)
                        .iter()
                        .rev()
                        .cmp(u_limbs.get_unchecked(..overlap).iter().rev())
                        == Ordering::Greater
            }
        }
    };
    if overshoots {
        quotient_out.decrement();
    }
}

/// Corrects a one-limb truncated estimate without storing its low product.
///
/// For S=B^split, a=U mod S, b=D mod S and 0<=r<q<B, write
/// q*b=h*S+p. Limb subtraction gives p-a=delta-beta*S, with
/// 0<=delta<S and beta in {0,1}. Thus q*b-(r*S+a) equals
/// (h-r-beta)*S+delta. Its sign needs only h, beta and delta!=0.
/// Admission supplies `0<split<divisor.len()<=numerator.len()`; the
/// estimate is positive, occupies one limb, and is at most one too large.
/// Trivial division has already established U>=D, so a decremented
/// estimate remains positive and needs no carry propagation or normalization.
#[expect(
    clippy::as_conversions,
    reason = "DoubleLimb embeds native Limb operands exactly; the product low limb is reduction modulo B and the shifted carry fits Limb on 16-, 32-, and 64-bit targets"
)]
#[cfg_attr(
    not(target_pointer_width = "16"),
    expect(
        clippy::cast_possible_truncation,
        reason = "the product low limb is its residue modulo B; the shifted carry is below the one-limb quotient"
    )
)]
fn correct_truncated_scalar_quotient(
    numerator: &[Limb],
    divisor: &[Limb],
    quotient: &mut InternalMpUint,
    remainder: &InternalMpUint,
    split: NonZeroUsize,
) {
    // SAFETY: the caller established one quotient limb, r<q and the
    // admitted split within both initialized operand slices.
    let (digit, numerator_low, divisor_low) = unsafe {
        (
            *quotient.limbs().get_unchecked(0),
            numerator.get_unchecked(..split.get()),
            divisor.get_unchecked(..split.get()),
        )
    };
    let residue = remainder.limbs().first().copied().unwrap_or(0);
    let mut carry: Limb = 0;
    let mut borrow: Limb = 0;
    let mut nonzero = false;
    for (&num, &den) in numerator_low.iter().zip(divisor_low) {
        // SAFETY: den<B, digit<B and carry<digit. Their product plus
        // carry is below B^2, fitting DoubleLimb on every pointer width.
        let product = unsafe {
            (den as DoubleLimb)
                .unchecked_mul(digit as DoubleLimb)
                .unchecked_add(carry as DoubleLimb)
        };
        carry = (product >> LIMB_BITS) as Limb;
        let (low_difference, underflow) = (product as Limb).overflowing_sub(num);
        let (difference, borrowed) = low_difference.overflowing_sub(borrow);
        borrow = Limb::from(underflow | borrowed);
        nonzero |= difference != 0;
    }
    // SAFETY: residue<digit<=Limb::MAX and borrow<=1 prove this sum fits.
    let threshold = unsafe { residue.unchecked_add(borrow) };
    if carry > threshold || (carry == threshold && nonzero) {
        // SAFETY: the caller supplied one initialized quotient limb.
        // The strict excess proves it is one above floor(U/D)>=1;
        // digit>=2 and digit-1 therefore preserves its canonical length.
        unsafe {
            *quotient.limbs_mut().get_unchecked_mut(0) = digit.unchecked_sub(1);
        }
    }
}
