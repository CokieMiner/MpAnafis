//! Residue reconstruction and partial products for Newton division.
//!
//! For an n-limb divisor D and a k-limb multiplier Q, set s=n+1-k.
//! With D=D0+D1*B^s, reduction modulo B^(n+1) gives
//! Q*D=Q*D0+(Q*D1 mod B^k)*B^s. The full rectangular product has
//! exactly n+1 limbs; the remaining low product has only k limbs.

#![expect(
    unsafe_code,
    reason = "Newton width bounds establish disjoint low-product partitions, initialized outputs and implicit leading correction digits"
)]

use core::num::NonZeroUsize;

#[cfg(not(target_pointer_width = "16"))]
use super::Multiplication;
use super::{
    Addition, ArchKernels, DivScratch, Division, HighProduct, Limb, LowProduct, MulScratch,
    ScratchBuffer,
};

impl Division {
    /// Replaces U by its exact nonnegative residue U-Q0*D before correction.
    ///
    /// D is normalized and nonempty; U has at least `D.len()+1` limbs. The
    /// estimator in `v_padded[start..]` is a lower quotient bound with error
    /// at most three and width at most `D.len()`. Thus `R<4D<B^(D.len()+1)`, and
    /// a linear low product or its Mersenne residue determines all of R.
    /// A cyclic zero returns an empty span. Unless `DISCARD_REMAINDER`, its
    /// low divisor-width span is cleared for consumption by the next block.
    pub fn newton_remainder<'numerator, const DISCARD_REMAINDER: bool>(
        num: &'numerator mut [Limb],
        den: &[Limb],
        start: usize,
        scratch: &mut DivScratch,
    ) -> &'numerator mut [Limb] {
        // SAFETY: normalized block admission supplies a nonempty divisor;
        // its positive width determines every residue and product partition.
        let divisor_width = unsafe { NonZeroUsize::new_unchecked(den.len()) };
        let n = divisor_width.get();
        // SAFETY: the estimator retains an initialized quotient after start
        // and one proven zero high guard. Discard that guard directly; any
        // further leading zeros preserve the quotient value and lower bound.
        let mut estimate = unsafe {
            let guard = scratch.v_padded.len().unchecked_sub(1);
            scratch.v_padded.get_unchecked(start..guard)
        };
        while let Some((&0, prefix)) = estimate.split_last() {
            estimate = prefix;
        }
        // SAFETY: den is a materialized Limb slice and num has at least n+1
        // limbs, so the residue width fits usize on every pointer width.
        let check_len = unsafe { n.unchecked_add(1) };
        if estimate.is_empty() {
            // Q0=0 implies R=U: no product, subtraction or scratch is needed.
            // SAFETY: the block contains the complete n+1-limb residue prefix.
            return unsafe { num.get_unchecked_mut(..check_len) };
        }
        if let [digit] = estimate {
            // A scalar estimate permits product subtraction in one pass,
            // retaining its high product and low subtraction borrow in registers.
            // SAFETY: num contains n+1 initialized limbs, disjoint from the
            // n-limb divisor. Both the low destination and its guard exist.
            unsafe {
                let remainder = num.get_unchecked_mut(..check_len);
                let (lower, guard) = remainder.split_at_mut_unchecked(n);
                let (carry, borrow) = ArchKernels::sub_mul_limbs_unchecked(
                    lower.as_mut_ptr(),
                    den.as_ptr(),
                    n,
                    *digit,
                );
                // D<B^n gives carry<digit; borrow<=1 makes their sum at most
                // digit<=Limb::MAX. The high subtraction is modular: R<B^(n+1)
                // proves that any borrow is absorbed by discarded input limbs.
                let high = guard.get_unchecked_mut(0);
                *high = high.wrapping_sub(carry.unchecked_add(borrow));
                return remainder;
            }
        }
        #[cfg(not(target_pointer_width = "16"))]
        let wrapped = Multiplication::try_mul_mod_bnm1::<false, false>(
            estimate,
            den,
            n,
            &mut scratch.q_den_low,
            &mut scratch.mul_scratch,
        );
        #[cfg(target_pointer_width = "16")]
        let wrapped = false;
        if wrapped {
            // Modulo B^w-1 gives the cyclic residue; R modulo eight recovers
            // its at most three wraps since B^w-1=-1 modulo eight. The
            // alternate zero requires the additional signed count of minus one.
            // SAFETY: the block and nonempty estimate contain their low limbs.
            // Wrapping subtraction and multiplication preserve the low three bits.
            let residue_mod8 = unsafe {
                num.get_unchecked(0).wrapping_sub(
                    estimate
                        .get_unchecked(0)
                        .wrapping_mul(*den.get_unchecked(0)),
                ) & 7
            };
            // SAFETY: scalar estimates returned above, so 2<=n<=w. Cyclic
            // admission gives w<full_len<=num.len()<=2n<=2w, leaving 1..=w
            // initialized tail limbs disjoint from the fold and product.
            let (fold, tail) = unsafe { num.split_at_mut_unchecked(scratch.q_den_low.len()) };
            let width =
                newton_wrapped_remainder(fold, tail, residue_mod8, scratch.q_den_low.as_slice());
            if !DISCARD_REMAINDER && width == 0 {
                // SAFETY: cyclic zero denotes R=0. The dividend contains n
                // initialized limbs to clear for the following block.
                unsafe {
                    num.get_unchecked_mut(..n).fill(0);
                }
            }
            // SAFETY: reconstruction returns zero or an initialized prefix
            // no wider than the original dividend window.
            return unsafe { num.get_unchecked_mut(..width) };
        }

        scratch.q_den_low.reset_with_capacity(check_len);
        // SAFETY: reservation supplies n+1 spare limbs, disjoint from
        // the independently owned operands. The rectangular product
        // initializes every residue limb before its length is committed.
        let low_product = unsafe {
            let _ = LowProduct::mul_with_guard(
                den,
                estimate,
                scratch
                    .q_den_low
                    .spare_capacity_mut()
                    .get_unchecked_mut(..check_len),
                &mut scratch.newton_c_buf,
                &mut scratch.den_pad,
                &mut scratch.mul_scratch,
            );
            scratch.q_den_low.set_len(check_len);
            scratch.q_den_low.as_slice()
        };
        // R<B^(n+1) makes subtraction modulo B^(n+1) exact even if the
        // truncated product's subtraction borrows from the discarded limbs.
        // SAFETY: num has n+1 initialized limbs disjoint from low_product.
        let remainder = unsafe { num.get_unchecked_mut(..check_len) };
        let _ = Addition::sub_slice_in_place(remainder, low_product);
        remainder
    }

    /// Forms the correction `floor(H*W/B^(k+1))` for `W=B^k+W0`.
    ///
    /// H is nonempty with at most k+1 limbs; a k+1-limb H has leading one.
    /// The residue bound R<2D<2B^n and H=floor(R/B^(k-1)) establish this
    /// domain at k=floor(n/2)+1: even n gives H<B^k; odd n gives H<2B^k.
    /// Only floor(H*W0/B^k) is multiplied.
    /// The implicit W digit adds H; an implicit H digit adds W0. The return
    /// index skips the final low correction digit within the retained buffer.
    pub fn newton_correction_product(
        error_high: &[Limb],
        inverse_low: &[Limb],
        output: &mut ScratchBuffer,
        carry_product: &mut ScratchBuffer,
        mul_scratch: &mut MulScratch,
    ) -> usize {
        let k = inverse_low.len();
        debug_assert!(
            !error_high.is_empty() && k > 0,
            "reciprocal correction has nonempty error and inverse operands"
        );
        debug_assert!(
            error_high.len().saturating_sub(k) <= 1,
            "R<2B^n bounds H below 2B^k"
        );
        let leading_one = error_high.len() > k;
        let factor = if leading_one {
            debug_assert_eq!(
                error_high.last(),
                Some(&1),
                "a k+1-limb error prefix has leading digit one"
            );
            // SAFETY: the extra leading one follows exactly k>0 low limbs.
            unsafe { error_high.get_unchecked(..k) }
        } else {
            error_high
        };
        let product = HighProduct::mul(factor, inverse_low, k, output, carry_product, mul_scratch);
        let carry = Addition::add_slice_in_place(
            product,
            if leading_one { inverse_low } else { error_high },
        );
        let initialized_len = output.len();
        // SAFETY: the high-product writer reserved one additional carry slot.
        // Its first write initializes that slot before set_len exposes it.
        unsafe {
            output.as_mut_ptr().add(initialized_len).write(carry);
            output.set_len(initialized_len.unchecked_add(1));
        }
        if leading_one {
            // SAFETY: the k-digit high product and its written carry occupy the
            // final k+1 initialized limbs, disjoint from the error prefix.
            let correction =
                unsafe { output.get_unchecked_mut(initialized_len.unchecked_sub(k)..) };
            let overflow = Addition::add_slice_in_place(correction, error_high);
            debug_assert_eq!(overflow, 0, "H*W/B^k < 4B^k fits its high guard");
        }
        // SAFETY: output ends with factor.len()+1 initialized correction digits;
        // discarding its low digit leaves exactly factor.len() retained digits.
        unsafe { output.len().unchecked_sub(factor.len()) }
    }
}

/// Recovers the exact nonnegative residue from a Mersenne product.
///
/// `cyclic_product` contains Q*D modulo M=B^w-1. `fold` has w>=n>=2 limbs,
/// and its adjacent `tail` has 1..=w limbs; U=fold+tail*B^w.
/// Both dividend partitions are disjoint from the cyclic product.
/// The quotient estimate is at most three below the exact quotient, giving
/// 0<=R=U-QD<4D<=4M. For a cyclic residue 0<=r<=M, write R=r+kM.
/// Thus -1<=k<=3; k=-1 occurs only for r=M and R=0.
/// Since M=-1 modulo eight, k=(r-residue_mod8) mod 8. A count of seven
/// identifies the alternate zero without scanning the residue. Three bits
/// distinguish all five possible counts and recover the complete remainder
/// without a guard limb in the transform.
/// Reconstruction replaces `fold` and at most the first tail limb, returning
/// the active width of their combined contiguous dividend prefix.
pub fn newton_wrapped_remainder(
    fold: &mut [Limb],
    tail: &mut [Limb],
    residue_mod8: Limb,
    cyclic_product: &[Limb],
) -> usize {
    // B^w = 1 modulo M. Folding at most 2w input limbs requires one
    // addition and at most one end-around carry. The high dividend span
    // is consumed before any reconstructed high guard is written.
    // SAFETY: zero and scalar estimates return before cyclic admission;
    // 2<=estimate.len()<=n<=w excludes one-limb folds. The caller supplies
    // w initialized fold/product limbs and 1..=w initialized tail limbs.
    let (lower, _) = unsafe { fold.split_last_chunk_mut::<2>().unwrap_unchecked() };
    // SAFETY: the two fixed high digits belong to the materialized fold;
    // the adjacent initialized tail has at least one limb by cyclic admission.
    let (width, tail_width) = unsafe {
        (
            lower.len().unchecked_add(2),
            NonZeroUsize::new_unchecked(tail.len()),
        )
    };
    // SAFETY: tail.len()<=width by the admitted dividend geometry.
    // The addition prefix and complementary carry span remain disjoint.
    let (addends, carry_span) = unsafe { fold.split_at_mut_unchecked(tail_width.get()) };
    let carry = Addition::add_slice_in_place(addends, tail);
    let carry_out = Addition::propagate_carry(carry_span, carry);
    let overflow = Addition::propagate_carry(fold, carry_out);
    debug_assert_eq!(
        overflow, 0,
        "a folded carry leaves room for its end-around one"
    );

    // SAFETY: cyclic admission initializes exactly width product limbs;
    // this view establishes the subtraction width from the fold itself.
    let product = unsafe { cyclic_product.get_unchecked(..width) };
    let borrow = Addition::sub_slice_in_place(fold, product);
    // Subtraction modulo B^w adds B^w after a borrow. Subtracting one
    // replaces that wrap by M. The difference lies in [-M, M], so this
    // end-around borrow cannot underflow the complete residue.
    let underflow = Addition::propagate_borrow(fold, borrow);
    debug_assert_eq!(
        underflow, 0,
        "a modular difference absorbs its end-around borrow"
    );
    // SAFETY: width >= n > 0 bounds the initialized fold's low digit.
    let wraps = unsafe { fold.get_unchecked(0) }.wrapping_sub(residue_mod8) & 7;
    if wraps == 7 {
        // R<4M excludes a nonnegative count of seven. Its residue class
        // therefore denotes k=-1, r=M and R=0. The caller clears any
        // divisor-width remainder required by the following block.
        return 0;
    }
    debug_assert!(wraps <= 3, "R<4M bounds the nonnegative wrap count");
    if wraps != 0 {
        // r + k*(B^w-1) = (r-k) + k*B^w. Subtraction's borrow is
        // absorbed by the positive k, so the high guard is k-borrow.
        // SAFETY: width>=2 supplies the initialized low limb and a
        // nonempty disjoint suffix for the subtraction borrow.
        let (low, higher) = unsafe { fold.split_first_mut().unwrap_unchecked() };
        let (difference, borrow_low) = low.overflowing_sub(wraps);
        *low = difference;
        let wrap_borrow = Addition::propagate_borrow(higher, Limb::from(borrow_low));
        // SAFETY: the nonempty consumed tail has a first initialized digit.
        // wraps>=1 absorbs the binary borrow; width+1 fits the dividend.
        unsafe {
            *tail.get_unchecked_mut(0) = wraps.unchecked_sub(wrap_borrow);
            return width.unchecked_add(1);
        }
    }
    width
}
