//! In-place normalized Karatsuba root and remainder reconstruction.
//!
//! Reference: P. Zimmermann, "Karatsuba Square Root", INRIA RR-3805, 1999.
//! <https://inria.hal.science/inria-00072854>

#![expect(
    unsafe_code,
    reason = "normalized root geometry bounds basecase widths, quotient spans, and one-bit shifts"
)]

use core::cmp::Ordering;

use super::{
    Addition, ArchKernels, DivScratch, Division, InternalMpUint, Limb, MulScratch, Multiplication,
    Roots,
};

impl Roots {
    /// Writes the root of a normalized 2n-limb input into n limbs. With REMAINDER,
    /// its exact residue replaces the low n+1 input limbs; the other input limbs
    /// become scratch. The highest two input bits are not both zero.
    ///
    /// Scratch has at least n+floor(n/2)+2 limbs. Recursion runs before partitioning
    /// that span, so every level reuses the same quotient and square storage.
    #[expect(
        clippy::too_many_lines,
        reason = "The normalized root recurrence keeps borrowed quotient, square, and correction spans in one coupled scratch layout"
    )]
    pub fn sqrt_rem_recursive<const REMAINDER: bool>(
        input: &mut [Limb],
        root: &mut [Limb],
        scratch: &mut [Limb],
        division: &mut DivScratch,
        multiplication: &mut MulScratch,
    ) {
        let n = root.len();
        debug_assert_eq!(
            n.checked_mul(2),
            Some(input.len()),
            "the normalized input has twice the root width"
        );
        if n <= 2 {
            let mut value = InternalMpUint::zero();
            value.clone_from_slice(input);
            // SAFETY: input has 2n <= 4 limbs, precisely the inline kernel's
            // accepted domain. Eliminating its Option test leaves an infallible leaf.
            let answer = unsafe { Self::isqrt_inline(&value).unwrap_unchecked() };
            root.fill(0);
            // SAFETY: input has 2n limbs, so its floor root has at most n.
            let (root_digits, _) = unsafe { root.split_at_mut_unchecked(answer.limbs().len()) };
            root_digits.copy_from_slice(answer.limbs());
            if REMAINDER {
                let remainder = value.sub(&answer.square());
                input.fill(0);
                // SAFETY: the nonnegative remainder is at most the input,
                // so its normalized digits fit the input's initialized span.
                let (rem_digits, _) =
                    unsafe { input.split_at_mut_unchecked(remainder.limbs().len()) };
                rem_digits.copy_from_slice(remainder.limbs());
            }
            return;
        }

        let k = n >> 1;
        // SAFETY: n >= 3 and input.len()=2n prove 2k <= n, k+1 <= n,
        // and n+1 <= input.len() on every supported pointer width.
        let (low_width, quotient_len, remainder_len) =
            unsafe { (k.unchecked_mul(2), k.unchecked_add(1), n.unchecked_add(1)) };
        // SAFETY: k=floor(n/2)<n and low_width=2k<=n<input.len()=2n.
        // The separate owners yield disjoint initialized mutable partitions.
        let (low_root, high_root, high_input) = unsafe {
            let (low, high) = root.split_at_mut_unchecked(k);
            (low, high, input.split_at_mut_unchecked(low_width).1)
        };
        Self::sqrt_rem_recursive::<true>(high_input, high_root, scratch, division, multiplication);

        // A=H*B^(2k)+M*B^k+L and H=S^2+R. Divide X=R*B^k+M by S,
        // giving v and r. Then q=floor(v/2), u=r+(v mod 2)*S establish
        // X=2*S*q+u without constructing 2S or normalizing a second divisor.
        // S >= B^(n-k)/2 sets its high bit. R <= 2S bounds q <= B^k,
        // and the high divisor-width window of X is floor(R/B) < S.
        // SAFETY: scratch.len()>=n+k+2 leaves n+1 limbs after k+1 quotient
        // limbs. The input suffix has 2n-k>=n+1 limbs because n-k>=1.
        let (quotient, rest, window) = unsafe {
            let (quotient, rest) = scratch.split_at_mut_unchecked(quotient_len);
            let upper = input.split_at_mut_unchecked(k).1;
            (
                quotient,
                rest,
                upper.split_at_mut_unchecked(remainder_len).0,
            )
        };
        Division::div_rem_normalized::<true>(window, high_root, quotient, division);
        // SAFETY: quotient_len=k+1 >= 2, so the initialized low quotient exists.
        let odd = unsafe { *quotient.get_unchecked(0) } & 1;
        // SAFETY: k+1 initialized quotient limbs are exclusively borrowed; one
        // is a positive shift strictly below every supported limb width.
        let _ = unsafe { ArchKernels::rshift_unchecked(quotient.as_mut_ptr(), quotient_len, 1) };
        // SAFETY: high_root.len()=n-k<n+1=window.len().
        let (residue, upper_window) = unsafe { window.split_at_mut_unchecked(high_root.len()) };
        let carry = if odd != 0 {
            Addition::add_slice_in_place(residue, high_root)
        } else {
            0
        };
        // SAFETY: window has n+1 limbs and high_root has n-k < n+1 limbs.
        // This first guard completes the h+1-limb value u < 2S.
        unsafe {
            *upper_window.get_unchecked_mut(0) = carry;
        }
        // SAFETY: quotient has k+1 initialized limbs, retaining one high digit.
        let (quotient_low, quotient_high) = unsafe { quotient.split_at_unchecked(k) };
        low_root.copy_from_slice(quotient_low);
        // SAFETY: quotient has k+1 limbs; q <= B^k makes its last limb binary.
        let high_digit = unsafe { *quotient_high.get_unchecked(0) };
        let root_carry = Addition::propagate_carry(high_root, high_digit);

        // If u >= q, u*B^k+L >= q^2 because q <= B^k. The root-only
        // specialization then avoids both the square and remainder reconstruction.
        // SAFETY: n>=3 gives n+1=remainder_len<=2n=input.len().
        let (remainder, _) = unsafe { input.split_at_mut_unchecked(remainder_len) };
        if !REMAINDER {
            // SAFETY: remainder has n+1 limbs and n-k>=k, so its suffix
            // after k digits contains at least k+1=quotient_len digits.
            let (comparable, excess) = unsafe {
                remainder
                    .split_at_unchecked(k)
                    .1
                    .split_at_unchecked(quotient_len)
            };
            if excess.iter().any(|&limb| limb != 0)
                || comparable.iter().rev().cmp(quotient.iter().rev()) != Ordering::Less
            {
                debug_assert_eq!(root_carry, 0, "a certified root fits n limbs");
                return;
            }
        }

        // SAFETY: the quotient partition leaves rest.len()>=n+1. Its square
        // prefix has 2k<=n digits, leaving at least one initialized guard.
        let (square, _) = unsafe { rest.split_at_mut_unchecked(remainder_len) };
        // SAFETY: low_width=2k<=n<n+1=square.len(); the partitions are disjoint.
        let (square_digits, square_guard) = unsafe { square.split_at_mut_unchecked(low_width) };
        Multiplication::sqr_limbs_with_scratch(quotient_low, square_digits, multiplication);
        // q=B^k is the only case with a high quotient limb. Its low k limbs
        // are zero and q^2 has just bit 2k. Every excess scratch limb is cleared.
        square_guard.fill(0);
        // SAFETY: 2k <= n < remainder_len proves the first square guard exists.
        unsafe {
            *square_guard.get_unchecked_mut(0) = high_digit;
        }
        let borrow = if REMAINDER {
            Addition::sub_slice_in_place(remainder, square)
        } else {
            Limb::from(remainder.iter().rev().cmp(square.iter().rev()) == Ordering::Less)
        };
        if borrow != 0 {
            // S >= B^k/2 bounds the estimate to at most one above the floor
            // root. For corrected root s, its remainder is R'+2s+1. R' is
            // stored modulo B^(n+1); the final carry consumes its negative sign.
            let root_borrow = Addition::propagate_borrow(root, 1);
            debug_assert_eq!(
                root_borrow, root_carry,
                "correction consumes the root guard"
            );
            if REMAINDER {
                let first = add_limbs_in_place(remainder, root);
                let second = add_limbs_in_place(remainder, root);
                let final_carry = Addition::propagate_carry(remainder, 1);
                debug_assert_eq!(
                    first
                        .checked_add(second)
                        .and_then(|sum| sum.checked_add(final_carry)),
                    Some(1),
                    "correction cancels one borrow"
                );
            }
        } else {
            debug_assert_eq!(root_carry, 0, "an uncorrected root fits n limbs");
        }
    }
}

/// Adds `src` into `dst`, including carry propagation above the source width.
/// Returns the high carry; `dst.len() >= src.len()`.
fn add_limbs_in_place(dst: &mut [Limb], src: &[Limb]) -> Limb {
    let src_len = src.len();
    if src_len == 0 {
        return 0;
    }
    let carry = Addition::add_slice_in_place(dst, src);
    if carry == 0 {
        return 0;
    }
    // SAFETY: the recursive remainder has at least the root's width,
    // so the disjoint carry suffix begins within the initialized destination.
    let dst_upper = unsafe { dst.get_unchecked_mut(src_len..) };
    Addition::propagate_carry(dst_upper, carry)
}
