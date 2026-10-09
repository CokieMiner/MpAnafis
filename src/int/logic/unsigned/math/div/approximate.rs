//! Recursive upper quotient bounds for normalized division windows.
//!
//! Each level computes the high quotient and its remainder exactly. The final
//! low block divides only its leading `2m` numerator and `m` divisor limbs,
//! retaining an upper quotient rather than reconstructing the omitted product.
//!
//! For `X < D*B^m`, `D = P*B^k+d` and `X' = floor(X/B^k)`, normalization
//! gives `P >= B^m/2`. Thus `floor(X'/P)` never underestimates `floor(X/D)`
//! and exceeds it by at most two. An exact leaf starts with zero error, so
//! at depth `h` the error is at most `2h`. Saturating an overflowing low block
//! to `B^m-1` preserves this bound because its exact quotient fits `m` limbs.
//! A final block narrower than the divisor applies the same truncation bound
//! before its balanced recursion. All preceding blocks retain exact residues.
//!
//! Each low recursion halves the width. A materialized limb slice has
//! `n < B/2`, hence `h < Limb::BITS` and the complete error is below
//! `2*Limb::BITS < B` on 16-, 32- and 64-bit targets. One quotient guard
//! absorbs this error; ambiguous guards use a full product comparison.

#![expect(
    unsafe_code,
    reason = "balanced normalized prefix geometry bounds the leading quotient, recursive windows and disjoint output partitions"
)]

use core::{cmp::Ordering, mem::MaybeUninit, num::NonZeroUsize, ops::Rem};

use super::{
    APPROXIMATE_DIVISION_BLOCK_LIMBS, Addition, BURNIKEL_ZIEGLER_BLOCK_LIMBS, Division,
    InternalMpUint, LIMB_BITS, Limb, MulScratch, PreparedDivisor,
};

const QUOTIENT_ERROR_BOUND: Limb = 2 * LIMB_BITS;

impl Division {
    /// Bounds a normalized quotient without constructing its final residue.
    ///
    /// The normalized divisor has `n>=3` limbs, the quotient is nonempty,
    /// and the numerator has `n+quotient.len()` limbs. Its high `n` limbs
    /// are below the divisor. Product storage holds at least `n` limbs.
    /// Operand and quotient spans are initialized; product storage is writable.
    /// All owners are disjoint. A zero return value proves
    /// `A=floor(U/D)`; otherwise the return value E proves
    /// `floor(U/D) <= A < floor(U/D) + E <= floor(U/D) + 2*Limb::BITS`.
    pub fn burnikel_div_approximate(
        dividend: &mut [Limb],
        divisor: &[Limb],
        quotient: &mut [Limb],
        product: &mut [MaybeUninit<Limb>],
        mul_scratch: &mut MulScratch,
    ) -> Limb {
        // SAFETY: guarded division retains at least three normalized divisor
        // limbs. Its fixed leading triple and remaining prefix determine n.
        let (lower, _) = unsafe { divisor.split_last_chunk::<3>().unwrap_unchecked() };
        // SAFETY: the fixed triple belongs to the materialized divisor.
        let n = unsafe { lower.len().unchecked_add(3) };
        debug_assert!(
            n >= 3,
            "recursive approximation requires at least three divisor limbs"
        );
        debug_assert_eq!(
            dividend.len().checked_sub(n),
            Some(quotient.len()),
            "the normalized numerator and quotient widths must agree"
        );
        debug_assert!(!quotient.is_empty(), "the quotient span is nonempty");
        // SAFETY: guarded division supplies at least three normalized divisor
        // limbs, a nonempty quotient and exactly n+quotient.len() initialized
        // numerator limbs. The positive output width defines that active view.
        let numerator = unsafe {
            let digits = NonZeroUsize::new_unchecked(quotient.len());
            dividend.get_unchecked_mut(..n.unchecked_add(digits.get()))
        };
        let prepared = PreparedDivisor::new(divisor);
        let mut end = quotient.len();
        if numerator.last() == Some(&0) {
            // D >= B^n/2 makes the window below the zero guard less than 2D.
            // Removing its zero-or-one quotient digit leaves an exact remainder.
            // SAFETY: end>0 and numerator.len()=n+end. The leading n-limb
            // window and its exclusive quotient output slot are initialized.
            unsafe {
                end = end.unchecked_sub(1);
                let high = numerator.get_unchecked_mut(end..end.unchecked_add(n));
                let digit = Limb::from(InternalMpUint::cmp_limbs(high, divisor) != Ordering::Less);
                if digit != 0 {
                    let borrow = Addition::sub_slice_in_place(high, divisor);
                    debug_assert_eq!(borrow, 0, "the leading window is at least D");
                }
                *quotient.get_unchecked_mut(end) = digit;
            }
        }
        // SAFETY: the normalized driver supplies n>=3, so this divisor is nonzero.
        let remainder = unsafe { Rem::rem(end, NonZeroUsize::new_unchecked(n)) };
        let mut size = if remainder == 0 { n } else { remainder };
        while end != 0 {
            // SAFETY: size<=end, numerator.len()=n+quotient.len(), and
            // start=end-size. The disjoint output and operand windows fit
            // their owners; each preceding block leaves a high remainder < D.
            let (start, window, digits) = unsafe {
                let start = end.unchecked_sub(size);
                (
                    start,
                    numerator.get_unchecked_mut(start..end.unchecked_add(n)),
                    quotient.get_unchecked_mut(start..end),
                )
            };
            if start == 0 && size >= 2 {
                // Only this block may omit the low divisor product. Prefix
                // truncation adds at most two to the recursive error, with
                // size<=n. No subsequent block consumes its remainder.
                // SAFETY: size<=n and window.len()=n+size. Removing n-size
                // low limbs leaves exactly 2*size initialized numerator limbs
                // and size divisor limbs with the same prepared leading pair.
                let (prefix, top_divisor) = unsafe {
                    let omitted = n.unchecked_sub(size);
                    (
                        window.get_unchecked_mut(omitted..),
                        divisor.get_unchecked(omitted..),
                    )
                };
                let high_bit = burnikel_div_approximate_recursive(
                    digits,
                    prefix,
                    top_divisor,
                    product,
                    mul_scratch,
                    &prepared,
                );
                if high_bit != 0 {
                    // X<D*B^size bounds the exact quotient below B^size.
                    // Clipping an overflowing upper estimate preserves its bound.
                    digits.fill(Limb::MAX);
                }
                // Earlier blocks have exact residues. An untruncated short
                // final leaf therefore makes the complete quotient exact.
                return if size == n && size <= APPROXIMATE_DIVISION_BLOCK_LIMBS {
                    0
                } else {
                    QUOTIENT_ERROR_BOUND
                };
            } else if size < BURNIKEL_ZIEGLER_BLOCK_LIMBS.max(4) {
                prepared.divide(window, divisor, digits);
            } else {
                let high_bit = Self::burnikel_div_block::<true>(
                    digits,
                    window,
                    divisor,
                    product,
                    mul_scratch,
                    &prepared,
                );
                debug_assert_eq!(high_bit, 0, "the preceding remainder is below D");
            }
            end = start;
            size = n;
        }
        // No recursive final block ran: the last full-width scalar step, or
        // the sole zero-or-one leading digit, computed the exact quotient.
        0
    }
}

/// Returns an upper quotient for `2n/n` and its exact bit of weight `B^n`.
/// The low `n` digits have error below `2*Limb::BITS`; no residue is consumed.
/// Each recursive prefix shares the prepared leading divisor pair.
fn burnikel_div_approximate_recursive(
    output: &mut [Limb],
    dividend: &mut [Limb],
    divisor: &[Limb],
    product: &mut [MaybeUninit<Limb>],
    mul_scratch: &mut MulScratch,
    prepared: &PreparedDivisor,
) -> Limb {
    // SAFETY: the driver supplies n>=2, n output limbs and 2n numerator
    // limbs. Each split retains floor(n/2)>=2 divisor limbs and exactly
    // twice that many numerator limbs. These equalities therefore hold
    // at every level and exclude empty or mismatched primitive windows.
    let (n, quotient, numerator) = unsafe {
        let (lower, _) = divisor.split_last_chunk::<2>().unwrap_unchecked();
        let n = lower.len().unchecked_add(2);
        (
            n,
            output.get_unchecked_mut(..n),
            dividend.get_unchecked_mut(..n.unchecked_mul(2)),
        )
    };
    if n <= APPROXIMATE_DIVISION_BLOCK_LIMBS {
        // SAFETY: every balanced prefix has n>=2 and 2n initialized
        // numerator limbs. Its top n-limb span is an exclusive view.
        let (_, high) = unsafe { numerator.split_at_mut_unchecked(n) };
        let high_bit = Limb::from(InternalMpUint::cmp_limbs(high, divisor) != Ordering::Less);
        if high_bit != 0 {
            let borrow = Addition::sub_slice_in_place(high, divisor);
            debug_assert_eq!(borrow, 0, "the high window is at least the divisor");
        }
        if n >= 3
                // SAFETY: this balanced leaf has 2n initialized numerator
                // limbs and n>=3 initialized divisor limbs.
                && unsafe {
                    *numerator.last().unwrap_unchecked() < *divisor.last().unwrap_unchecked()
                }
        {
            // The exact short leaf needs no remainder and shares the
            // upper-bound contract with the recursive approximation.
            let _ = prepared.divide_quotient::<false, false, false>(numerator, divisor, quotient);
        } else {
            prepared.divide(numerator, divisor, quotient);
        }
        return high_bit;
    }
    let low = n >> 1;
    // SAFETY: quotient.len()=n and numerator.len()=2n. Splitting at
    // floor(n/2) leaves disjoint outputs and an n+ceil(n/2) high window.
    let (quotient_low, quotient_high, high_window) = unsafe {
        let (low_digits, high_digits) = quotient.split_at_mut_unchecked(low);
        (low_digits, high_digits, numerator.get_unchecked_mut(low..))
    };
    let high_bit = Division::burnikel_div_block::<true>(
        quotient_high,
        high_window,
        divisor,
        product,
        mul_scratch,
        prepared,
    );
    // The exact high block leaves X < D*B^low. Its leading 2*low limbs
    // divided by the leading low divisor limbs give an upper estimate.
    // SAFETY: low=floor(n/2)>=2 and omitted=n-low<=n. The numerator
    // suffix ends at n+low<=2n and contains exactly 2*low limbs. The
    // divisor suffix contains low limbs and preserves the prepared pair.
    let (prefix, top_divisor) = unsafe {
        let omitted = n.unchecked_sub(low);
        (
            numerator.get_unchecked_mut(omitted..n.unchecked_add(low)),
            divisor.get_unchecked(omitted..),
        )
    };
    let low_bit = burnikel_div_approximate_recursive(
        quotient_low,
        prefix,
        top_divisor,
        product,
        mul_scratch,
        prepared,
    );
    if low_bit != 0 {
        // The exact low quotient is below B^low. Clipping an overflowing
        // upper estimate to its largest representable value remains upper.
        quotient_low.fill(Limb::MAX);
    }
    high_bit
}
