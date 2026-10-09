//! Divide-and-conquer division without divisor padding.
//!
//! A normalized `2n / n` division splits its quotient into `hi = ceil(n/2)`
//! high and `lo = floor(n/2)` low limbs. Each half is one block: an `m`-limb
//! quotient over an `(m + n)`-limb window. The block divides its top `2m`
//! limbs by the top `m` divisor limbs, then subtracts the block quotient's
//! product with the `n - m` omitted divisor limbs, restoring the exact partial
//! remainder for the next block. Both products of a split are `hi x lo`, and
//! no block pads the divisor, so odd and non-power-of-two lengths cost the
//! same recursion as their neighbours.
//!
//! # Truncated block quotient
//!
//! Let `X` be a block window, `D` the divisor and `k = n - m`, with
//! `X' = floor(X/B^k)`, `D' = floor(D/B^k)` and `X < D*B^m`. Since
//! `X/D < (X' + 1)/D'`, the truncated quotient never underestimates. Since
//! `X' < (D' + 1)*B^m` and normalization gives `D' >= B^m/2`, it exceeds
//! `X/D` by less than `X'/(D'(D' + 1)) < B^m/D' <= 2`. The repaired remainder
//! therefore lies in `[-2D, D)`, and at most two additions of `D` finish the
//! block.
//!
//! References:
//! - C. Burnikel and J. Ziegler, "Fast Recursive Division", Research Report MPI-I-98-1-022,
//!   Max-Planck-Institut für Informatik, Oct. 1998.

#![expect(
    unsafe_code,
    reason = "normalized block geometry bounds quotient windows and shifts; the multi-limb entry establishes nonzero block widths"
)]

use core::{cmp::Ordering, mem::MaybeUninit, num::NonZeroUsize, ops::Rem};

use super::{
    Addition, ArchKernels, BURNIKEL_QUOTIENT_THRESHOLD, BURNIKEL_ZIEGLER_BLOCK_LIMBS, DivScratch,
    Division, InternalMpUint, Limb, MulScratch, Multiplication, PreparedDivisor,
};

/// Smallest block the recursion splits. The tuned block width is an empirical
/// leaf size; the floor of four keeps the two leading limbs that the 3-by-2
/// basecase requires in both halves.
const SPLIT_LIMBS: usize = if BURNIKEL_ZIEGLER_BLOCK_LIMBS > 4 {
    BURNIKEL_ZIEGLER_BLOCK_LIMBS
} else {
    4
};

impl Division {
    /// Divides `num_a` by `den_b` with the divide-and-conquer recursion.
    ///
    /// A canonical nonzero divisor and one normalization guard admit a leading
    /// quotient block of `1..=n` limbs, then `n`-limb blocks sharing the remainder.
    pub fn burnikel_ziegler<const WRITE_REMAINDER: bool>(
        num_a: &InternalMpUint,
        den_b: &InternalMpUint,
        quotient_out: &mut InternalMpUint,
        rem_out: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        let v_limbs = den_b.limbs();
        debug_assert!(!v_limbs.is_empty(), "division requires a non-zero divisor");
        let u_limbs = num_a.limbs();
        if u_limbs.len() < v_limbs.len()
            || (u_limbs.len() == v_limbs.len()
                && InternalMpUint::cmp_limbs(u_limbs, v_limbs) == Ordering::Less)
        {
            quotient_out.clear();
            if WRITE_REMAINDER {
                rem_out.clone_from(num_a);
            }
            return;
        }
        if let [divisor] = v_limbs {
            let remainder = Self::div_rem_1::<true>(u_limbs, *divisor, quotient_out);
            if WRITE_REMAINDER {
                *rem_out = InternalMpUint::from_limb(remainder);
            }
            return;
        }

        // SAFETY: the nonzero canonical divisor has an initialized high limb.
        let shift = unsafe { v_limbs.last().unwrap_unchecked() }.leading_zeros();
        let divisor = if shift == 0 {
            v_limbs
        } else {
            Self::shift_limbs_left::<false>(v_limbs, shift, &mut scratch.v_norm);
            scratch.v_norm.as_slice()
        };
        Self::shift_limbs_left::<true>(u_limbs, shift, &mut scratch.u_norm);
        let n = v_limbs.len();
        // SAFETY: the width comparison above proves u_limbs.len() >= n;
        // the additional normalization guard leaves a nonempty quotient.
        let quotient_len = unsafe { scratch.u_norm.len().unchecked_sub(n) };
        scratch.recursive_product.reset_with_capacity(n);
        let quotient = quotient_out.ensure_capacity_set_len_get_limbs(quotient_len);
        let numerator = scratch.u_norm.as_mut_slice();
        // SAFETY: reservation supplies n disjoint writable spare limbs.
        // Recursive repair initializes each product span before reading it.
        let product = unsafe {
            scratch
                .recursive_product
                .spare_capacity_mut()
                .get_unchecked_mut(..n)
        };
        Self::burnikel_div_rem_normalized::<WRITE_REMAINDER>(
            numerator,
            divisor,
            quotient,
            product,
            &mut scratch.mul_scratch,
        );

        if WRITE_REMAINDER {
            // SAFETY: normalization initializes n+quotient_len limbs, and
            // the kernel preserves its low n-limb remainder in this span.
            let remainder = unsafe { numerator.get_unchecked(..n) };
            if shift == 0 {
                rem_out.clone_from_slice(remainder);
            } else {
                let mut output = rem_out.prepare_limb_write(n);
                // SAFETY: normalization gives 0<shift<LIMB_BITS. The kernel
                // reads n initialized remainder limbs and initializes every
                // reserved output limb in a disjoint allocation before commit.
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
        }
        quotient_out.normalize();
    }

    /// Divides disjoint normalized limb spans, leaving the remainder in place.
    ///
    /// The divisor has at least two limbs. The numerator has exactly
    /// `divisor.len()+quotient.len()` limbs, with its high divisor-width
    /// window below the divisor. Product storage has at least `divisor.len()` limbs.
    /// Quotient-only blocks may discard a certified final remainder.
    pub fn burnikel_div_rem_normalized<const WRITE_REMAINDER: bool>(
        numerator: &mut [Limb],
        divisor: &[Limb],
        quotient: &mut [Limb],
        product: &mut [MaybeUninit<Limb>],
        mul_scratch: &mut MulScratch,
    ) {
        // SAFETY: normalized dispatch retains the leading divisor pair and
        // supplies at least one output digit. The pair and prefix determine n.
        let (n, mut end) = unsafe {
            let (lower, _) = divisor.split_last_chunk::<2>().unwrap_unchecked();
            (
                lower.len().unchecked_add(2),
                NonZeroUsize::new_unchecked(quotient.len()).get(),
            )
        };
        if numerator.last() == Some(&0) {
            // Normalization gives D >= B^n/2. With a zero guard the leading
            // n-limb window is below B^n <= 2D, so its digit is zero or one.
            // Remove it directly before forming recursive quotient blocks.
            // SAFETY: quotient_len > 0 and numerator.len() = n+quotient_len.
            // The leading window and its initialized output digit both exist.
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
        let prepared = PreparedDivisor::new(divisor);
        // SAFETY: the single-limb divisor returned above, so n >= 2 > 0.
        let rem = unsafe { Rem::rem(end, NonZeroUsize::new_unchecked(n)) };
        let mut size = if rem == 0 { n } else { rem };
        while end != 0 {
            // SAFETY: each block has size <= end, and its n+size limbs fit
            // the existing numerator span of n+quotient_len initialized limbs.
            let (start, window_len) = unsafe { (end.unchecked_sub(size), size.unchecked_add(n)) };
            // SAFETY: start+window_len = end+n <= numerator.len(), and
            // start+size = end <= quotient.len(). Both owners are initialized;
            // each partition keeps the selected block exclusive.
            let (window, block) = unsafe {
                let (_, numerator_suffix) = numerator.split_at_mut_unchecked(start);
                let (window, _) = numerator_suffix.split_at_mut_unchecked(window_len);
                let (_, quotient_suffix) = quotient.split_at_mut_unchecked(start);
                let (block, _) = quotient_suffix.split_at_mut_unchecked(size);
                (window, block)
            };
            // A block narrower than the split width uses the full divisor;
            // its high window is the preceding remainder, already below D.
            let high_bit = if size < SPLIT_LIMBS {
                prepared.divide(window, divisor, block);
                0
            } else if start == 0 {
                Self::burnikel_div_block::<WRITE_REMAINDER>(
                    block,
                    window,
                    divisor,
                    product,
                    mul_scratch,
                    &prepared,
                )
            } else {
                Self::burnikel_div_block::<true>(
                    block,
                    window,
                    divisor,
                    product,
                    mul_scratch,
                    &prepared,
                )
            };
            debug_assert_eq!(
                high_bit, 0,
                "the zero-extended dividend keeps every window below the divisor"
            );
            end = start;
            size = n;
        }
    }
    /// Divides an `(m + n)`-limb `window` by the normalized `n`-limb `divisor`,
    /// `2 <= m <= n`, writing the `m` limbs of `quotient`.
    ///
    /// The leading division retains `m` divisor limbs, or `m + 1` when a
    /// quotient-only block can omit at least one more limb. The product of that
    /// quotient with the omitted divisor limbs (and
    /// their `B^m` multiple for a set high bit) is subtracted from the low `n`
    /// window limbs; the module bound limits the add-backs to two. The remainder
    /// replaces the low `n` window limbs when requested; a quotient-only block
    /// may certify its prefix quotient without forming that product.
    /// Returns the quotient bit of weight
    /// `B^m`, set only when the top `n` window limbs are at least the divisor.
    #[expect(
        clippy::too_many_lines,
        reason = "prefix division, quotient certification and bounded residue repair share the same borrowed block and product workspace"
    )]
    pub fn burnikel_div_block<const WRITE_REMAINDER: bool>(
        quotient: &mut [Limb],
        window: &mut [Limb],
        divisor: &[Limb],
        product: &mut [MaybeUninit<Limb>],
        mul_scratch: &mut MulScratch,
        prepared: &PreparedDivisor,
    ) -> Limb {
        let n = divisor.len();
        // SAFETY: block dispatch and recursive splits supply 2<=m<=n.
        let (m, omitted_count) = unsafe {
            let (lower, _) = quotient.split_last_chunk::<2>().unwrap_unchecked();
            let m = lower.len().unchecked_add(2);
            (m, n.unchecked_sub(m))
        };
        let guarded = !WRITE_REMAINDER && omitted_count > 1;
        let retained = if guarded {
            // SAFETY: guarded implies m+1 < n, so the additional retained limb
            // fits the existing divisor span and cannot overflow usize.
            unsafe { m.unchecked_add(1) }
        } else {
            m
        };
        // SAFETY: retained=m, or guarded proves retained=m+1<n.
        let omitted_len = unsafe { n.unchecked_sub(retained) };
        // SAFETY: omitted_len = n-retained <= n. Both immutable partitions
        // retain the initialized divisor's lifetime and alignment.
        let (omitted, top_divisor) = unsafe { divisor.split_at_unchecked(omitted_len) };
        let mut high_bit = {
            // SAFETY: window.len() = n+m and omitted_len <= n. The suffix
            // has retained+m initialized limbs for the admitted subproblem.
            let (_, top_window) = unsafe { window.split_at_mut_unchecked(omitted_len) };
            if guarded
                && retained < BURNIKEL_QUOTIENT_THRESHOLD
                // SAFETY: retained=m+1>=3 and top_window.len()=retained+m.
                // Both nonempty spans contain initialized leading limbs.
                && unsafe {
                    top_window.last().unwrap_unchecked() < top_divisor.last().unwrap_unchecked()
                }
            {
                // The short kernel retains m quotient digits and one divisor
                // guard. Its certificate gives R'>Q, proving the full quotient
                // exact without completing the prefix remainder or multiplying
                // the omitted divisor limbs.
                if prepared.divide_quotient::<true, true, false>(top_window, top_divisor, quotient)
                {
                    return 0;
                }
                0
            } else if guarded {
                Self::burnikel_div_block::<true>(
                    quotient,
                    top_window,
                    top_divisor,
                    product,
                    mul_scratch,
                    prepared,
                )
            } else if omitted_len == 0 {
                burnikel_div_recursive::<WRITE_REMAINDER>(
                    quotient,
                    top_window,
                    top_divisor,
                    product,
                    mul_scratch,
                    prepared,
                )
            } else {
                burnikel_div_recursive::<true>(
                    quotient,
                    top_window,
                    top_divisor,
                    product,
                    mul_scratch,
                    prepared,
                )
            }
        };
        if omitted_len == 0 {
            return high_bit;
        }
        if !WRITE_REMAINDER && high_bit == 0 {
            // U = U'*B^k+a, D = D'*B^k+b, U' = Q*D'+r. Thus
            // U-Q*D = r*B^k+a-Q*b >= 0 when r >= Q, since b < B^k.
            // Truncation gives an upper quotient estimate; nonnegative residue
            // therefore proves Q exact. No later block consumes this remainder.
            // SAFETY: window.len() = n+m and omitted_len+retained = n.
            // The retained prefix contains at least m limbs; its remaining
            // guard is initialized and either empty or one limb wide.
            let (comparable, guard) = unsafe {
                let (_, upper) = window.split_at_unchecked(omitted_len);
                let (remainder, _) = upper.split_at_unchecked(retained);
                remainder.split_at_unchecked(m)
            };
            if guard.iter().any(|&limb| limb != 0)
                || comparable.iter().rev().cmp(quotient.iter().rev()) != Ordering::Less
            {
                return 0;
            }
        }
        // SAFETY: window.len() = n+m includes the complete n-limb residue.
        let (region, _) = unsafe { window.split_at_mut_unchecked(n) };
        let mut borrow = if let [scalar] = omitted {
            // A guarded prefix omits one divisor limb. Fuse its product and
            // subtraction instead of writing and then reading an m+1-limb product.
            // SAFETY: m <= n = region.len(); omitted_len=1 implies m<n.
            let (digits, guard) = unsafe { region.split_at_mut_unchecked(m) };
            // SAFETY: digits and quotient each contain m initialized limbs and
            // their exclusive borrows are disjoint. The architecture facade
            // establishes any CPU-feature requirements of the selected kernel.
            let (carry, low_borrow) = unsafe {
                ArchKernels::sub_mul_limbs_unchecked(
                    digits.as_mut_ptr(),
                    quotient.as_ptr(),
                    m,
                    *scalar,
                )
            };
            // SAFETY: omitted_len = 1 implies n > m, so the guard is nonempty.
            // For scalar > 0, Q < B^m gives carry < scalar; low_borrow <= 1
            // therefore gives carry+low_borrow <= scalar <= Limb::MAX. A zero
            // scalar produces both zero carry and zero borrow.
            let (high, tail, high_subtrahend) = unsafe {
                let (high, tail) = guard.split_first_mut().unwrap_unchecked();
                (high, tail, carry.unchecked_add(low_borrow))
            };
            let (difference, high_borrow) = high.overflowing_sub(high_subtrahend);
            *high = difference;
            Addition::propagate_borrow(tail, Limb::from(high_borrow))
        } else {
            // SAFETY: the root reserves n limbs and recursive widths shrink.
            // m+omitted_len=n-usize::from(guarded)<=n fits on every target.
            let (repair, product_len) =
                unsafe { (product.get_unchecked_mut(..n), m.unchecked_add(omitted_len)) };
            // SAFETY: product_len<=n bounds the disjoint writable product
            // prefix and partitions the initialized residue at the same width.
            let (product_digits, (digits, guard)) = unsafe {
                (
                    repair.get_unchecked_mut(..product_len),
                    region.split_at_mut_unchecked(product_len),
                )
            };
            // SAFETY: m>0 and omitted_len>0 give nonempty disjoint operands.
            // Their exact product initializes precisely product_len limbs;
            // the unused product guard is never exposed as initialized storage.
            let initialized_product = unsafe {
                Multiplication::mul_nonempty_distinct_into_uninit(
                    quotient,
                    omitted,
                    product_digits,
                    mul_scratch,
                )
            };
            let low_borrow = Addition::sub_slice_in_place(digits, initialized_product);
            // U-Q*D uses the product prefix and a zero high product limb.
            // Only a borrow can change the remaining zero-or-one residue guard.
            Addition::propagate_borrow(guard, low_borrow)
        };
        if high_bit != 0 {
            // SAFETY: m+omitted_len <= n = region.len(). These disjoint
            // initialized spans subtract the omitted divisor's high-bit product.
            let (digits, guard) = unsafe {
                let (_, upper) = region.split_at_mut_unchecked(m);
                upper.split_at_mut_unchecked(omitted_len)
            };
            let high_borrow = Addition::sub_slice_in_place(digits, omitted);
            let extra_borrow = Addition::propagate_borrow(guard, high_borrow);
            // SAFETY: both subtraction borrows are binary, so their sum is
            // at most two, fitting every supported native limb.
            borrow = unsafe { borrow.unchecked_add(extra_borrow) };
        }
        while borrow != 0 {
            let quotient_borrow = Addition::propagate_borrow(quotient, 1);
            let carry = Addition::add_slice_in_place(region, divisor);
            // SAFETY: a negative residue proves the complete quotient is
            // positive, so a low-span borrow is absorbed by high_bit. The
            // loop has borrow >= 1, which absorbs the binary divisor-add carry.
            unsafe {
                high_bit = high_bit.unchecked_sub(quotient_borrow);
                borrow = borrow.unchecked_sub(carry);
            }
        }
        high_bit
    }
}

/// Divides the `2n` limbs of `numerator` by the normalized `n`-limb `divisor`.
///
/// The high block's window is `numerator[lo..2n]`; the low block's window,
/// `numerator[..n + lo]`, starts from the high block's remainder, so its
/// quotient fits the low limbs. Returns the quotient bit of weight `B^n`.
/// `product` holds at least `n` limbs.
fn burnikel_div_recursive<const WRITE_REMAINDER: bool>(
    quotient_storage: &mut [Limb],
    numerator_storage: &mut [Limb],
    divisor: &[Limb],
    product: &mut [MaybeUninit<Limb>],
    mul_scratch: &mut MulScratch,
    prepared: &PreparedDivisor,
) -> Limb {
    // SAFETY: balanced roots and recursive halves retain the leading
    // divisor pair, n quotient digits and 2n initialized numerator digits;
    // SPLIT_LIMBS>=4 keeps every half at least two limbs wide.
    let (n, quotient, numerator) = unsafe {
        let (lower, _) = divisor.split_last_chunk::<2>().unwrap_unchecked();
        let n = lower.len().unchecked_add(2);
        (
            n,
            quotient_storage.get_unchecked_mut(..n),
            numerator_storage.get_unchecked_mut(..n.unchecked_mul(2)),
        )
    };
    if n < SPLIT_LIMBS {
        // SAFETY: numerator has exactly 2n initialized limbs; its n-limb
        // high half is an exclusive view for the leading-bit subtraction.
        let (_, top) = unsafe { numerator.split_at_mut_unchecked(n) };
        let high_bit = Limb::from(InternalMpUint::cmp_limbs(top, divisor) != Ordering::Less);
        if high_bit != 0 {
            let borrow = Addition::sub_slice_in_place(top, divisor);
            debug_assert_eq!(borrow, 0, "the top window is at least the divisor");
        }
        prepared.divide(numerator, divisor, quotient);
        return high_bit;
    }
    let low = n >> 1;
    // SAFETY: quotient.len() = n and low=floor(n/2) <= n. Both quotient
    // partitions are initialized and remain exclusive through recursion.
    let (quotient_low, quotient_high) = unsafe { quotient.split_at_mut_unchecked(low) };
    let high_bit = {
        // SAFETY: low <= n < 2n = numerator.len(); the suffix contains
        // n+quotient_high.len() initialized limbs for the high subproblem.
        let (_, high_window) = unsafe { numerator.split_at_mut_unchecked(low) };
        Division::burnikel_div_block::<true>(
            quotient_high,
            high_window,
            divisor,
            product,
            mul_scratch,
            prepared,
        )
    };
    // SAFETY: low=floor(n/2) and numerator has 2n initialized limbs;
    // n+low fits its existing span and usize.
    let (low_window, _) = unsafe {
        let low_end = n.unchecked_add(low);
        numerator.split_at_mut_unchecked(low_end)
    };
    let low_bit = Division::burnikel_div_block::<WRITE_REMAINDER>(
        quotient_low,
        low_window,
        divisor,
        product,
        mul_scratch,
        prepared,
    );
    debug_assert_eq!(low_bit, 0, "the low block quotient fits its limbs");
    high_bit
}
