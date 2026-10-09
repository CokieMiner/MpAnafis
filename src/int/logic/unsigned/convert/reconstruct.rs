//! Reconstruction of validated radix chunks using cached binary split powers.

#![expect(
    unsafe_code,
    reason = "Materialized chunk counts and cached exponents bound child slices and reserved product spans; kernels initialize each active limb before commitment"
)]

use core::{
    mem::MaybeUninit,
    num::NonZeroUsize,
    ptr::{copy_nonoverlapping, write_bytes},
    slice::from_raw_parts_mut,
};

use alloc::vec::Vec;

use super::{
    Addition, Convert, DoubleLimb, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, MulScratch,
    Multiplication,
};

/// Represents `value * B^low_limbs = chunk_base^exponent`, with `B = 2^LIMB_BITS`.
struct ParsePower {
    value: InternalMpUint,
    low_limbs: usize,
    exponent: usize,
}

impl Convert {
    /// Reconstructs nonempty little-endian chunks below `chunk_base < B`.
    ///
    /// `leaf_chunks` is nonzero. Split powers and multiplication scratch are
    /// constructed once and reused throughout the recursive reconstruction.
    pub fn reconstruct_chunks(
        chunks: &[Limb],
        chunk_base: Limb,
        leaf_chunks: usize,
    ) -> InternalMpUint {
        debug_assert!(
            !chunks.is_empty() && leaf_chunks != 0,
            "nonempty chunks and leaves are required"
        );
        if chunks.len() <= leaf_chunks {
            return reconstruct_leaf(chunks, chunk_base);
        }
        let powers = build_parse_powers(chunks.len(), chunk_base);
        let mut scratch = MulScratch::default();
        combine_chunks(chunks, &powers, chunk_base, leaf_chunks, &mut scratch)
    }
}

/// Builds powers for successive binary prefixes of `floor(chunk_count / 2)`.
fn build_parse_powers(chunk_count: usize, chunk_base: Limb) -> Vec<ParsePower> {
    debug_assert!(chunk_count > 1, "recursive input contains multiple chunks");
    let midpoint = chunk_count >> 1;
    // SAFETY: midpoint != 0. Its bit width is in 1..=usize::BITS <= 64.
    let levels = unsafe {
        usize::try_from(usize::BITS.unchecked_sub(midpoint.leading_zeros())).unwrap_unchecked()
    };
    let mut powers = Vec::with_capacity(levels);
    powers.push(ParsePower {
        value: InternalMpUint::from_limb(chunk_base),
        low_limbs: 0,
        exponent: 1,
    });
    // SAFETY: midpoint != 0 establishes levels >= 1.
    for bit in (0..unsafe { levels.unchecked_sub(1) }).rev() {
        let exponent = midpoint >> bit;
        // SAFETY: the initial push and every previous iteration leave a nonempty table.
        let last = unsafe { powers.last().unwrap_unchecked() };
        let mut value = last.value.square();
        // exponent = 2 * last.exponent + (exponent & 1).
        if exponent & 1 != 0 {
            let len = value.limbs().len();
            // SAFETY: len <= 2 * last.exponent; this odd exponent includes
            // one further chunk and is below the materialized chunk count.
            // Thus len + 1 fits usize; no failure branch is reachable here.
            let reserved = unsafe { len.checked_add(1).unwrap_unchecked() };
            let dst = value.prepare_limb_write(reserved).as_mut_ptr();
            // SAFETY: the reservation preserves the normalized initialized prefix
            // and supplies one exclusive aligned guard; add = 0 < chunk_base < B.
            let updated_len = unsafe { Convert::mul_small_add(dst, len, chunk_base, 0) };
            // SAFETY: the update initialized its normalized prefix in the fixed span.
            unsafe {
                value.set_len(updated_len);
            }
        }
        // v2(last.value) < W and v2(chunk_base) < W introduce at most two zero limbs.
        let stripped = value.limbs().iter().take_while(|&&limb| limb == 0).count();
        debug_assert!(
            stripped <= 2,
            "a binary-prefix step introduces at most two zero limbs"
        );
        // SAFETY: the normalized binary-prefix step strips at most two limbs.
        // LIMB_BITS <= 64 bounds the shift by 128 even on 16-bit targets.
        let shift = unsafe { stripped.checked_mul(LIMB_BITS).unwrap_unchecked() };
        if shift != 0 {
            value.shr_assign(shift);
        }
        debug_assert_ne!(
            value.limbs().first(),
            Some(&0),
            "the power factor is indivisible by B"
        );
        // SAFETY: chunk_base < B makes the power's whole-limb offset smaller
        // than its exponent, which is below the addressable chunk count. Both
        // exact checked operations succeed; infallible extraction removes their error path.
        let low_limbs = unsafe {
            last.low_limbs
                .checked_mul(2)
                .and_then(|offset| offset.checked_add(stripped))
                .unwrap_unchecked()
        };
        powers.push(ParsePower {
            value,
            low_limbs,
            exponent,
        });
    }
    powers
}

/// Evaluates `upper * chunk_base^mid + lower` in one initialized reservation.
fn combine_chunks(
    chunks: &[Limb],
    powers: &[ParsePower],
    chunk_base: Limb,
    leaf_chunks: usize,
    scratch: &mut MulScratch,
) -> InternalMpUint {
    if chunks.len() <= leaf_chunks {
        return reconstruct_leaf(chunks, chunk_base);
    }
    // SAFETY: the driver builds a nonempty table; children retain a nonempty
    // prefix or the final exponent-one power.
    let power = unsafe { powers.last().unwrap_unchecked() };
    let mid = power.exponent;
    // Each exponent is the floor half of its successor. Both child widths are
    // at least the preceding exponent; the final exponent one handles the tail.
    debug_assert!(
        mid != 0 && mid < chunks.len(),
        "the split is internal to its chunk block"
    );
    let child_powers = if powers.len() > 1 {
        // SAFETY: removing one entry leaves a nonempty in-bounds prefix.
        unsafe { powers.get_unchecked(..powers.len().unchecked_sub(1)) }
    } else {
        powers
    };
    // SAFETY: 0 < mid < chunks.len() gives two nonempty initialized child slices.
    let (lower, upper) = unsafe {
        (
            combine_chunks(
                chunks.get_unchecked(..mid),
                child_powers,
                chunk_base,
                leaf_chunks,
                scratch,
            ),
            combine_chunks(
                chunks.get_unchecked(mid..),
                child_powers,
                chunk_base,
                leaf_chunks,
                scratch,
            ),
        )
    };
    if upper.is_zero() {
        return lower;
    }
    let upper_limbs = upper.limbs();
    let factor = power.value.limbs();
    let lower_limbs = lower.limbs();
    let offset = power.low_limbs;
    // SAFETY: upper.len() <= chunks.len() - mid and factor.len() + offset
    // <= mid since chunk_base < B. Their total is at most the addressable
    // chunk count; the complete product reservation has no arithmetic failure path.
    let reserved = unsafe {
        upper_limbs
            .len()
            .checked_add(factor.len())
            .and_then(|width| width.checked_add(offset))
            .unwrap_unchecked()
    };
    debug_assert!(
        reserved <= chunks.len(),
        "radix chunks bound binary output width"
    );
    let mut result = InternalMpUint::zero();
    let len = {
        let mut write = result.prepare_limb_write(reserved);
        let dst = write.as_mut_ptr();
        let copied = lower_limbs.len().min(offset);
        // SAFETY: copied <= offset < reserved bounds the fresh disjoint prefix.
        // Copying and zero filling initialize the entire offset-limb prefix.
        unsafe {
            copy_nonoverlapping(lower_limbs.as_ptr(), dst, copied);
            write_bytes(dst.add(copied), 0, offset.unchecked_sub(copied));
        }
        // SAFETY: the aligned exclusive suffix has upper.len() + factor.len()
        // slots and is disjoint from both nonempty inputs. Multiplication writes
        // every MaybeUninit slot before returning initialized limbs.
        let product = unsafe {
            let output: &mut [MaybeUninit<Limb>] =
                from_raw_parts_mut(dst.add(offset).cast(), reserved.unchecked_sub(offset));
            Multiplication::mul_nonempty_distinct_into_uninit(upper_limbs, factor, output, scratch)
        };
        if lower_limbs.len() > offset {
            // SAFETY: offset is inside lower; lower < chunk_base^mid bounds its
            // suffix by factor.len(), strictly below the full product width.
            let tail = unsafe { lower_limbs.get_unchecked(offset..) };
            let carry = Addition::add_slice_in_place(product, tail);
            // SAFETY: tail.len() <= factor.len() < product.len().
            let escaped = Addition::propagate_carry(
                unsafe { product.get_unchecked_mut(tail.len()..) },
                carry,
            );
            // upper * power + lower < (upper + 1) * power < B^reserved.
            debug_assert_eq!(
                escaped, 0,
                "the reconstruction fits its complete product span"
            );
        }
        // SAFETY: normalized nonzero factors produce at least two initialized
        // limbs and at most one zero top limb. Addition cannot shorten the value.
        unsafe { reserved.unchecked_sub(usize::from(*product.last().unwrap_unchecked() == 0)) }
    };
    // SAFETY: the copied prefix and initialized product cover the reservation;
    // the top guard check proves the committed prefix is normalized.
    unsafe {
        result.set_len(len);
    }
    result
}

/// Accumulates one leaf with one reservation and one normalized length commit.
fn reconstruct_leaf(chunks: &[Limb], chunk_base: Limb) -> InternalMpUint {
    let mut width = chunks.len();
    while width != 0 {
        // SAFETY: 0 < width <= chunks.len() bounds the predecessor and top chunk.
        let top = unsafe { *chunks.get_unchecked(width.unchecked_sub(1)) };
        if top != 0 {
            break;
        }
        // SAFETY: the loop condition proves width > 0.
        width = unsafe { width.unchecked_sub(1) };
    }
    if width == 0 {
        return InternalMpUint::zero();
    }
    // SAFETY: the scan found a nonzero chunk at the in-bounds index width - 1.
    let top = unsafe { *chunks.get_unchecked(width.unchecked_sub(1)) };
    if width == 1 {
        return InternalMpUint::from_limb(top);
    }
    let mut inline = [0; INLINE_LIMBS + 2];
    let small = width <= inline.len();
    let mut result = InternalMpUint::zero();
    let dst = if small {
        inline.as_mut_ptr()
    } else {
        result.prepare_limb_write(width).as_mut_ptr()
    };
    // SAFETY: the exclusive destination reserves at least width >= 2 aligned limbs.
    unsafe {
        dst.write(top);
    }
    let mut length = NonZeroUsize::MIN;
    // SAFETY: width > 0 bounds this predecessor and initialized chunk prefix.
    for &chunk in unsafe { chunks.get_unchecked(..width.unchecked_sub(1)) }
        .iter()
        .rev()
    {
        // SAFETY: j chunks occupy at most j limbs since chunk_base < B.
        // Before each update j < width leaves one writable guard. Each chunk
        // is below chunk_base and the prefix is initialized and normalized.
        let updated_len = unsafe { Convert::mul_small_add(dst, length.get(), chunk_base, chunk) };
        // SAFETY: a nonzero top chunk and positive base keep every prefix nonzero.
        length = unsafe { NonZeroUsize::new_unchecked(updated_len) };
    }
    let len = length.get();
    if small {
        if len <= INLINE_LIMBS {
            return InternalMpUint::from_limbs_4(inline[0], inline[1], inline[2], inline[3]);
        }
        // SAFETY: len <= width <= inline.len(); the active prefix is initialized
        // and its final carry establishes a nonzero top limb.
        return unsafe {
            InternalMpUint::from_limbs_normalized(inline.get_unchecked(..len).to_vec())
        };
    }
    // SAFETY: the updates initialized a normalized prefix within the fixed reservation.
    unsafe {
        result.set_len(len);
    }
    result
}

impl Convert {
    /// Updates a normalized prefix to `dst * mul + add` and returns its length.
    ///
    /// # Safety
    /// `dst` is aligned, exclusive, and writable for `len + 1` limbs. The first
    /// `len` limbs are initialized and normalized; `0 <= add < mul < B`.
    #[inline]
    #[expect(
        clippy::as_conversions,
        reason = "Limb widens losslessly to DoubleLimb; narrowing extracts the low native half"
    )]
    #[cfg_attr(
        target_pointer_width = "32",
        expect(
            clippy::cast_possible_truncation,
            reason = "The low half of a double limb is reduced modulo the 32-bit limb base"
        )
    )]
    pub unsafe fn mul_small_add(dst: *mut Limb, len: usize, mul: Limb, add: Limb) -> usize {
        debug_assert!(
            add < mul,
            "the additive chunk is below the native radix base"
        );
        let mut carry = add;
        for index in 0..len {
            // SAFETY: index < len bounds an initialized exclusive limb.
            // carry < mul and x < B give x*mul + carry <= B*mul-1, fitting
            // DoubleLimb and preserving next carry < mul.
            unsafe {
                let product = (*dst.add(index) as DoubleLimb)
                    .unchecked_mul(mul as DoubleLimb)
                    .unchecked_add(carry as DoubleLimb);
                dst.add(index).write(product as Limb);
                carry = Limb::try_from(product >> LIMB_BITS).unwrap_unchecked();
            }
        }
        if carry == 0 {
            len
        } else {
            // SAFETY: the caller reserves the guard at len. Writing the nonzero
            // carry initializes the new top limb; the reserved len + 1 fits usize.
            unsafe {
                dst.add(len).write(carry);
                len.unchecked_add(1)
            }
        }
    }
}
