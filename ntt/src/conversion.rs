//! Native-limb conversion with a complete-coefficient path and one zero-extended tail.

#![expect(
    unsafe_code,
    reason = "Admitted coefficient geometry proves bit positions, native input reads, and exclusive output spans"
)]

use super::{LIMB_BITS, Limb, Prime};

/// Native-limb coefficient conversion.
#[derive(Debug)]
pub struct Conversion;

struct Packing<'input> {
    limbs: &'input [Limb],
    prime: &'input Prime,
    width: usize,
    words: usize,
    high_bits: usize,
}

impl Conversion {
    /// Packs admitted binary chunks and initializes only padding read by the first DIF stage.
    pub fn pack(buffer: &mut [f64], limbs: &[Limb], prime: &Prime, chunk_bits: u32, digits: usize) {
        // SAFETY: the immutable native plan admits chunk_bits > 0, its usize
        // conversion, digits <= buffer.len(), and limbs.len()*LIMB_BITS. Each
        // complete coefficient ends inside the source; only the final partial
        // coefficient requires zero-extended source reads. The first DIF stage
        // reads a quarter, a half, or the full input; read_len lies in
        // digits..=buffer.len(), and the omitted tail is overwritten before use.
        unsafe {
            let width = usize::try_from(chunk_bits).unwrap_unchecked();
            let words = width.div_ceil(64);
            let packing = Packing {
                limbs,
                prime,
                width,
                words,
                high_bits: width.unchecked_sub(words.unchecked_sub(1) << 6),
            };
            let complete = limbs
                .len()
                .unchecked_mul(LIMB_BITS)
                .div_euclid(width)
                .min(digits);
            let len = buffer.len();
            let first_span = if len.trailing_zeros() & 1 != 0 {
                len >> 1
            } else {
                len >> 2
            };
            let read_len = if digits <= first_span {
                first_span
            } else if digits <= len >> 1 {
                len >> 1
            } else {
                len
            };
            let (active, rest) = buffer.split_at_mut_unchecked(digits);
            // Sparse radix-2 overwrites its upper half. Sparse radix-4
            // overwrites two or three quarters without reading them. Only
            // digits..read_len needs physical zeros before the first stage.
            let padding = rest.get_unchecked_mut(..read_len.unchecked_sub(digits));
            let (full, tail) = active.split_at_mut_unchecked(complete);
            pack_digits::<true>(full, &packing, 0);
            pack_digits::<false>(tail, &packing, complete.unchecked_mul(width));
            padding.fill(0.0);
        }
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Each reconstructed residue is canonical below 2^50 and exactly representable in f64"
)]
unsafe fn pack_digits<const COMPLETE: bool>(
    output: &mut [f64],
    packing: &Packing<'_>,
    mut bit_pos: usize,
) {
    // SAFETY: the caller derives every coefficient position from an admitted
    // native plan. Complete chunks are entirely in bounds; the tail uses zero
    // extension. Canonical reductions remain below p, so Horner products use
    // canonical factors and each sum is below 2*p < 2^51.
    unsafe {
        for value in output {
            let high_pos = bit_pos.unchecked_add(packing.words.unchecked_sub(1) << 6);
            let high = extract_word::<COMPLETE>(packing.limbs, high_pos, packing.high_bits);
            let high_mask = u64::MAX >> 64_usize.unchecked_sub(packing.high_bits);
            let mut residue = packing.prime.reduce_word(high & high_mask);
            for word_index in (0..packing.words.unchecked_sub(1)).rev() {
                let position = bit_pos.unchecked_add(word_index << 6);
                let word = extract_word::<COMPLETE>(packing.limbs, position, 64);
                let product = packing
                    .prime
                    .mulmod_fast(residue, packing.prime.word_radix());
                let sum = product.unchecked_add(packing.prime.reduce_word(word));
                let correction = packing
                    .prime
                    .modulus()
                    .unchecked_mul(u64::from(sum >= packing.prime.modulus()));
                residue = sum.unchecked_sub(correction);
            }
            *value = residue as f64;
            bit_pos = bit_pos.unchecked_add(packing.width);
        }
    }
}

#[expect(
    clippy::as_conversions,
    reason = "Native limbs have at most 64 bits, so widening to u64 is exact"
)]
unsafe fn extract_word<const COMPLETE: bool>(limbs: &[Limb], bit_pos: usize, bits: usize) -> u64 {
    let index = bit_pos.div_euclid(LIMB_BITS);
    let offset = bit_pos % LIMB_BITS;
    // SAFETY: bits is in 1..=64. In the complete specialization the caller
    // proves bit_pos+bits <= source_bits, which bounds every contributing limb.
    // The partial specialization checks reads and supplies zero outside the
    // source. Required shifts are strictly below 64; admitted positions bound
    // all index additions on every native pointer width.
    unsafe {
        let first = if COMPLETE {
            *limbs.get_unchecked(index)
        } else {
            limbs.get(index).copied().unwrap_or(0)
        };
        let mut value = (first as u64) >> offset;
        let pieces = offset.unchecked_add(bits).div_ceil(LIMB_BITS);
        for piece in 1..pieces {
            let source = index.unchecked_add(piece);
            let limb = if COMPLETE {
                *limbs.get_unchecked(source)
            } else {
                limbs.get(source).copied().unwrap_or(0)
            };
            let shift = piece.unchecked_mul(LIMB_BITS).unchecked_sub(offset);
            value |= (limb as u64).unchecked_shl(u32::try_from(shift).unwrap_unchecked());
        }
        value
    }
}
