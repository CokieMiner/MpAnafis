//! Limb-slice primitives shared by the division kernels.
//!
//! Odd modular residues, normalization, and carry/borrow
//! propagation operate directly on initialized limb spans.

#![expect(
    unsafe_code,
    reason = "odd divisors and initialized limb spans establish modular residues, carry bounds, and output initialization"
)]

use core::{num::NonZeroUsize, ops::Rem, ptr::copy_nonoverlapping};

use super::{ArchKernels, Division, Limb, ScratchBuffer};

#[cfg(not(any(
    target_pointer_width = "16",
    target_pointer_width = "32",
    target_pointer_width = "64"
)))]
compile_error!("Modular inverse iterations must be updated for >64-bit platforms");

/// Multiplicative inverse of `2*i + 1` modulo B for `i in 0..128`.
/// Full-limb entries resolve byte-sized odd divisors directly. Their low eight
/// bits also seed Newton lifting for arbitrary divisors with the same low byte.
#[expect(
    clippy::indexing_slicing,
    reason = "the const-evaluation loop checks index < table.len() before both accesses"
)]
const INVERT_LIMB_TABLE: [Limb; 128] = {
    let mut table: [Limb; 128] = [
        0x01, 0xab, 0xcd, 0xb7, 0x39, 0xa3, 0xc5, 0xef, 0xf1, 0x1b, 0x3d, 0xa7, 0x29, 0x13, 0x35,
        0xdf, 0xe1, 0x8b, 0xad, 0x97, 0x19, 0x83, 0xa5, 0xcf, 0xd1, 0xfb, 0x1d, 0x87, 0x09, 0xf3,
        0x15, 0xbf, 0xc1, 0x6b, 0x8d, 0x77, 0xf9, 0x63, 0x85, 0xaf, 0xb1, 0xdb, 0xfd, 0x67, 0xe9,
        0xd3, 0xf5, 0x9f, 0xa1, 0x4b, 0x6d, 0x57, 0xd9, 0x43, 0x65, 0x8f, 0x91, 0xbb, 0xdd, 0x47,
        0xc9, 0xb3, 0xd5, 0x7f, 0x81, 0x2b, 0x4d, 0x37, 0xb9, 0x23, 0x45, 0x6f, 0x71, 0x9b, 0xbd,
        0x27, 0xa9, 0x93, 0xb5, 0x5f, 0x61, 0x0b, 0x2d, 0x17, 0x99, 0x03, 0x25, 0x4f, 0x51, 0x7b,
        0x9d, 0x07, 0x89, 0x73, 0x95, 0x3f, 0x41, 0xeb, 0x0d, 0xf7, 0x79, 0xe3, 0x05, 0x2f, 0x31,
        0x5b, 0x7d, 0xe7, 0x69, 0x53, 0x75, 0x1f, 0x21, 0xcb, 0xed, 0xd7, 0x59, 0xc3, 0xe5, 0x0f,
        0x11, 0x3b, 0x5d, 0xc7, 0x49, 0x33, 0x55, 0xff,
    ];
    let mut index = 0_usize;
    while index < table.len() {
        let odd = (index << 1) | 1;
        let mut inverse = table[index];
        let mut bits = 8;
        while bits < Limb::BITS {
            inverse = inverse.wrapping_mul(2_usize.wrapping_sub(odd.wrapping_mul(inverse)));
            bits <<= 1;
        }
        table[index] = inverse;
        // SAFETY: index < table.len() == 128, so index + 1 fits usize on every target.
        index = unsafe { index.unchecked_add(1) };
    }
    table
};

impl Division {
    /// Computes the modular multiplicative inverse of an odd limb modulo `2^LIMB_BITS`.
    ///
    /// The table contains full inverses for odd divisors below 256. For larger
    /// divisors its matching low byte seeds eight correct bits. Each Newton iteration
    /// `y <- y * (2 - x*y)` doubles the number of correct low bits:
    /// - 1 iteration achieves 16 bits (covers 16-bit targets)
    /// - 2 iterations achieve 32 bits (covers 32-bit targets)
    /// - 3 iterations achieve 64 bits (covers 64-bit targets)
    ///
    /// # Preconditions
    ///
    /// `odd` must be odd.
    #[must_use]
    #[inline]
    pub fn modular_inverse_limb(odd: Limb) -> Limb {
        debug_assert_eq!(odd & 1, 1, "the modular inverse seed must be odd");
        let index = (odd >> 1) & 0x7F;
        // SAFETY: index is bounded by 0x7F < 128 = INVERT_LIMB_TABLE.len().
        let mut inv = unsafe { *INVERT_LIMB_TABLE.get_unchecked(index) };
        if odd < 256 {
            return inv;
        }
        inv = inv.wrapping_mul(2_usize.wrapping_sub(odd.wrapping_mul(inv)));
        #[cfg(not(target_pointer_width = "16"))]
        {
            inv = inv.wrapping_mul(2_usize.wrapping_sub(odd.wrapping_mul(inv)));
        }
        #[cfg(not(any(target_pointer_width = "16", target_pointer_width = "32")))]
        {
            inv = inv.wrapping_mul(2_usize.wrapping_sub(odd.wrapping_mul(inv)));
        }
        inv
    }

    /// Computes the exact modular residue `r` satisfying `r * B^k == -A (mod odd_d)`
    /// where `0 <= r < odd_d`.
    ///
    /// A single input limb uses `k = 0`. For longer inputs, `k` is the
    /// number of input limbs minus one when the final limb is at most the
    /// divisor; otherwise it is the full input length.
    ///
    /// For an odd divisor `d`, `gcd(B^k, d) = 1`, so `gcd(r, d) == gcd(A, d)`.
    /// In addition, `A` is divisible by `odd_d` if and only if `r == 0`.
    ///
    /// This kernel operates from least to most significant limb using only
    /// one 1-limb multiply, one high product (`mulhi`), and one subtract per limb,
    /// avoiding all hardware division instructions.
    ///
    /// # Preconditions
    ///
    /// `odd_d` must be odd.
    #[must_use]
    pub fn modexact_1_odd(num_limbs: &[Limb], odd_d: Limb) -> Limb {
        debug_assert_eq!(odd_d & 1, 1, "modexact_1_odd requires an odd divisor");
        if num_limbs.is_empty() || odd_d == 1 {
            return 0;
        }
        let len = num_limbs.len();
        if len == 1 {
            // SAFETY: len == 1 proves index 0 exists.
            let a0 = unsafe { *num_limbs.get_unchecked(0) };
            let rem = if a0 < odd_d {
                a0
            } else {
                // SAFETY: odd_d is odd and odd_d != 1, so odd_d >= 3 > 0.
                unsafe { Rem::rem(a0, NonZeroUsize::new_unchecked(odd_d)) }
            };
            return if rem == 0 {
                0
            } else {
                // SAFETY: scalar remainder satisfies 0 < rem < odd_d.
                unsafe { odd_d.unchecked_sub(rem) }
            };
        }

        let d_inv = Self::modular_inverse_limb(odd_d);
        // SAFETY: len >= 2 proves split_last succeeds.
        let (last, init) = unsafe { num_limbs.split_last().unwrap_unchecked() };
        let mut carry: Limb = 0;
        for &x in init {
            let (diff, borrow) = x.overflowing_sub(carry);
            let q = diff.wrapping_mul(d_inv);
            let (_, hi) = ArchKernels::mul_limb_lo_hi(q, odd_d);
            // q * odd_d <= (B - 1) * odd_d gives hi <= odd_d - 1. Equality forces
            // the low product <= B - odd_d, whereas a borrow forces diff >= B - odd_d + 1.
            // SAFETY: the incoming carry is below odd_d. The incompatible
            // equality and borrow conditions give hi+borrow < odd_d <= Limb::MAX.
            carry = unsafe { hi.unchecked_add(Limb::from(borrow)) };
        }

        let last_val = *last;
        if last_val <= odd_d {
            let (sub, underflow) = carry.overflowing_sub(last_val);
            // The signed difference lies in [-odd_d, odd_d). Adding odd_d
            // to a negative difference wraps modulo B into [0, odd_d).
            sub.wrapping_add(if underflow { odd_d } else { 0 })
        } else {
            let (diff, borrow) = last_val.overflowing_sub(carry);
            let q = diff.wrapping_mul(d_inv);
            let (_, hi) = ArchKernels::mul_limb_lo_hi(q, odd_d);
            // SAFETY: the same cancellation bound gives hi+borrow < odd_d.
            unsafe { hi.unchecked_add(Limb::from(borrow)) }
        }
    }

    /// Shifts a limb slice left into `out`, preserving its input width.
    /// With `GUARD`, one high carry limb is always written, including zero.
    /// Otherwise only a nonzero carry is appended. Zero-padded prefixes and
    /// all-zero input are valid. Existing scratch capacity is reused; insufficient
    /// storage is acquired from the pool before any output limb is written.
    ///
    /// `shift` must be less than [`LIMB_BITS`].
    pub fn shift_limbs_left<const GUARD: bool>(
        limbs: &[Limb],
        shift: u32,
        out: &mut ScratchBuffer,
    ) {
        let len = limbs.len();
        if shift == 0 || len == 0 {
            // SAFETY: a materialized Limb slice plus one guard fits usize on
            // every pointer width. Reservation supplies final_len writable
            // limbs disjoint from the initialized input. The copy and optional
            // guard write initialize every element before its length is set.
            unsafe {
                let final_len = len.unchecked_add(usize::from(GUARD));
                out.reset_with_capacity(final_len);
                let destination = out.as_mut_ptr();
                copy_nonoverlapping(limbs.as_ptr(), destination, len);
                if GUARD {
                    destination.add(len).write(0);
                }
                out.set_len(final_len);
            }
            return;
        }

        debug_assert!(shift < Limb::BITS, "normalization shift must be sub-limb");
        // SAFETY: this branch has len > 0 and 0 < shift < Limb::BITS.
        // A materialized Limb slice plus one carry slot fits usize. Guard
        // admission fixes the extra slot; otherwise the leading limb bounds it.
        let final_len = unsafe {
            let carry_shift = Limb::BITS.unchecked_sub(shift);
            let extra = usize::from(GUARD || *limbs.last().unwrap_unchecked() >> carry_shift != 0);
            len.unchecked_add(extra)
        };
        out.reset_with_capacity(final_len);
        let dest_ptr = out.as_mut_ptr();
        // SAFETY: len>0 and 0<shift<Limb::BITS. Reservation supplies len
        // aligned writable limbs disjoint from the initialized source. The
        // architecture facade selects a valid backend, which initializes
        // every destination limb and returns the outgoing high carry.
        let carry =
            unsafe { ArchKernels::lshift_into_unchecked(dest_ptr, limbs.as_ptr(), len, shift) };

        if final_len != len {
            // SAFETY: guard admission or a nonzero high carry reserved this
            // extra slot after the len outputs; the shift kernel computed it.
            unsafe {
                dest_ptr.add(len).write(carry);
            }
        }

        // SAFETY: every element below final_len was initialized above.
        unsafe {
            out.set_len(final_len);
        }
    }
}
