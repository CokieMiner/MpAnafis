//! 3-Prime Garner CRT Reconstruction for 50-bit Floating-Point Harvey NTT.
//!
//! Reconstructs exact convolution coefficients up to 150 bits from three 50-bit
//! residues using Garner's algorithm with architecture-accelerated division kernels,
//! propagating carries directly into the destination native limb buffer.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer reconstruction into validated destination limbs"
)]

use super::{ArchKernels, FLOAT_PRIME_1, FLOAT_PRIME_2, FLOAT_PRIME_3, LIMB_BITS, Limb, Ntt};

// Precomputed Garner constants:
// c12 = inv(p1) mod p2
// c13 = inv(p1) mod p3
// c23 = inv(p2) mod p3
const C12: u128 = 651_790_492_945_564;
const C13: u128 = 364_158_251_119_407;
const C23: u128 = 523_477_485_984_149;

const P1: u128 = 1_108_307_720_798_209;
const P2: u128 = 1_086_317_488_242_689;
const P3: u128 = 910_395_627_798_529;

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "50-bit prime moduli fit in native Limb on supported pointer widths"
)]
const P2_LIMB: Limb = P2 as Limb;
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "50-bit prime moduli fit in native Limb on supported pointer widths"
)]
const P3_LIMB: Limb = P3 as Limb;

impl Ntt {
    /// Reconstructs 3-prime residues into native destination limbs.
    ///
    /// # Safety
    /// - `dst` is valid and initialized for writing at least `dst.len()` limbs.
    /// - `r1`, `r2`, and `r3` each contain at least `len` elements.
    #[allow(
        clippy::too_many_lines,
        reason = "Unrolled Garner CRT reconstruction loop and carry flush"
    )]
    pub unsafe fn reconstruct_into_limbs(
        dst: &mut [Limb],
        r1: &[f64],
        r2: &[f64],
        r3: &[f64],
        len: usize,
        digit_bits: u32,
    ) {
        let digit_mask = (1_u128 << digit_bits).wrapping_sub(1);
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS <= 64 fits in u32"
        )]
        let limb_bits_u32 = LIMB_BITS as u32;
        let mut carry = 0_u128;
        let mut accum_bits = 0_u32;
        let mut accum_val = 0_u128;
        let mut dst_idx = 0_usize;
        let dst_len = dst.len();

        let r1_ptr = r1.as_ptr();
        let r2_ptr = r2.as_ptr();
        let r3_ptr = r3.as_ptr();
        let dst_ptr = dst.as_mut_ptr();

        for i in 0..len {
            // SAFETY: caller establishes slice lengths >= len.
            let (f1, f2, f3) = unsafe { (*r1_ptr.add(i), *r2_ptr.add(i), *r3_ptr.add(i)) };

            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "Values are in [-p, p], normalized to [0, p)"
            )]
            let (u1, u2, u3) = {
                let n1 = if f1 < 0.0 { f1 + FLOAT_PRIME_1 } else { f1 };
                let n2 = if f2 < 0.0 { f2 + FLOAT_PRIME_2 } else { f2 };
                let n3 = if f3 < 0.0 { f3 + FLOAT_PRIME_3 } else { f3 };
                let raw1 = (n1 + 0.5) as u128;
                let raw2 = (n2 + 0.5) as u128;
                let raw3 = (n3 + 0.5) as u128;
                (
                    if raw1 >= P1 {
                        raw1.wrapping_sub(P1)
                    } else {
                        raw1
                    },
                    if raw2 >= P2 {
                        raw2.wrapping_sub(P2)
                    } else {
                        raw2
                    },
                    if raw3 >= P3 {
                        raw3.wrapping_sub(P3)
                    } else {
                        raw3
                    },
                )
            };

            // Garner's Algorithm (Hardware-Accelerated Modulo):
            // x1 = u1
            // x2 = ((u2 - x1) * c12) mod p2
            // x3 = ((((u3 - x1) * c13) mod p3 - x2) * c23) mod p3
            // X  = x1 + p1 * (x2 + p2 * x3)
            let x1 = u1;
            let x1_mod_p2 = if x1 >= P2 { x1.wrapping_sub(P2) } else { x1 };
            let diff2 = if u2 >= x1_mod_p2 {
                u2.wrapping_sub(x1_mod_p2)
            } else {
                u2.wrapping_add(P2).wrapping_sub(x1_mod_p2)
            };
            #[allow(
                clippy::as_conversions,
                clippy::cast_lossless,
                reason = "Limb remainder widening to u128"
            )]
            // SAFETY: diff2 * C12 < 2^100, (diff2 * C12 >> 64) < 2^36 < P2_LIMB, and P2_LIMB != 0.
            let x2 = unsafe { ArchKernels::rem_u128_unchecked(diff2.wrapping_mul(C12), P2_LIMB) }
                as u128;

            let x1_mod_p3 = if x1 >= P3 { x1.wrapping_sub(P3) } else { x1 };
            let diff3 = if u3 >= x1_mod_p3 {
                u3.wrapping_sub(x1_mod_p3)
            } else {
                u3.wrapping_add(P3).wrapping_sub(x1_mod_p3)
            };
            #[allow(
                clippy::as_conversions,
                clippy::cast_lossless,
                reason = "Limb remainder widening to u128"
            )]
            // SAFETY: diff3 * C13 < 2^100, (diff3 * C13 >> 64) < 2^36 < P3_LIMB, and P3_LIMB != 0.
            let step3 = unsafe { ArchKernels::rem_u128_unchecked(diff3.wrapping_mul(C13), P3_LIMB) }
                as u128;
            let x2_mod_p3 = if x2 >= P3 { x2.wrapping_sub(P3) } else { x2 };
            let diff3_2 = if step3 >= x2_mod_p3 {
                step3.wrapping_sub(x2_mod_p3)
            } else {
                step3.wrapping_add(P3).wrapping_sub(x2_mod_p3)
            };
            #[allow(
                clippy::as_conversions,
                clippy::cast_lossless,
                reason = "Limb remainder widening to u128"
            )]
            // SAFETY: diff3_2 * C23 < 2^100, (diff3_2 * C23 >> 64) < 2^36 < P3_LIMB, and P3_LIMB != 0.
            let x3 = unsafe { ArchKernels::rem_u128_unchecked(diff3_2.wrapping_mul(C23), P3_LIMB) }
                as u128;

            // Exact combined coefficient: X = x1 + P1 * (x2 + P2 * x3)
            // Note: x2 + P2 * x3 <= 101 bits, which fits in u128.
            let t = x2.wrapping_add(P2.wrapping_mul(x3));
            let t0 = t & digit_mask;
            let t1 = (t >> digit_bits) & digit_mask;
            let t2 = t >> (digit_bits.wrapping_mul(2));

            // term0 = x1 + P1 * t0 + carry <= 102 bits (fits in u128)
            let term0 = x1.wrapping_add(P1.wrapping_mul(t0)).wrapping_add(carry);
            let digit_val = term0 & digit_mask;
            carry = (term0 >> digit_bits)
                .wrapping_add(P1.wrapping_mul(t1))
                .wrapping_add(P1.wrapping_mul(t2) << digit_bits);

            accum_val |= digit_val << accum_bits;
            accum_bits = accum_bits.wrapping_add(digit_bits);

            while accum_bits >= limb_bits_u32 && dst_idx < dst_len {
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "Native limb extraction"
                )]
                let limb_val = accum_val as Limb;
                // SAFETY: dst_idx < dst_len verified.
                unsafe {
                    *dst_ptr.add(dst_idx) = limb_val;
                }
                dst_idx = dst_idx.wrapping_add(1);
                accum_val >>= LIMB_BITS;
                accum_bits = accum_bits.wrapping_sub(limb_bits_u32);
            }
        }

        while carry > 0 && dst_idx < dst_len {
            let digit_val = carry & digit_mask;
            accum_val |= digit_val << accum_bits;
            accum_bits = accum_bits.wrapping_add(digit_bits);
            carry >>= digit_bits;

            while accum_bits >= limb_bits_u32 && dst_idx < dst_len {
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "Native limb extraction"
                )]
                let limb_val = accum_val as Limb;
                // SAFETY: dst_idx < dst_len verified.
                unsafe {
                    *dst_ptr.add(dst_idx) = limb_val;
                }
                dst_idx = dst_idx.wrapping_add(1);
                accum_val >>= LIMB_BITS;
                accum_bits = accum_bits.wrapping_sub(limb_bits_u32);
            }
        }

        while accum_bits > 0 && dst_idx < dst_len {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "Tail limb extraction"
            )]
            let tail_val = accum_val as Limb;
            // SAFETY: dst_idx < dst_len verified.
            unsafe {
                *dst_ptr.add(dst_idx) = tail_val;
            }
            dst_idx = dst_idx.wrapping_add(1);
            accum_val >>= LIMB_BITS;
            accum_bits = accum_bits.saturating_sub(limb_bits_u32);
        }

        // SAFETY: every write guard above maintains dst_idx <= dst.len(), so the
        // padding range is fully in bounds. `fill` lowers to a vectorized
        // memset instead of a scalar store loop.
        unsafe {
            dst.get_unchecked_mut(dst_idx..).fill(0);
        }
    }
}
