//! `AArch64` multiply-add limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// This computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// This is the `AArch64` inline assembly implementation. It uses `mul` and `umulh`
/// instructions to compute the 128-bit product, and `adds`/`adcs` for accumulation.
///
/// The loop processes 2 limbs per iteration utilizing instruction-level parallelism.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(clippy::inline_always, reason = "Performance critical inner loop")]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry: Limb = 0;
    let chunks = len >> 1;
    let rem = len & 1;
    // SAFETY: Caller guarantees dst and src are valid for `len` elements
    unsafe {
        asm!(
            "cbz {chunks}, 1f",
            "2:",
            // Load 2 source values and 2 dest values
            "ldp {src_val0}, {src_val1}, [{src}], #16",
            "ldp {dst_val0}, {dst_val1}, [{dst}]",
            // Process first limb
            "mul {p_lo0}, {src_val0}, {scalar}",
            "umulh {p_hi0}, {src_val0}, {scalar}",
            "adds {p_lo0}, {p_lo0}, {carry}",
            "adc {p_hi0}, {p_hi0}, xzr",
            "adds {dst_val0}, {dst_val0}, {p_lo0}",
            "adc {carry}, {p_hi0}, xzr",
            // Process second limb
            "mul {p_lo1}, {src_val1}, {scalar}",
            "umulh {p_hi1}, {src_val1}, {scalar}",
            "adds {p_lo1}, {p_lo1}, {carry}",
            "adc {p_hi1}, {p_hi1}, xzr",
            "adds {dst_val1}, {dst_val1}, {p_lo1}",
            "adc {carry}, {p_hi1}, xzr",
            // Store 2 results
            "stp {dst_val0}, {dst_val1}, [{dst}], #16",
            "sub {chunks}, {chunks}, #1",
            "cbnz {chunks}, 2b",
            // Handle remainder
            "1:",
            "cbz {rem}, 3f",
            "ldr {src_val0}, [{src}], #8",
            "ldr {dst_val0}, [{dst}]",
            "mul {p_lo0}, {src_val0}, {scalar}",
            "umulh {p_hi0}, {src_val0}, {scalar}",
            "adds {p_lo0}, {p_lo0}, {carry}",
            "adc {p_hi0}, {p_hi0}, xzr",
            "adds {dst_val0}, {dst_val0}, {p_lo0}",
            "adc {carry}, {p_hi0}, xzr",
            "str {dst_val0}, [{dst}], #8",
            "3:",
            carry = inout(reg) carry,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            scalar = in(reg) scalar,
            src_val0 = out(reg) _,
            src_val1 = out(reg) _,
            dst_val0 = out(reg) _,
            dst_val1 = out(reg) _,
            p_lo0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi0 = out(reg) _,
            p_hi1 = out(reg) _,
            options(nostack)
        );
    }
    carry
}
