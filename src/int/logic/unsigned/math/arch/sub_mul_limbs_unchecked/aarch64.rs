//! `AArch64` multiply-subtract limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// This computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// This is the `AArch64` inline assembly implementation. It processes 2 limbs
/// per iteration utilizing `mul` and `umulh` along with dual accumulation chains
/// for the product and subtraction.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(clippy::inline_always, reason = "Performance critical inner loop")]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry_hi: Limb = 0;
    let mut borrow: Limb = 0;
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
            "adds {p_lo0}, {p_lo0}, {carry_hi}",
            "adc {p_hi0}, {p_hi0}, xzr",
            "subs {dst_val0}, {dst_val0}, {p_lo0}",
            "cset {b1}, cc",
            "subs {dst_val0}, {dst_val0}, {borrow}",
            "cset {b2}, cc",
            "orr {borrow}, {b1}, {b2}",
            "mov {carry_hi}, {p_hi0}",
            // Process second limb
            "mul {p_lo1}, {src_val1}, {scalar}",
            "umulh {p_hi1}, {src_val1}, {scalar}",
            "adds {p_lo1}, {p_lo1}, {carry_hi}",
            "adc {p_hi1}, {p_hi1}, xzr",
            "subs {dst_val1}, {dst_val1}, {p_lo1}",
            "cset {b1}, cc",
            "subs {dst_val1}, {dst_val1}, {borrow}",
            "cset {b2}, cc",
            "orr {borrow}, {b1}, {b2}",
            "mov {carry_hi}, {p_hi1}",
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
            "adds {p_lo0}, {p_lo0}, {carry_hi}",
            "adc {p_hi0}, {p_hi0}, xzr",
            "subs {dst_val0}, {dst_val0}, {p_lo0}",
            "cset {b1}, cc",
            "subs {dst_val0}, {dst_val0}, {borrow}",
            "cset {b2}, cc",
            "orr {borrow}, {b1}, {b2}",
            "mov {carry_hi}, {p_hi0}",
            "str {dst_val0}, [{dst}], #8",
            "3:",
            carry_hi = inout(reg) carry_hi,
            borrow = inout(reg) borrow,
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
            b1 = out(reg) _,
            b2 = out(reg) _,
            options(nostack)
        );
    }
    (carry_hi, borrow)
}
