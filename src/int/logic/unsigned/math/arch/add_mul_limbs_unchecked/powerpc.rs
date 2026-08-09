//! PowerPC 32-bit architecture-specific limb operations.
//!
//! Uses `mullw`/`mulhwu` for efficient 32x32->64 multiplication and
//! `addc`/`addze` for carry tracking.

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
/// This is the `PowerPC` 32-bit inline assembly implementation utilizing the `mullw` and
/// `mulhwu` instructions for 32x32->64-bit multiplication, and `addc`/`addze` for
/// accumulation and carry propagation.
///
/// The loop is **4-way unrolled** for optimal performance, utilizing the CTR register
/// for zero-overhead loop control.
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
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: Caller guarantees dst and src are valid for `len` elements.
    unsafe {
        asm!(
            "cmpwi {chunks}, 0",
            "beq 1f",                     // skip chunk loop if chunks == 0
            "mtctr {chunks}",             // CTR = chunks

            ".p2align 4",
            "2:",                         // --- Unrolled Loop x4 ---

            // Load all four limbs
            "lwz {src_v0}, 0({src})",
            "lwz {src_v1}, 4({src})",
            "lwz {src_v2}, 8({src})",
            "lwz {src_v3}, 12({src})",
            "lwz {dst_v0}, 0({dst})",
            "lwz {dst_v1}, 4({dst})",
            "lwz {dst_v2}, 8({dst})",
            "lwz {dst_v3}, 12({dst})",

            // Hoist all eight multiplies
            "mullw {p_lo0}, {src_v0}, {scalar}",
            "mulhwu {p_hi0}, {src_v0}, {scalar}",
            "mullw {p_lo1}, {src_v1}, {scalar}",
            "mulhwu {p_hi1}, {src_v1}, {scalar}",
            "mullw {p_lo2}, {src_v2}, {scalar}",
            "mulhwu {p_hi2}, {src_v2}, {scalar}",
            "mullw {p_lo3}, {src_v3}, {scalar}",
            "mulhwu {p_hi3}, {src_v3}, {scalar}",

            // Strictly serial CA chain — products already in flight
            "addc {p_lo0}, {p_lo0}, {carry}",
            "addze {p_hi0}, {p_hi0}",
            "addc {dst_v0}, {dst_v0}, {p_lo0}",
            "addze {carry}, {p_hi0}",
            "stw {dst_v0}, 0({dst})",

            "addc {p_lo1}, {p_lo1}, {carry}",
            "addze {p_hi1}, {p_hi1}",
            "addc {dst_v1}, {dst_v1}, {p_lo1}",
            "addze {carry}, {p_hi1}",
            "stw {dst_v1}, 4({dst})",

            "addc {p_lo2}, {p_lo2}, {carry}",
            "addze {p_hi2}, {p_hi2}",
            "addc {dst_v2}, {dst_v2}, {p_lo2}",
            "addze {carry}, {p_hi2}",
            "stw {dst_v2}, 8({dst})",

            "addc {p_lo3}, {p_lo3}, {carry}",
            "addze {p_hi3}, {p_hi3}",
            "addc {dst_v3}, {dst_v3}, {p_lo3}",
            "addze {carry}, {p_hi3}",
            "stw {dst_v3}, 12({dst})",

            "addi {src}, {src}, 16",
            "addi {dst}, {dst}, 16",
            "bdnz 2b",                    // --CTR; loop if CTR != 0

            "1:",                         // --- Remainder Loop ---
            "cmpwi {rem}, 0",
            "beq 3f",                     // skip tail if rem == 0
            "mtctr {rem}",                // CTR = rem
            "addi {src}, {src}, -4",
            "addi {dst}, {dst}, -4",

            ".p2align 4",
            "4:",
            "lwzu {src_v0}, 4({src})",
            "lwzu {dst_v0}, 4({dst})",
            "mullw {p_lo0}, {src_v0}, {scalar}",
            "mulhwu {p_hi0}, {src_v0}, {scalar}",
            "addc {p_lo0}, {p_lo0}, {carry}",
            "addze {p_hi0}, {p_hi0}",
            "addc {dst_v0}, {dst_v0}, {p_lo0}",
            "addze {carry}, {p_hi0}",
            "stw {dst_v0}, 0({dst})",

            "bdnz 4b",

            "3:",                         // --- End ---

            carry = inout(reg) carry,
            dst = inout(reg_nonzero) dst => _,
            src = inout(reg_nonzero) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            scalar = in(reg) scalar,
            src_v0 = out(reg) _, src_v1 = out(reg) _, src_v2 = out(reg) _, src_v3 = out(reg) _,
            dst_v0 = out(reg) _, dst_v1 = out(reg) _, dst_v2 = out(reg) _, dst_v3 = out(reg) _,
            p_lo0 = out(reg) _, p_lo1 = out(reg) _, p_lo2 = out(reg) _, p_lo3 = out(reg) _,
            p_hi0 = out(reg) _, p_hi1 = out(reg) _, p_hi2 = out(reg) _, p_hi3 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    carry
}
