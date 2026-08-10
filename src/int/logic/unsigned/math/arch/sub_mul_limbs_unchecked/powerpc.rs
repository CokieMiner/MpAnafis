//! PowerPC 32-bit subtraction kernels (inline assembly).
//!
//! Uses the XER[CA] (carry) bit for borrow propagation.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from `dst`,
/// and return the final borrow.
///
/// This computes:
///
/// ```text
///   (borrow, dst[0..len]) = dst[0..len] - (src[0..len] × scalar) - borrow
/// ```
///
/// This is the `PowerPC` 32-bit inline assembly implementation utilizing `mullw`/`mulhwu`
/// for multiplication. For subtraction, it uses `subfc` and propagates the subtraction
/// borrow (using `subfe` and `subf`) smoothly across the high product.
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
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry_hi: Limb = 0;
    let mut borrow_reg: Limb = 0;
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

            // Multiply carry chain + subtraction borrow chain (mask form)
            "addc {p_lo0}, {p_lo0}, {carry_hi}",
            "addze {carry_hi}, {p_hi0}",
            "subfic {temp}, {borrow_reg}, 0",
            "subfe {dst_v0}, {p_lo0}, {dst_v0}",
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}",
            "stw {dst_v0}, 0({dst})",

            "addc {p_lo1}, {p_lo1}, {carry_hi}",
            "addze {carry_hi}, {p_hi1}",
            "subfic {temp}, {borrow_reg}, 0",
            "subfe {dst_v1}, {p_lo1}, {dst_v1}",
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}",
            "stw {dst_v1}, 4({dst})",

            "addc {p_lo2}, {p_lo2}, {carry_hi}",
            "addze {carry_hi}, {p_hi2}",
            "subfic {temp}, {borrow_reg}, 0",
            "subfe {dst_v2}, {p_lo2}, {dst_v2}",
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}",
            "stw {dst_v2}, 8({dst})",

            "addc {p_lo3}, {p_lo3}, {carry_hi}",
            "addze {carry_hi}, {p_hi3}",
            "subfic {temp}, {borrow_reg}, 0",
            "subfe {dst_v3}, {p_lo3}, {dst_v3}",
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}",
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
            "addc {p_lo0}, {p_lo0}, {carry_hi}",
            "addze {carry_hi}, {p_hi0}",
            "subfic {temp}, {borrow_reg}, 0",
            "subfe {dst_v0}, {p_lo0}, {dst_v0}",
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}",
            "stw {dst_v0}, 0({dst})",

            "bdnz 4b",

            "3:",
            "neg {borrow_reg}, {borrow_reg}",  // convert mask (0/-1) to 0/1

            carry_hi = inout(reg) carry_hi,
            borrow_reg = inout(reg) borrow_reg,
            dst = inout(reg_nonzero) dst => _,
            src = inout(reg_nonzero) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            scalar = in(reg) scalar,
            src_v0 = out(reg) _, src_v1 = out(reg) _, src_v2 = out(reg) _, src_v3 = out(reg) _,
            dst_v0 = out(reg) _, dst_v1 = out(reg) _, dst_v2 = out(reg) _, dst_v3 = out(reg) _,
            p_lo0 = out(reg) _, p_lo1 = out(reg) _, p_lo2 = out(reg) _, p_lo3 = out(reg) _,
            p_hi0 = out(reg) _, p_hi1 = out(reg) _, p_hi2 = out(reg) _, p_hi3 = out(reg) _,
            temp = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    (carry_hi, borrow_reg)
}
