//! `PowerPC64` multiply-subtract limb kernel.

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
/// This is the `PowerPC64` inline assembly implementation utilizing `mulld`/`mulhdu`
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
            "cmpldi {chunks}, 0",
            "beq 1f",                     // skip chunk loop if chunks == 0
            "mtctr {chunks}",             // CTR = chunks
            ".p2align 4",

            "2:",                         // --- Unrolled Loop x4 ---

            // Load all four limbs
            "ld {src_v0}, 0({src})",
            "ld {src_v1}, 8({src})",
            "ld {src_v2}, 16({src})",
            "ld {src_v3}, 24({src})",
            "ld {dst_v0}, 0({dst})",
            "ld {dst_v1}, 8({dst})",
            "ld {dst_v2}, 16({dst})",
            "ld {dst_v3}, 24({dst})",

            // Hoist all eight multiplies
            "mulld {p_lo0}, {src_v0}, {scalar}",
            "mulhdu {p_hi0}, {src_v0}, {scalar}",
            "mulld {p_lo1}, {src_v1}, {scalar}",
            "mulhdu {p_hi1}, {src_v1}, {scalar}",
            "mulld {p_lo2}, {src_v2}, {scalar}",
            "mulhdu {p_hi2}, {src_v2}, {scalar}",
            "mulld {p_lo3}, {src_v3}, {scalar}",
            "mulhdu {p_hi3}, {src_v3}, {scalar}",

            // Multiply carry chain + subtraction borrow chain (mask form)
            "addc {p_lo0}, {p_lo0}, {carry_hi}",
            "addze {carry_hi}, {p_hi0}",
            "subfic {temp}, {borrow_reg}, 0",
            "subfe {dst_v0}, {p_lo0}, {dst_v0}",
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}",
            "std {dst_v0}, 0({dst})",

            "addc {p_lo1}, {p_lo1}, {carry_hi}",
            "addze {carry_hi}, {p_hi1}",
            "subfic {temp}, {borrow_reg}, 0",
            "subfe {dst_v1}, {p_lo1}, {dst_v1}",
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}",
            "std {dst_v1}, 8({dst})",

            "addc {p_lo2}, {p_lo2}, {carry_hi}",
            "addze {carry_hi}, {p_hi2}",
            "subfic {temp}, {borrow_reg}, 0",
            "subfe {dst_v2}, {p_lo2}, {dst_v2}",
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}",
            "std {dst_v2}, 16({dst})",

            "addc {p_lo3}, {p_lo3}, {carry_hi}",
            "addze {carry_hi}, {p_hi3}",
            "subfic {temp}, {borrow_reg}, 0",
            "subfe {dst_v3}, {p_lo3}, {dst_v3}",
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}",
            "std {dst_v3}, 24({dst})",

            "addi {src}, {src}, 32",
            "addi {dst}, {dst}, 32",
            "bdnz 2b",                    // --CTR; loop if CTR != 0

            "1:",                         // --- Remainder Loop ---
            "cmpldi {rem}, 0",
            "beq 3f",                     // skip tail if rem == 0
            "mtctr {rem}",                // CTR = rem
            "addi {src}, {src}, -8",
            "addi {dst}, {dst}, -8",

            ".p2align 4",
            "4:",
            "ldu {src_v0}, 8({src})",
            "ldu {dst_v0}, 8({dst})",
            "mulld {p_lo0}, {src_v0}, {scalar}",
            "mulhdu {p_hi0}, {src_v0}, {scalar}",
            "addc {p_lo0}, {p_lo0}, {carry_hi}",
            "addze {carry_hi}, {p_hi0}",
            "subfic {temp}, {borrow_reg}, 0",
            "subfe {dst_v0}, {p_lo0}, {dst_v0}",
            "subfe {borrow_reg}, {borrow_reg}, {borrow_reg}",
            "std {dst_v0}, 0({dst})",

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
