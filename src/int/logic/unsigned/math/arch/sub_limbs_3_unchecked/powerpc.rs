//! `PowerPC32` implementation of `sub_limbs_3_unchecked`.

use core::arch::asm;

use super::Limb;

/// Compute `dst[i] = src1[i] - src2[i] - borrow` for `len` limbs, returning
/// the final borrow.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span: the kernel writes `dst`
///   while it reads `src1` and `src2`.
/// - `src1` and `src2` are read-only and may alias each other.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn sub_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    let mut borrow: Limb;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: Assembly block accesses arrays using lengths guaranteed to be valid by caller logic.
    unsafe {
        asm!(
            "subfc {borrow}, {borrow}, {borrow}",
            "cmpwi {chunks}, 0",
            "beq 1f",
            "mtctr {chunks}",
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "lwz {src2_v0}, 0({src2})",
            "lwz {src2_v1}, 4({src2})",
            "lwz {src2_v2}, 8({src2})",
            "lwz {src2_v3}, 12({src2})",
            "lwz {src1_v0}, 0({src1})",
            "lwz {src1_v1}, 4({src1})",
            "lwz {src1_v2}, 8({src1})",
            "lwz {src1_v3}, 12({src1})",
            "subfe {src1_v0}, {src2_v0}, {src1_v0}",
            "subfe {src1_v1}, {src2_v1}, {src1_v1}",
            "subfe {src1_v2}, {src2_v2}, {src1_v2}",
            "subfe {src1_v3}, {src2_v3}, {src1_v3}",
            "stw {src1_v0}, 0({dst})",
            "stw {src1_v1}, 4({dst})",
            "stw {src1_v2}, 8({dst})",
            "stw {src1_v3}, 12({dst})",
            "addi {src1}, {src1}, 16",
            "addi {src2}, {src2}, 16",
            "addi {dst}, {dst}, 16",
            "bdnz 2b",
            "1:",
            "cmpwi {rem}, 0",
            "beq 3f",
            "mtctr {rem}",
            ".p2align 4",                          // align loop header for fetch efficiency
            "4:",
            "lwz {src2_v0}, 0({src2})",
            "lwz {src1_v0}, 0({src1})",
            "subfe {src1_v0}, {src2_v0}, {src1_v0}",
            "stw {src1_v0}, 0({dst})",
            "addi {src1}, {src1}, 4",
            "addi {src2}, {src2}, 4",
            "addi {dst}, {dst}, 4",
            "bdnz 4b",
            "3:",
            "li {borrow}, 0",
            "subfe {borrow}, {borrow}, {borrow}",
            "neg {borrow}, {borrow}",
            borrow = out(reg) borrow,
            dst = inout(reg_nonzero) dst => _,
            src1 = inout(reg_nonzero) src1 => _,
            src2 = inout(reg_nonzero) src2 => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src2_v0 = out(reg) _, src2_v1 = out(reg) _, src2_v2 = out(reg) _, src2_v3 = out(reg) _,
            src1_v0 = out(reg) _, src1_v1 = out(reg) _, src1_v2 = out(reg) _, src1_v3 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    borrow
}
