//! `PowerPC32` implementation of `sub_limbs_3_unchecked`.
//!
//! Evaluates `dst = src1 - src2` using 4-way unrolled loops with hardware borrow `XER[CA]`
//! via `subfe` (subtract from extended) and CTR hardware branch looping (`bdnz`).

use core::arch::asm;

use super::Limb;

/// Compute `dst[i] = src1[i] - src2[i] - borrow` for `len` limbs, returning
/// the final borrow.
///
/// Computes:
///
/// ```text
///   (borrow, dst[0..len]) = src1[0..len] - src2[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// PowerPC hardware maintains the carry/borrow in `XER[CA]` across `subfe` instructions.
/// `subfe dst, src, dst` evaluates `dst - src + CA - 1` (where CA=1 means no borrow, CA=0 means borrow).
/// The 4-way unrolled loop loads 4 words each from `src1` and `src2`, subtracts them with borrow,
/// and uses hardware CTR register (`bdnz`) to eliminate branch latency.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span.
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

    // SAFETY:
    // 1. `dst`, `src1`, and `src2` are valid for `len` 32-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "subfc {borrow}, {borrow}, {borrow}",       // CA = 1 (no initial borrow)
            "cmpwi {chunks}, 0",                        // Check if chunks == 0
            "beq 1f",                                   // If chunks == 0, skip to remainder (1f)
            "mtctr {chunks}",                           // Load chunk count into CTR register
            ".p2align 4",
            "2:",
            // [Load 4 Limbs from src1 and src2]
            "lwz {src2_v0}, 0({src2})",                  // Load src2[0]
            "lwz {src2_v1}, 4({src2})",                  // Load src2[1]
            "lwz {src2_v2}, 8({src2})",                  // Load src2[2]
            "lwz {src2_v3}, 12({src2})",                 // Load src2[3]
            "lwz {src1_v0}, 0({src1})",                  // Load src1[0]
            "lwz {src1_v1}, 4({src1})",                  // Load src1[1]
            "lwz {src1_v2}, 8({src1})",                  // Load src1[2]
            "lwz {src1_v3}, 12({src1})",                 // Load src1[3]

            // [Subtract with Borrow: dst = src1 - src2 + CA - 1]
            "subfe {src1_v0}, {src2_v0}, {src1_v0}",    // src1_v0 = src1[0] - src2[0] + CA - 1
            "subfe {src1_v1}, {src2_v1}, {src1_v1}",    // src1_v1 = src1[1] - src2[1] + CA - 1
            "subfe {src1_v2}, {src2_v2}, {src1_v2}",    // src1_v2 = src1[2] - src2[2] + CA - 1
            "subfe {src1_v3}, {src2_v3}, {src1_v3}",    // src1_v3 = src1[3] - src2[3] + CA - 1

            // [Store 4 Limbs to dst]
            "stw {src1_v0}, 0({dst})",                  // Store dst[0]
            "stw {src1_v1}, 4({dst})",                  // Store dst[1]
            "stw {src1_v2}, 8({dst})",                  // Store dst[2]
            "stw {src1_v3}, 12({dst})",                 // Store dst[3]

            // Advance pointers by 16 bytes and loop via CTR
            "addi {src1}, {src1}, 16",                  // Advance src1
            "addi {src2}, {src2}, 16",                  // Advance src2
            "addi {dst}, {dst}, 16",                    // Advance dst
            "bdnz 2b",                                  // Decrement CTR and branch if != 0

            // Remainder entry point (0 to 3 limbs)
            "1:",
            "cmpwi {rem}, 0",                            // Check if rem == 0
            "beq 3f",                                    // If rem == 0, exit (3f)
            "mtctr {rem}",                               // Load remainder count into CTR
            ".p2align 4",

            // 1-limb tail loop
            "4:",
            "lwz {src2_v0}, 0({src2})",                  // Load single src2 limb
            "lwz {src1_v0}, 0({src1})",                  // Load single src1 limb
            "subfe {src1_v0}, {src2_v0}, {src1_v0}",    // Subtract with borrow
            "stw {src1_v0}, 0({dst})",                  // Store single dst limb
            "addi {src1}, {src1}, 4",                    // Advance src1
            "addi {src2}, {src2}, 4",                    // Advance src2
            "addi {dst}, {dst}, 4",                      // Advance dst
            "bdnz 4b",                                   // Decrement CTR and branch if != 0

            // Exit: capture final borrow bit from XER[CA]
            "3:",
            "li {borrow}, 0",                            // borrow = 0
            "subfe {borrow}, {borrow}, {borrow}",        // borrow = 0 - 0 + CA - 1 (0 if no borrow, -1 if borrow)
            "neg {borrow}, {borrow}",                    // borrow = 0 or 1

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
