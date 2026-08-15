//! PowerPC 32-bit subtraction kernels (inline assembly).
//!
//! Evaluates `dst -= src` using 4-way unrolled loops with hardware borrow `XER[CA]`
//! via `subfe` (subtract from extended) and CTR hardware branch looping (`bdnz`).

use core::arch::asm;

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` and return the final borrow.
///
/// Computes:
///
/// ```text
///   (borrow, dst[0..len]) = dst[0..len] − src[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// PowerPC hardware maintains the carry/borrow in `XER[CA]` across `subfe` instructions.
/// `subfe dst, src, dst` evaluates `dst - src + CA - 1` (where CA=1 means no borrow, CA=0 means borrow).
/// The 4-way unrolled loop loads 4 words each from `src` and `dst`, subtracts them with borrow,
/// and uses hardware CTR register (`bdnz`) to eliminate branch latency.
///
/// # Safety
///
/// `dst` and `src` must each be valid for `len` elements of type `Limb`.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow: Limb;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` and `src` are valid for `len` 32-bit `Limb` elements.
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
            // [Load 4 Limbs from src and dst]
            "lwz {src_v0}, 0({src})",                   // Load src[0]
            "lwz {src_v1}, 4({src})",                   // Load src[1]
            "lwz {src_v2}, 8({src})",                   // Load src[2]
            "lwz {src_v3}, 12({src})",                  // Load src[3]
            "lwz {dst_v0}, 0({dst})",                   // Load dst[0]
            "lwz {dst_v1}, 4({dst})",                   // Load dst[1]
            "lwz {dst_v2}, 8({dst})",                   // Load dst[2]
            "lwz {dst_v3}, 12({dst})",                  // Load dst[3]

            // [Subtract with Borrow: dst = dst - src + CA - 1]
            "subfe {dst_v0}, {src_v0}, {dst_v0}",       // dst_v0 = dst[0] - src[0] + CA - 1
            "subfe {dst_v1}, {src_v1}, {dst_v1}",       // dst_v1 = dst[1] - src[1] + CA - 1
            "subfe {dst_v2}, {src_v2}, {dst_v2}",       // dst_v2 = dst[2] - src[2] + CA - 1
            "subfe {dst_v3}, {src_v3}, {dst_v3}",       // dst_v3 = dst[3] - src[3] + CA - 1

            // [Store 4 Limbs to dst]
            "stw {dst_v0}, 0({dst})",                   // Store dst[0]
            "stw {dst_v1}, 4({dst})",                   // Store dst[1]
            "stw {dst_v2}, 8({dst})",                   // Store dst[2]
            "stw {dst_v3}, 12({dst})",                  // Store dst[3]

            // Advance pointers by 16 bytes and loop via CTR
            "addi {src}, {src}, 16",                    // Advance src pointer
            "addi {dst}, {dst}, 16",                    // Advance dst pointer
            "bdnz 2b",                                  // Decrement CTR and branch if != 0

            // Remainder entry point (0 to 3 limbs)
            "1:",
            "cmpwi {rem}, 0",                           // Check if rem == 0
            "beq 3f",                                   // If rem == 0, exit (3f)
            "mtctr {rem}",                              // Load remainder count into CTR
            ".p2align 4",

            // 1-limb tail loop
            "4:",
            "lwz {src_v0}, 0({src})",                   // Load single src limb
            "lwz {dst_v0}, 0({dst})",                   // Load single dst limb
            "subfe {dst_v0}, {src_v0}, {dst_v0}",       // Subtract with borrow
            "stw {dst_v0}, 0({dst})",                   // Store single dst limb
            "addi {src}, {src}, 4",                     // Advance src
            "addi {dst}, {dst}, 4",                     // Advance dst
            "bdnz 4b",                                  // Decrement CTR and branch if != 0

            // Exit: capture final borrow bit from XER[CA]
            "3:",
            "li {borrow}, 0",                           // borrow = 0
            "subfe {borrow}, {borrow}, {borrow}",       // borrow = 0 - 0 + CA - 1 (0 if no borrow, -1 if borrow)
            "neg {borrow}, {borrow}",                   // borrow = 0 or 1

            borrow = out(reg) borrow,
            dst = inout(reg_nonzero) dst => _,
            src = inout(reg_nonzero) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src_v0 = out(reg) _, src_v1 = out(reg) _, src_v2 = out(reg) _, src_v3 = out(reg) _,
            dst_v0 = out(reg) _, dst_v1 = out(reg) _, dst_v2 = out(reg) _, dst_v3 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    borrow
}
