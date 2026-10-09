//! `PowerPC64` subtraction kernels (inline assembly).
//!
//! Evaluates `dst = src1 - src2` using 4-way unrolled loops with hardware borrow `XER[CA]`
//! via `subfe` (subtract from extended) and CTR hardware branch looping (`bdnz`).

use core::arch::asm;

use super::Limb;

/// Compute `dst[i] = src1[i] - src2[i] - borrow` for `len` limbs,
/// returning the final borrow.
///
/// `dst - borrow*B^len = src1 - src2`, with `borrow` in `{0,1}`.
///
/// # Safety
///
/// Both sources must cover `len` aligned initialized readable limbs and may
/// alias each other. `dst` must cover `len` aligned writable limbs, disjoint
/// from both sources; its contents may be uninitialized. Each byte span must
/// fit in `isize::MAX`. `len == 0` performs no pointer access.
#[expect(clippy::inline_always, reason = "Inlining exposes the four-limb assembly borrow chain to callers")]
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

    // SAFETY: four-limb blocks and the len&3 tail access exactly len aligned
    // initialized source limbs and disjoint writable output limbs. Sources may
    // coincide. Tail update loads read only the sources; STDU advances dst
    // while initializing it. SUBFE carries the inverted borrow in CA across
    // CTR branches. Every temporary is an early output; CTR, XER, CR0 are clobbered.
    unsafe {
        asm!(
            "subfc {borrow}, {borrow}, {borrow}",       // CA = 1 (no initial borrow)

            "cmpldi {chunks}, 0",                       // Check if chunks == 0
            "beq 1f",                                   // If chunks == 0, skip to remainder (1f)
            "mtctr {chunks}",                           // Load chunk count into CTR register
            ".p2align 4",
            "2:",
            // [Load 4 Limbs from src1 and src2]
            "ld {src1_v0}, 0({src1})",                  // Load src1[0]
            "ld {src1_v1}, 8({src1})",                  // Load src1[1]
            "ld {src1_v2}, 16({src1})",                 // Load src1[2]
            "ld {src1_v3}, 24({src1})",                 // Load src1[3]
            "ld {src2_v0}, 0({src2})",                  // Load src2[0]
            "ld {src2_v1}, 8({src2})",                  // Load src2[1]
            "ld {src2_v2}, 16({src2})",                 // Load src2[2]
            "ld {src2_v3}, 24({src2})",                 // Load src2[3]

            // [Subtract with Borrow: dst = src1 - src2 + CA - 1]
            "subfe {src1_v0}, {src2_v0}, {src1_v0}",    // src1_v0 = src1[0] - src2[0] + CA - 1
            "subfe {src1_v1}, {src2_v1}, {src1_v1}",    // src1_v1 = src1[1] - src2[1] + CA - 1
            "subfe {src1_v2}, {src2_v2}, {src1_v2}",    // src1_v2 = src1[2] - src2[2] + CA - 1
            "subfe {src1_v3}, {src2_v3}, {src1_v3}",    // src1_v3 = src1[3] - src2[3] + CA - 1

            // [Store 4 Limbs to dst]
            "std {src1_v0}, 0({dst})",                  // Store dst[0]
            "std {src1_v1}, 8({dst})",                  // Store dst[1]
            "std {src1_v2}, 16({dst})",                 // Store dst[2]
            "std {src1_v3}, 24({dst})",                 // Store dst[3]

            // Advance pointers by 32 bytes and loop via CTR
            "addi {src1}, {src1}, 32",                  // Advance src1 pointer
            "addi {src2}, {src2}, 32",                  // Advance src2 pointer
            "addi {dst}, {dst}, 32",                    // Advance dst pointer
            "bdnz 2b",                                  // Decrement CTR and branch if != 0

            // Remainder entry point (0 to 3 limbs)
            "1:",
            "cmpldi {rem}, 0",                          // Check if rem == 0
            "beq 3f",                                   // If rem == 0, exit (3f)
            "mtctr {rem}",                              // Load remainder count into CTR
            "addi {src1}, {src1}, -8",                  // Pre-adjust pointers for update-form loads
            "addi {src2}, {src2}, -8",
            "addi {dst}, {dst}, -8",
            ".p2align 4",

            // 1-limb tail loop
            "4:",
            "ldu {src1_v0}, 8({src1})",                 // Load src1 limb with update (+8)
            "ldu {src2_v0}, 8({src2})",                 // Load src2 limb with update (+8)
            "subfe {src1_v0}, {src2_v0}, {src1_v0}",    // Subtract with borrow
            "stdu {src1_v0}, 8({dst})",                 // Initialize dst limb and advance
            "bdnz 4b",                                  // Decrement CTR and branch if != 0

            // Exit: capture final borrow bit from XER[CA]
            "3:",
            "li {borrow}, 0",                           // borrow = 0
            "subfe {borrow}, {borrow}, {borrow}",       // borrow = 0 - 0 + CA - 1 (0 if no borrow, -1 if borrow)
            "neg {borrow}, {borrow}",                   // borrow = 0 or 1

            borrow = out(reg) borrow,
            dst = inout(reg_nonzero) dst => _,
            src1 = inout(reg_nonzero) src1 => _,
            src2 = inout(reg_nonzero) src2 => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1_v0 = out(reg) _, src1_v1 = out(reg) _, src1_v2 = out(reg) _, src1_v3 = out(reg) _,
            src2_v0 = out(reg) _, src2_v1 = out(reg) _, src2_v2 = out(reg) _, src2_v3 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
        borrow
    }
}
