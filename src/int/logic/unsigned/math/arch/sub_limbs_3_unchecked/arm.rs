//! ARM 32-bit (`ARMv6` / ARMv7-A) implementation of `sub_limbs_3_unchecked`.
//!
//! Evaluates `dst = src1 - src2` using 4-way unrolled `sbcs` chains with post-increment addressing.

use core::arch::asm;

use super::Limb;

/// Compute `dst[i] = src1[i] − src2[i] − borrow` for `len` limbs,
/// returning the final borrow.
///
/// Computes:
///
/// ```text
///   (borrow, dst[0..len]) = src1[0..len] - src2[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// ARM hardware uses inverted borrow in the C flag (C=1: no borrow, C=0: borrow).
/// `rsbs` converts between standard boolean borrow (0/1) and ARM inverted carry flag.
/// The 4-way unrolled loop loads and subtracts 4 words per iteration using post-indexed addressing.
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
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst`, `src1`, `src2` are valid for `len` 32-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "cmp {chunks}, #0",                          // Check if chunks == 0
            "beq 2f",                                    // If chunks == 0, skip to remainder (2f)
            "rsbs {borrow}, {borrow}, #0",               // C = 1 - borrow (sets C flag for initial borrow)
            ".p2align 4",

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ldr {s1}, [{src1}], #4",                    // Load src1[0] and advance (+4)
            "ldr {s2}, [{src2}], #4",                    // Load src2[0] and advance (+4)
            "sbcs {s1}, {s1}, {s2}",                     // s1 = s1 - s2 - !C (updates C flag)
            "str {s1}, [{dst}], #4",                     // Store dst[0] and advance (+4)

            // [Limb 1]
            "ldr {s1}, [{src1}], #4",                    // Load src1[1]
            "ldr {s2}, [{src2}], #4",                    // Load src2[1]
            "sbcs {s1}, {s1}, {s2}",                     // Subtract with borrow
            "str {s1}, [{dst}], #4",                     // Store dst[1]

            // [Limb 2]
            "ldr {s1}, [{src1}], #4",                    // Load src1[2]
            "ldr {s2}, [{src2}], #4",                    // Load src2[2]
            "sbcs {s1}, {s1}, {s2}",                     // Subtract with borrow
            "str {s1}, [{dst}], #4",                     // Store dst[2]

            // [Limb 3]
            "ldr {s1}, [{src1}], #4",                    // Load src1[3]
            "ldr {s2}, [{src2}], #4",                    // Load src2[3]
            "sbcs {s1}, {s1}, {s2}",                     // Subtract with borrow
            "str {s1}, [{dst}], #4",                     // Store dst[3]

            // Loop iteration check preserving borrow across branch
            "mov {borrow}, #0",                          // borrow = 0
            "movcc {borrow}, #1",                        // If Carry Clear (CC), borrow = 1
            "subs {chunks}, {chunks}, #1",               // Decrement chunk counter
            "beq 2f",                                    // If chunks == 0, proceed to remainder
            "rsbs {borrow}, {borrow}, #0",               // Restore C flag from borrow
            "b 1b",                                      // Repeat loop

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "cmp {rem}, #0",                             // Check if rem == 0
            "beq 4f",                                    // If rem == 0, exit (4f)
            "rsbs {borrow}, {borrow}, #0",               // Restore C flag
            ".p2align 4",

            // 1-limb tail loop
            "3:",
            "ldr {s1}, [{src1}], #4",                    // Load single src1 limb
            "ldr {s2}, [{src2}], #4",                    // Load single src2 limb
            "sbcs {s1}, {s1}, {s2}",                     // Subtract with borrow
            "str {s1}, [{dst}], #4",                     // Store single dst limb

            "mov {borrow}, #0",                          // borrow = 0
            "movcc {borrow}, #1",                        // If Carry Clear, borrow = 1
            "subs {rem}, {rem}, #1",                     // Decrement remainder counter
            "beq 4f",                                    // If rem == 0, exit
            "rsbs {borrow}, {borrow}, #0",               // Restore C flag
            "b 3b",

            // Exit
            "4:",

            borrow = inout(reg) borrow,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            dst = inout(reg) dst => _,
            s1 = out(reg) _,
            s2 = out(reg) _,
            options(nostack)
        );
        borrow
    }
}
