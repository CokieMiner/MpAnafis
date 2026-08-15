//! ARM implementation of `sub_limbs_unchecked`.
//!
//! Evaluates `dst -= src` using 4-way unrolled `sbcs` chains with post-increment addressing.

use core::arch::asm;

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` with borrow propagation.
///
/// Returns the final borrow-out limb (0 or 1).
///
/// Computes:
///
/// ```text
///   (borrow, dst[0..len]) = dst[0..len] - src[0..len]
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
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` and `src` are valid for `len` 32-bit `Limb` elements.
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
            "ldr {s}, [{src}], #4",                      // Load src[0] and advance (+4)
            "ldr {d}, [{dst}]",                          // Load dst[0]
            "sbcs {d}, {d}, {s}",                        // d = dst[0] - src[0] - !C (updates C flag)
            "str {d}, [{dst}], #4",                      // Store updated dst[0] and advance (+4)

            // [Limb 1]
            "ldr {s}, [{src}], #4",                      // Load src[1]
            "ldr {d}, [{dst}]",                          // Load dst[1]
            "sbcs {d}, {d}, {s}",                        // Subtract with borrow
            "str {d}, [{dst}], #4",                      // Store dst[1]

            // [Limb 2]
            "ldr {s}, [{src}], #4",                      // Load src[2]
            "ldr {d}, [{dst}]",                          // Load dst[2]
            "sbcs {d}, {d}, {s}",                        // Subtract with borrow
            "str {d}, [{dst}], #4",                      // Store dst[2]

            // [Limb 3]
            "ldr {s}, [{src}], #4",                      // Load src[3]
            "ldr {d}, [{dst}]",                          // Load dst[3]
            "sbcs {d}, {d}, {s}",                        // Subtract with borrow
            "str {d}, [{dst}], #4",                      // Store dst[3]

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
            "ldr {s}, [{src}], #4",                      // Load single src limb
            "ldr {d}, [{dst}]",                          // Load single dst limb
            "sbcs {d}, {d}, {s}",                        // Subtract with borrow
            "str {d}, [{dst}], #4",                      // Store single dst limb

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
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            s = out(reg) _,
            d = out(reg) _,
            options(nostack)
        );
        borrow
    }
}
