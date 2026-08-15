//! RISC-V 64-bit subtraction kernels (inline assembly).
//!
//! Evaluates `dst -= src` using 4-way unrolled loops with branchless `sltu` borrow tracking.

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
/// RISC-V 64-bit tracks borrows branchlessly using `sub` and `sltu` (set-less-than unsigned).
/// Two-stage comparison captures borrow from `dst < src` and subsequent borrow from subtracting previous borrow.
///
/// # Safety
///
/// `dst` and `src` must each be valid for `len` elements of type `Limb`.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` and `src` are valid for `len` 64-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder (2f)
            ".p2align 4",

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ld {t0}, 0({src})",                         // Load src[0]
            "ld {t1}, 0({dst})",                         // Load dst[0]
            "sltu {b0}, {t1}, {t0}",                     // b0 = 1 if dst[0] < src[0]
            "sub {t2}, {t1}, {t0}",                      // t2 = dst[0] - src[0]
            "sltu {b1}, {t2}, {borrow}",                 // b1 = 1 if diff < borrow
            "sub {t2}, {t2}, {borrow}",                  // t2 -= borrow
            "or {borrow}, {b0}, {b1}",                   // Combined borrow for next limb
            "sd {t2}, 0({dst})",                         // Store updated dst[0]

            // [Limb 1]
            "ld {t0}, 8({src})",                         // Load src[1]
            "ld {t1}, 8({dst})",                         // Load dst[1]
            "sltu {b0}, {t1}, {t0}",                     // Detect primary borrow
            "sub {t2}, {t1}, {t0}",                      // Subtract limbs
            "sltu {b1}, {t2}, {borrow}",                 // Detect secondary borrow
            "sub {t2}, {t2}, {borrow}",                  // Subtract borrow
            "or {borrow}, {b0}, {b1}",                   // Combine borrow
            "sd {t2}, 8({dst})",                         // Store dst[1]

            // [Limb 2]
            "ld {t0}, 16({src})",                        // Load src[2]
            "ld {t1}, 16({dst})",                        // Load dst[2]
            "sltu {b0}, {t1}, {t0}",                     // Detect primary borrow
            "sub {t2}, {t1}, {t0}",                      // Subtract limbs
            "sltu {b1}, {t2}, {borrow}",                 // Detect secondary borrow
            "sub {t2}, {t2}, {borrow}",                  // Subtract borrow
            "or {borrow}, {b0}, {b1}",                   // Combine borrow
            "sd {t2}, 16({dst})",                        // Store dst[2]

            // [Limb 3]
            "ld {t0}, 24({src})",                        // Load src[3]
            "ld {t1}, 24({dst})",                        // Load dst[3]
            "sltu {b0}, {t1}, {t0}",                     // Detect primary borrow
            "sub {t2}, {t1}, {t0}",                      // Subtract limbs
            "sltu {b1}, {t2}, {borrow}",                 // Detect secondary borrow
            "sub {t2}, {t2}, {borrow}",                  // Subtract borrow
            "or {borrow}, {b0}, {b1}",                   // Combine borrow
            "sd {t2}, 24({dst})",                        // Store dst[3]

            // Advance pointers by 32 bytes and loop
            "addi {src}, {src}, 32",                     // Advance src pointer
            "addi {dst}, {dst}, 32",                     // Advance dst pointer
            "addi {chunks}, {chunks}, -1",               // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "beqz {rem}, 4f",                            // If rem == 0, exit (4f)
            ".p2align 4",

            // 1-limb tail loop
            "3:",
            "ld {t0}, 0({src})",                         // Load single src limb
            "ld {t1}, 0({dst})",                         // Load single dst limb
            "sltu {b0}, {t1}, {t0}",                     // Detect primary borrow
            "sub {t2}, {t1}, {t0}",                      // Subtract limbs
            "sltu {b1}, {t2}, {borrow}",                 // Detect secondary borrow
            "sub {t2}, {t2}, {borrow}",                  // Subtract borrow
            "or {borrow}, {b0}, {b1}",                   // Combine borrow
            "sd {t2}, 0({dst})",                         // Store dst limb
            "addi {src}, {src}, 8",                      // Advance src
            "addi {dst}, {dst}, 8",                      // Advance dst
            "addi {rem}, {rem}, -1",                     // Decrement rem
            "bnez {rem}, 3b",                            // Repeat while rem != 0

            // Exit
            "4:",

            borrow = inout(reg) borrow,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            b0 = out(reg) _,
            b1 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
