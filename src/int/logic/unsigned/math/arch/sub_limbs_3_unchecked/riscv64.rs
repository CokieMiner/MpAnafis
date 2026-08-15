//! RISC-V 64-bit subtraction kernels (inline assembly).
//!
//! Evaluates `dst = src1 - src2` using 4-way unrolled loops with branchless `sltu` borrow tracking.

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
/// RISC-V 64-bit tracks borrows branchlessly using `sub` and `sltu` (set-less-than unsigned).
/// Two-stage comparison captures borrow from `src1 < src2` and subsequent borrow from subtracting previous borrow.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
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
    // 1. `dst`, `src1`, and `src2` are valid for `len` 64-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder (2f)
            ".p2align 4",

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ld {t0}, 0({src1})",                        // Load src1[0]
            "ld {t1}, 0({src2})",                        // Load src2[0]
            "sltu {b0}, {t0}, {t1}",                     // b0 = 1 if src1[0] < src2[0]
            "sub {t2}, {t0}, {t1}",                      // t2 = src1[0] - src2[0]
            "sltu {b1}, {t2}, {borrow}",                 // b1 = 1 if diff < borrow
            "sub {t2}, {t2}, {borrow}",                  // t2 -= borrow
            "or {borrow}, {b0}, {b1}",                   // Combined borrow for next limb
            "sd {t2}, 0({dst})",                         // Store dst[0]

            // [Limb 1]
            "ld {t0}, 8({src1})",                        // Load src1[1]
            "ld {t1}, 8({src2})",                        // Load src2[1]
            "sltu {b0}, {t0}, {t1}",                     // Detect primary borrow
            "sub {t2}, {t0}, {t1}",                      // Subtract limbs
            "sltu {b1}, {t2}, {borrow}",                 // Detect secondary borrow
            "sub {t2}, {t2}, {borrow}",                  // Subtract borrow
            "or {borrow}, {b0}, {b1}",                   // Combine borrow
            "sd {t2}, 8({dst})",                         // Store dst[1]

            // [Limb 2]
            "ld {t0}, 16({src1})",                       // Load src1[2]
            "ld {t1}, 16({src2})",                       // Load src2[2]
            "sltu {b0}, {t0}, {t1}",                     // Detect primary borrow
            "sub {t2}, {t0}, {t1}",                      // Subtract limbs
            "sltu {b1}, {t2}, {borrow}",                 // Detect secondary borrow
            "sub {t2}, {t2}, {borrow}",                  // Subtract borrow
            "or {borrow}, {b0}, {b1}",                   // Combine borrow
            "sd {t2}, 16({dst})",                        // Store dst[2]

            // [Limb 3]
            "ld {t0}, 24({src1})",                       // Load src1[3]
            "ld {t1}, 24({src2})",                       // Load src2[3]
            "sltu {b0}, {t0}, {t1}",                     // Detect primary borrow
            "sub {t2}, {t0}, {t1}",                      // Subtract limbs
            "sltu {b1}, {t2}, {borrow}",                 // Detect secondary borrow
            "sub {t2}, {t2}, {borrow}",                  // Subtract borrow
            "or {borrow}, {b0}, {b1}",                   // Combine borrow
            "sd {t2}, 24({dst})",                        // Store dst[3]

            // Advance pointers by 32 bytes and loop
            "addi {src1}, {src1}, 32",                   // Advance src1
            "addi {src2}, {src2}, 32",                   // Advance src2
            "addi {dst}, {dst}, 32",                     // Advance dst
            "addi {chunks}, {chunks}, -1",               // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "beqz {rem}, 4f",                            // If rem == 0, exit (4f)
            ".p2align 4",

            // 1-limb tail loop
            "3:",
            "ld {t0}, 0({src1})",                        // Load single src1 limb
            "ld {t1}, 0({src2})",                        // Load single src2 limb
            "sltu {b0}, {t0}, {t1}",                     // Detect primary borrow
            "sub {t2}, {t0}, {t1}",                      // Subtract limbs
            "sltu {b1}, {t2}, {borrow}",                 // Detect secondary borrow
            "sub {t2}, {t2}, {borrow}",                  // Subtract borrow
            "or {borrow}, {b0}, {b1}",                   // Combine borrow
            "sd {t2}, 0({dst})",                         // Store dst limb
            "addi {src1}, {src1}, 8",                    // Advance src1
            "addi {src2}, {src2}, 8",                    // Advance src2
            "addi {dst}, {dst}, 8",                      // Advance dst
            "addi {rem}, {rem}, -1",                     // Decrement rem
            "bnez {rem}, 3b",                            // Repeat while rem != 0

            // Exit
            "4:",

            borrow = inout(reg) borrow,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            dst = inout(reg) dst => _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            b0 = out(reg) _,
            b1 = out(reg) _,
            options(nostack)
        );
        borrow
    }
}
