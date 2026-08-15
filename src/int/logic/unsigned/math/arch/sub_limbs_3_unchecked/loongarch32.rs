//! `LoongArch32` implementation of `sub_limbs_3_unchecked`.
//!
//! Evaluates `dst = src1 - src2` using 4-way unrolled loops with branchless `sltu` borrow tracking.

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
/// `LoongArch32` tracks borrows branchlessly using `sub.w` and `sltu` (set-less-than unsigned).
/// Two-stage comparison captures borrow from `src1 < src2` and subsequent borrow from subtracting previous borrow.
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
    // 1. `dst`, `src1`, and `src2` are valid for `len` 32-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "beq {chunks}, $zero, 2f",                   // If chunks == 0, skip to remainder (2f)
            ".p2align 4",

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ld.w {t0}, {src2}, 0",                      // Load src2[0]
            "ld.w {t1}, {src1}, 0",                      // Load src1[0]
            "sltu {c0}, {t1}, {t0}",                     // c0 = 1 if src1[0] < src2[0]
            "sub.w {t1}, {t1}, {t0}",                    // t1 = src1[0] - src2[0]
            "sltu {c1}, {t1}, {borrow}",                 // c1 = 1 if diff < borrow
            "sub.w {t1}, {t1}, {borrow}",                // t1 -= borrow
            "or {borrow}, {c0}, {c1}",                   // Combined borrow for next limb
            "st.w {t1}, {dst}, 0",                       // Store dst[0]

            // [Limb 1]
            "ld.w {t0}, {src2}, 4",                      // Load src2[1]
            "ld.w {t1}, {src1}, 4",                      // Load src1[1]
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "sub.w {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "sub.w {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "st.w {t1}, {dst}, 4",                       // Store dst[1]

            // [Limb 2]
            "ld.w {t0}, {src2}, 8",                      // Load src2[2]
            "ld.w {t1}, {src1}, 8",                      // Load src1[2]
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "sub.w {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "sub.w {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "st.w {t1}, {dst}, 8",                       // Store dst[2]

            // [Limb 3]
            "ld.w {t0}, {src2}, 12",                     // Load src2[3]
            "ld.w {t1}, {src1}, 12",                     // Load src1[3]
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "sub.w {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "sub.w {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "st.w {t1}, {dst}, 12",                      // Store dst[3]

            // Advance pointers by 16 bytes and loop
            "addi.w {src1}, {src1}, 16",                 // Advance src1
            "addi.w {src2}, {src2}, 16",                 // Advance src2
            "addi.w {dst}, {dst}, 16",                   // Advance dst
            "addi.w {chunks}, {chunks}, -1",             // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "beq {rem}, $zero, 4f",                      // If rem == 0, exit (4f)
            ".p2align 4",

            // 1-limb tail loop
            "3:",
            "ld.w {t0}, {src2}, 0",                      // Load single src2 limb
            "ld.w {t1}, {src1}, 0",                      // Load single src1 limb
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "sub.w {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "sub.w {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "st.w {t1}, {dst}, 0",                       // Store dst limb

            "addi.w {src1}, {src1}, 4",                  // Advance src1
            "addi.w {src2}, {src2}, 4",                  // Advance src2
            "addi.w {dst}, {dst}, 4",                    // Advance dst
            "addi.w {rem}, {rem}, -1",                   // Decrement rem
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
            c0 = out(reg) _,
            c1 = out(reg) _,
            options(nostack)
        );
        borrow
    }
}
