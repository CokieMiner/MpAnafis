//! `LoongArch32` subtraction kernels (inline assembly).
//!
//! Evaluates `dst -= src` using 4-way unrolled loops with branchless `sltu` borrow tracking.

use core::arch::asm;

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` with borrow propagation and
/// return the final borrow-out limb (0 or 1).
///
/// Computes:
///
/// ```text
///   (borrow, dst[0..len]) = dst[0..len] - src[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// `LoongArch32` tracks borrows branchlessly using `sub.w` and `sltu` (set-less-than unsigned).
/// Two-stage comparison captures borrow from `dst < src` and subsequent borrow from subtracting previous borrow.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint.
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
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder (2f)
            ".p2align 4",

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ld.w {t0}, {src}, 0",                      // Load src[0]
            "ld.w {t1}, {dst}, 0",                      // Load dst[0]
            "sltu {c0}, {t1}, {t0}",                    // c0 = 1 if dst[0] < src[0]
            "sub.w {t1}, {t1}, {t0}",                   // t1 = dst[0] - src[0]
            "sltu {c1}, {t1}, {borrow}",                // c1 = 1 if diff < borrow
            "sub.w {t1}, {t1}, {borrow}",               // t1 -= borrow
            "or {borrow}, {c0}, {c1}",                  // Combined borrow for next limb
            "st.w {t1}, {dst}, 0",                      // Store updated dst[0]

            // [Limb 1]
            "ld.w {t0}, {src}, 4",                      // Load src[1]
            "ld.w {t1}, {dst}, 4",                      // Load dst[1]
            "sltu {c0}, {t1}, {t0}",                    // Detect primary borrow
            "sub.w {t1}, {t1}, {t0}",                   // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                // Detect secondary borrow
            "sub.w {t1}, {t1}, {borrow}",               // Subtract borrow
            "or {borrow}, {c0}, {c1}",                  // Combine borrow
            "st.w {t1}, {dst}, 4",                      // Store dst[1]

            // [Limb 2]
            "ld.w {t0}, {src}, 8",                      // Load src[2]
            "ld.w {t1}, {dst}, 8",                      // Load dst[2]
            "sltu {c0}, {t1}, {t0}",                    // Detect primary borrow
            "sub.w {t1}, {t1}, {t0}",                   // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                // Detect secondary borrow
            "sub.w {t1}, {t1}, {borrow}",               // Subtract borrow
            "or {borrow}, {c0}, {c1}",                  // Combine borrow
            "st.w {t1}, {dst}, 8",                      // Store dst[2]

            // [Limb 3]
            "ld.w {t0}, {src}, 12",                     // Load src[3]
            "ld.w {t1}, {dst}, 12",                     // Load dst[3]
            "sltu {c0}, {t1}, {t0}",                    // Detect primary borrow
            "sub.w {t1}, {t1}, {t0}",                   // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                // Detect secondary borrow
            "sub.w {t1}, {t1}, {borrow}",               // Subtract borrow
            "or {borrow}, {c0}, {c1}",                  // Combine borrow
            "st.w {t1}, {dst}, 12",                     // Store dst[3]

            // Advance pointers by 16 bytes and loop
            "addi.w {src}, {src}, 16",                  // Advance src pointer
            "addi.w {dst}, {dst}, 16",                  // Advance dst pointer
            "addi.w {chunks}, {chunks}, -1",            // Decrement chunk counter
            "bnez {chunks}, 1b",                        // Repeat while chunks != 0

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "beqz {rem}, 4f",                           // If rem == 0, exit (4f)
            ".p2align 4",

            // 1-limb tail loop
            "3:",
            "ld.w {t0}, {src}, 0",                      // Load single src limb
            "ld.w {t1}, {dst}, 0",                      // Load single dst limb
            "sltu {c0}, {t1}, {t0}",                    // Detect primary borrow
            "sub.w {t1}, {t1}, {t0}",                   // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                // Detect secondary borrow
            "sub.w {t1}, {t1}, {borrow}",               // Subtract borrow
            "or {borrow}, {c0}, {c1}",                  // Combine borrow
            "st.w {t1}, {dst}, 0",                      // Store dst limb

            "addi.w {src}, {src}, 4",                   // Advance src
            "addi.w {dst}, {dst}, 4",                   // Advance dst
            "addi.w {rem}, {rem}, -1",                  // Decrement rem
            "bnez {rem}, 3b",                           // Repeat while rem != 0

            // Exit
            "4:",

            borrow = inout(reg) borrow,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
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
