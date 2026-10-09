//! `MIPS64` implementation of `sub_limbs_3_unchecked`.
//!
//! Evaluates `dst = src1 - src2` using 4-way unrolled loops with branchless `sltu` borrow tracking.

use core::arch::asm;

use super::Limb;

/// Compute `dst[i] = src1[i] - src2[i] - borrow` for `len` limbs, returning
/// the final borrow.
///
/// `dst - borrow*B^len = src1 - src2`, with `borrow` in `{0,1}`.
///
/// # Safety
///
/// Both sources must cover `len` aligned initialized readable limbs and may
/// alias each other. `dst` must cover `len` aligned writable limbs, disjoint
/// from both sources; its contents may be uninitialized. Each byte span must
/// fit in `isize::MAX`. `len == 0` performs no pointer access.
#[expect(
    clippy::inline_always,
    reason = "Inlining exposes the four-limb assembly borrow recurrence to callers"
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

    // SAFETY: four-limb blocks and the len&3 tail cover exactly len aligned
    // initialized source limbs and disjoint writable output limbs. Sources
    // may coincide. The two SLTU comparisons capture the limb subtraction and
    // incoming binary borrow independently. Every changed register is an early
    // output; no old destination value or stack memory is read.
    unsafe {
        asm!(
            ".set noat",
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder (2f)
            ".p2align 4",

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ld {t0}, 0({src2})",                        // Load src2[0]
            "ld {t1}, 0({src1})",                        // Load src1[0]
            "sltu {c0}, {t1}, {t0}",                     // c0 = 1 if src1[0] < src2[0]
            "dsubu {t1}, {t1}, {t0}",                    // t1 = src1[0] - src2[0]
            "sltu {c1}, {t1}, {borrow}",                 // c1 = 1 if diff < borrow
            "dsubu {t1}, {t1}, {borrow}",                // t1 -= borrow
            "or {borrow}, {c0}, {c1}",                   // Combined borrow for next limb
            "sd {t1}, 0({dst})",                         // Store dst[0]

            // [Limb 1]
            "ld {t0}, 8({src2})",                        // Load src2[1]
            "ld {t1}, 8({src1})",                        // Load src1[1]
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "dsubu {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "dsubu {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "sd {t1}, 8({dst})",                         // Store dst[1]

            // [Limb 2]
            "ld {t0}, 16({src2})",                       // Load src2[2]
            "ld {t1}, 16({src1})",                       // Load src1[2]
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "dsubu {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "dsubu {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "sd {t1}, 16({dst})",                        // Store dst[2]

            // [Limb 3]
            "ld {t0}, 24({src2})",                       // Load src2[3]
            "ld {t1}, 24({src1})",                       // Load src1[3]
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "dsubu {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "dsubu {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "sd {t1}, 24({dst})",                        // Store dst[3]

            // Advance pointers by 32 bytes and loop
            "daddiu {src1}, {src1}, 32",                 // Advance src1
            "daddiu {src2}, {src2}, 32",                 // Advance src2
            "daddiu {dst}, {dst}, 32",                   // Advance dst
            "daddiu {chunks}, {chunks}, -1",             // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "beqz {rem}, 4f",                            // If rem == 0, exit (4f)
            ".p2align 4",

            // 1-limb tail loop
            "3:",
            "ld {t0}, 0({src2})",                        // Load single src2 limb
            "ld {t1}, 0({src1})",                        // Load single src1 limb
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "dsubu {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "dsubu {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "sd {t1}, 0({dst})",                         // Store dst limb

            "daddiu {src1}, {src1}, 8",                  // Advance src1
            "daddiu {src2}, {src2}, 8",                  // Advance src2
            "daddiu {dst}, {dst}, 8",                    // Advance dst
            "daddiu {rem}, {rem}, -1",                   // Decrement rem
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
