//! MIPS 64-bit subtraction kernels (inline assembly).
//!
//! Evaluates `dst -= src` using 4-way unrolled loops with branchless `sltu` borrow tracking.

use core::arch::asm;

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` with borrow propagation and
/// return the final borrow-out limb (0 or 1).
///
/// `dst_after - borrow*B^len = dst_before - src`, with `borrow` in `{0,1}`.
///
/// # Safety
///
/// Both pointers must cover `len` aligned initialized limbs; `dst` must be
/// writable. Spans must be identical or disjoint and their byte widths must
/// fit in `isize::MAX`. `len == 0` performs no pointer access.
#[expect(
    clippy::inline_always,
    reason = "Inlining exposes the four-limb assembly borrow recurrence to callers"
)]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: four-limb blocks and the len&3 tail partition exactly len aligned
    // initialized limbs. Each source is read before its matching write, so
    // exact alias is valid. Two SLTU tests capture the subtraction and incoming
    // binary borrow. Changed pointers and temporaries are early outputs.
    unsafe {
        asm!(
            ".set noat",
            "beqz {chunks}, 2f",                         // If chunks == 0, skip to remainder (2f)
            ".p2align 4",

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ld {t0}, 0({src})",                         // Load src[0]
            "ld {t1}, 0({dst})",                         // Load dst[0]
            "sltu {c0}, {t1}, {t0}",                     // c0 = 1 if dst[0] < src[0]
            "dsubu {t1}, {t1}, {t0}",                    // t1 = dst[0] - src[0]
            "sltu {c1}, {t1}, {borrow}",                 // c1 = 1 if diff < borrow
            "dsubu {t1}, {t1}, {borrow}",                // t1 -= borrow
            "or {borrow}, {c0}, {c1}",                   // Combined borrow for next limb
            "sd {t1}, 0({dst})",                         // Store updated dst[0]

            // [Limb 1]
            "ld {t0}, 8({src})",                         // Load src[1]
            "ld {t1}, 8({dst})",                         // Load dst[1]
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "dsubu {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "dsubu {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "sd {t1}, 8({dst})",                         // Store dst[1]

            // [Limb 2]
            "ld {t0}, 16({src})",                        // Load src[2]
            "ld {t1}, 16({dst})",                        // Load dst[2]
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "dsubu {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "dsubu {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "sd {t1}, 16({dst})",                        // Store dst[2]

            // [Limb 3]
            "ld {t0}, 24({src})",                        // Load src[3]
            "ld {t1}, 24({dst})",                        // Load dst[3]
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "dsubu {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "dsubu {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "sd {t1}, 24({dst})",                        // Store dst[3]

            // Advance pointers by 32 bytes and loop
            "daddiu {src}, {src}, 32",                   // Advance src pointer
            "daddiu {dst}, {dst}, 32",                   // Advance dst pointer
            "daddiu {chunks}, {chunks}, -1",             // Decrement chunk counter
            "bnez {chunks}, 1b",                         // Repeat while chunks != 0

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "beqz {rem}, 4f",                            // If rem == 0, exit (4f)
            ".p2align 4",

            // 1-limb tail loop
            "3:",
            "ld {t0}, 0({src})",                         // Load single src limb
            "ld {t1}, 0({dst})",                         // Load single dst limb
            "sltu {c0}, {t1}, {t0}",                     // Detect primary borrow
            "dsubu {t1}, {t1}, {t0}",                    // Subtract limbs
            "sltu {c1}, {t1}, {borrow}",                 // Detect secondary borrow
            "dsubu {t1}, {t1}, {borrow}",                // Subtract borrow
            "or {borrow}, {c0}, {c1}",                   // Combine borrow
            "sd {t1}, 0({dst})",                         // Store dst limb
            "daddiu {src}, {src}, 8",                    // Advance src
            "daddiu {dst}, {dst}, 8",                    // Advance dst
            "daddiu {rem}, {rem}, -1",                   // Decrement rem
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
            c0 = out(reg) _,
            c1 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
