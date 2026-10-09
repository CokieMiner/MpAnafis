//! RISC-V 64-bit subtraction kernels (inline assembly).
//!
//! Evaluates `dst -= src` using 4-way unrolled loops with branchless `sltu` borrow tracking.

use core::arch::asm;

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` and return the final borrow.
///
/// `dst_after - borrow*B^len = dst_before - src`, with `borrow` in `{0,1}`.
///
/// # Safety
///
/// Both pointers must cover `len` aligned initialized limbs; `dst` must be
/// writable. Spans must be identical or disjoint and their byte widths must
/// fit in `isize::MAX`. `len == 0` performs no pointer access.
#[expect(clippy::inline_always, reason = "Inlining exposes the four-limb assembly borrow recurrence to callers")]
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
