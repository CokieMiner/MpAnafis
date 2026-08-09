//! MIPS 64-bit subtraction kernels (inline assembly).
//!
//! MIPS has no carry flag, so borrow is tracked manually with `sltu`
//! (set-less-than unsigned) and `or`.
//!
//! The loop is **4-way unrolled** (`len >> 2`) for maximum throughput.

use core::arch::asm;

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` with borrow propagation and
/// return the final borrow-out limb (0 or 1).
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint: the kernel
///   reads `src[i]` and `dst[i]` and then writes `dst[i]`, so a partial
///   overlap is a data race.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    unsafe {
        asm!(
            ".set noat",
            "beqz {chunks}, 2f",           // skip main loop if chunks == 0
            // ── 4‑way unrolled loop ────────────────────────────────────
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            "ld {t0}, 0({src})",           // t0 = src[0]
            "ld {t1}, 0({dst})",           // t1 = dst[0]
            "sltu {c0}, {t1}, {t0}",       // c0 = dst < src (borrow from src)
            "dsubu {t1}, {t1}, {t0}",      // t1 = dst - src
            "sltu {c1}, {t1}, {borrow}",   // c1 = t1 < borrow (borrow from previous borrow)
            "dsubu {t1}, {t1}, {borrow}",  // t1 = t1 - borrow
            "or {borrow}, {c0}, {c1}",     // combined borrow for next limb
            "sd {t1}, 0({dst})",           // store result

            "ld {t0}, 8({src})",           // t0 = src[1]
            "ld {t1}, 8({dst})",           // t1 = dst[1]
            "sltu {c0}, {t1}, {t0}",
            "dsubu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "dsubu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sd {t1}, 8({dst})",

            "ld {t0}, 16({src})",          // t0 = src[2]
            "ld {t1}, 16({dst})",          // t1 = dst[2]
            "sltu {c0}, {t1}, {t0}",
            "dsubu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "dsubu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sd {t1}, 16({dst})",

            "ld {t0}, 24({src})",          // t0 = src[3]
            "ld {t1}, 24({dst})",          // t1 = dst[3]
            "sltu {c0}, {t1}, {t0}",
            "dsubu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "dsubu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sd {t1}, 24({dst})",

            "daddiu {src}, {src}, 32",     // advance src by 32 bytes (4 × u64)
            "daddiu {dst}, {dst}, 32",     // advance dst by 32 bytes
            "daddiu {chunks}, {chunks}, -1",// decrement chunk counter
            "bnez {chunks}, 1b",           // loop back if chunks != 0

            // ── Tail: single‑limb remainder loop ───────────────────────
            "2:",
            "beqz {rem}, 4f",              // skip tail if rem == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "ld {t0}, 0({src})",           // t0 = src[i]
            "ld {t1}, 0({dst})",           // t1 = dst[i]
            "sltu {c0}, {t1}, {t0}",
            "dsubu {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "dsubu {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sd {t1}, 0({dst})",           // store result
            "daddiu {src}, {src}, 8",      // advance src by 8 bytes
            "daddiu {dst}, {dst}, 8",      // advance dst by 8 bytes
            "daddiu {rem}, {rem}, -1",     // decrement remainder counter
            "bnez {rem}, 3b",              // loop back if rem != 0
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
