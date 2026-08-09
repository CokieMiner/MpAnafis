//! RISC-V 32-bit subtraction kernels (inline assembly).
//!
//! RISC-V has no carry flag, so borrow is tracked manually with `sltu`
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
            "beqz {chunks}, 2f",           // skip main loop if chunks == 0
            // ── 4‑way unrolled loop ────────────────────────────────────
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            "lw {t0}, 0({src})",           // t0 = src[0]
            "lw {t1}, 0({dst})",           // t1 = dst[0]
            "sltu {c0}, {t1}, {t0}",       // c0 = dst < src (borrow from src)
            "sub {t1}, {t1}, {t0}",        // t1 = dst - src
            "sltu {c1}, {t1}, {borrow}",   // c1 = t1 < borrow (borrow from previous borrow)
            "sub {t1}, {t1}, {borrow}",    // t1 = t1 - borrow
            "or {borrow}, {c0}, {c1}",     // combined borrow for next limb
            "sw {t1}, 0({dst})",           // store result

            "lw {t0}, 4({src})",           // t0 = src[1]
            "lw {t1}, 4({dst})",           // t1 = dst[1]
            "sltu {c0}, {t1}, {t0}",
            "sub {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "sub {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sw {t1}, 4({dst})",

            "lw {t0}, 8({src})",           // t0 = src[2]
            "lw {t1}, 8({dst})",           // t1 = dst[2]
            "sltu {c0}, {t1}, {t0}",
            "sub {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "sub {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sw {t1}, 8({dst})",

            "lw {t0}, 12({src})",          // t0 = src[3]
            "lw {t1}, 12({dst})",          // t1 = dst[3]
            "sltu {c0}, {t1}, {t0}",
            "sub {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "sub {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sw {t1}, 12({dst})",

            "addi {src}, {src}, 16",       // advance src by 16 bytes (4 × u32)
            "addi {dst}, {dst}, 16",       // advance dst by 16 bytes
            "addi {chunks}, {chunks}, -1", // decrement chunk counter
            "bnez {chunks}, 1b",           // loop back if chunks != 0

            // ── Tail: single‑limb remainder loop ───────────────────────
            "2:",
            "beqz {rem}, 4f",              // skip tail if rem == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "lw {t0}, 0({src})",           // t0 = src[i]
            "lw {t1}, 0({dst})",           // t1 = dst[i]
            "sltu {c0}, {t1}, {t0}",
            "sub {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "sub {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "sw {t1}, 0({dst})",           // store result
            "addi {src}, {src}, 4",        // advance src by 4 bytes
            "addi {dst}, {dst}, 4",        // advance dst by 4 bytes
            "addi {rem}, {rem}, -1",       // decrement remainder counter
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
