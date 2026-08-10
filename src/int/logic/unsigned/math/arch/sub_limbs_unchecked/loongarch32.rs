//! `LoongArch32` subtraction kernels (inline assembly).
//!
//! `LoongArch32` uses the same carry-tracking idiom as RISC-V 32:
//! `sub.w` for subtraction and `sltu` (set-less-than unsigned) to detect
//! borrow. There is no borrow flag in the ISA.
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
            "beqz {chunks}, 2f",
            // ── 4-way unrolled loop ────────────────────────────────────
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            // Limb 0
            "ld.w {t0}, {src}, 0",
            "ld.w {t1}, {dst}, 0",
            "sltu {c0}, {t1}, {t0}",
            "sub.w {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "sub.w {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "st.w {t1}, {dst}, 0",

            // Limb 1
            "ld.w {t0}, {src}, 4",
            "ld.w {t1}, {dst}, 4",
            "sltu {c0}, {t1}, {t0}",
            "sub.w {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "sub.w {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "st.w {t1}, {dst}, 4",

            // Limb 2
            "ld.w {t0}, {src}, 8",
            "ld.w {t1}, {dst}, 8",
            "sltu {c0}, {t1}, {t0}",
            "sub.w {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "sub.w {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "st.w {t1}, {dst}, 8",

            // Limb 3
            "ld.w {t0}, {src}, 12",
            "ld.w {t1}, {dst}, 12",
            "sltu {c0}, {t1}, {t0}",
            "sub.w {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "sub.w {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "st.w {t1}, {dst}, 12",

            "addi.w {src}, {src}, 16",
            "addi.w {dst}, {dst}, 16",
            "addi.w {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            // ── Tail: single-limb remainder loop ───────────────────────
            "2:",
            "beqz {rem}, 4f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "ld.w {t0}, {src}, 0",
            "ld.w {t1}, {dst}, 0",
            "sltu {c0}, {t1}, {t0}",
            "sub.w {t1}, {t1}, {t0}",
            "sltu {c1}, {t1}, {borrow}",
            "sub.w {t1}, {t1}, {borrow}",
            "or {borrow}, {c0}, {c1}",
            "st.w {t1}, {dst}, 0",

            "addi.w {src}, {src}, 4",
            "addi.w {dst}, {dst}, 4",
            "addi.w {rem}, {rem}, -1",
            "bnez {rem}, 3b",
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
