//! `PowerPC64` POWER9 fused dual-row multiply-add kernel.
//!
//! Uses `maddld`/`maddhdu` for fused multiply‑accumulate with two independent
//! carry‑chains via `addc`/`addze` (the carry chain uses `addc`, which ignores
//! the stale `CA` from `maddld`, rather than `adde`, which would read it). The
//! `d_cur`/`d_next` register carry‑forward eliminates the store→load forwarding
//! stall between iterations.

use core::arch::asm;

use super::Limb;

/// Fused `add_mul_2` kernel for PowerPC 64-bit (POWER9 ISA 3.0).
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len + 1` limbs: the second
///   row writes one limb ahead of the first, so the last store lands at
///   `dst[len]`.
/// - `src` must be valid for reads of `len` limbs.
/// - `dst` and `src` must not overlap, even partially: the loop reads `src`
///   while it writes `dst`, so any overlap is a data race.
#[allow(clippy::inline_always, reason = "Performance critical inner loop")]
#[inline(always)]
pub unsafe fn add_mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) -> (Limb, Limb) {
    let mut c0: Limb = 0;
    let mut c1: Limb = 0;

    // SAFETY: Caller guarantees dst and src are valid for len elements.
    unsafe {
        asm!(
            "cmpldi {len}, 0",
            "beq 2f",
            "ld {d_cur}, 0({dst})",
            "mtctr {len}",
            ".p2align 4",
            "1:",
            "ld {s}, 0({src})",
            "ld {d_next}, 8({dst})",

            // --- s0 chain: finish dst[j] ---
            "maddld {t0}, {s}, {s0}, {d_cur}",
            "maddhdu {hi0}, {s}, {s0}, {d_cur}",
            "addc {d_cur}, {t0}, {c0}",
            "addze {c0}, {hi0}",
            "std {d_cur}, 0({dst})",

            // --- s1 chain: compute dst[j+1], carry forward as next d_cur ---
            "maddld {t1}, {s}, {s1}, {d_next}",
            "maddhdu {hi1}, {s}, {s1}, {d_next}",
            "addc {d_cur}, {t1}, {c1}",
            "addze {c1}, {hi1}",

            "addi {src}, {src}, 8",
            "addi {dst}, {dst}, 8",
            "bdnz 1b",
            "std {d_cur}, 0({dst})",     // flush the pending high word
            "2:",

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg_nonzero) src => _,
            dst = inout(reg_nonzero) dst => _,
            len = in(reg) len,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            s = out(reg) _,
            d_cur = out(reg) _,
            d_next = out(reg) _,
            t0 = out(reg) _,
            hi0 = out(reg) _,
            t1 = out(reg) _,
            hi1 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    (c0, c1)
}
