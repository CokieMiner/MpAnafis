//! `PowerPC32` fused dual-row multiply-add kernel.

use core::arch::asm;

use super::Limb;

/// Fused `add_mul_2` kernel for PowerPC 32-bit.
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
            "cmpwi {len}, 0",
            "beq 2f",
            "lwz {d_cur}, 0({dst})",
            "mtctr {len}",
            ".p2align 4",
            "1:",
            "lwz {s}, 0({src})",
            "lwz {d_next}, 4({dst})",

            // Hoist both multiply pairs
            "mullw {p_lo0}, {s}, {s0}",
            "mulhwu {p_hi0}, {s}, {s0}",
            "mullw {p_lo1}, {s}, {s1}",
            "mulhwu {p_hi1}, {s}, {s1}",

            // --- s0 chain: finish dst[j] ---
            "addc {t_lo0}, {p_lo0}, {c0}",
            "addze {p_hi0}, {p_hi0}",
            "addc {d_cur}, {t_lo0}, {d_cur}",
            "addze {c0}, {p_hi0}",
            "stw {d_cur}, 0({dst})",

            // --- s1 chain: compute dst[j+1], carry forward as next d_cur ---
            "addc {t_lo1}, {p_lo1}, {c1}",
            "addze {p_hi1}, {p_hi1}",
            "addc {d_cur}, {t_lo1}, {d_next}",
            "addze {c1}, {p_hi1}",

            "addi {src}, {src}, 4",
            "addi {dst}, {dst}, 4",
            "bdnz 1b",
            "stw {d_cur}, 0({dst})",     // flush the pending high word
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
            p_lo0 = out(reg) _,
            p_hi0 = out(reg) _,
            t_lo0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi1 = out(reg) _,
            t_lo1 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    (c0, c1)
}
