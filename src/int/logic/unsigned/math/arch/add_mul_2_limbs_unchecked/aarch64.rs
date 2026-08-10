//! `AArch64` fused dual-row multiply-add kernel.

use core::arch::asm;

use super::Limb;

/// Fused `add_mul_2` kernel for `AArch64`.
///
/// Computes:
/// ```text
/// dst[0..len] += src[0..len] * s0 + c0
/// dst[1..len+1] += src[0..len] * s1 + c1
/// ```
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

    if len == 0 {
        return (0, 0);
    }

    // SAFETY: Caller guarantees dst and src are valid for len elements.
    unsafe {
        asm!(
            "1:",
            // Load 1 source value
            "ldr {src_val}, [{src}], #8",

            // Load 2 dest values: dst[j] and dst[j+1]
            "ldp {dst0}, {dst1}, [{dst}]",

            // --- s0 chain: dst[j] += src[j] * s0 + c0 ---
            "mul {lo0}, {src_val}, {s0}",
            "umulh {hi0}, {src_val}, {s0}",

            "adds {lo0}, {lo0}, {c0}",
            "adc {hi0}, {hi0}, xzr",

            "adds {dst0}, {dst0}, {lo0}",
            "adc {c0}, {hi0}, xzr",

            // --- s1 chain: dst[j+1] += src[j] * s1 + c1 ---
            "mul {lo1}, {src_val}, {s1}",
            "umulh {hi1}, {src_val}, {s1}",

            "adds {lo1}, {lo1}, {c1}",
            "adc {hi1}, {hi1}, xzr",

            "adds {dst1}, {dst1}, {lo1}",
            "adc {c1}, {hi1}, xzr",

            // Store back the updated dst values
            // Note: dst is advanced by 8 bytes (1 limb) because the next iteration
            // will process dst[j+1] and dst[j+2].
            "stp {dst0}, {dst1}, [{dst}], #8",

            "subs {len}, {len}, #1",
            "b.ne 1b",

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = inout(reg) len => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            src_val = out(reg) _,
            dst0 = out(reg) _,
            dst1 = out(reg) _,
            lo0 = out(reg) _,
            hi0 = out(reg) _,
            lo1 = out(reg) _,
            hi1 = out(reg) _,
            options(nostack)
        );
    }
    (c0, c1)
}
