//! `AArch64` write-only dual-row multiplication kernel.

use core::arch::asm;

use super::Limb;

/// Write `src * (s0 + s1 * B)` into `dst` without reading its old contents.
///
/// # Safety
///
/// `src` must be valid for `len` limbs and `dst` for `len + 2` limbs. The
/// input and output regions must not overlap. A zero length returns without
/// dereferencing either pointer.
#[allow(
    clippy::inline_always,
    reason = "Critical basecase initialization kernel; removing its call boundary matters for small products"
)]
#[inline(always)]
pub unsafe fn mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) {
    if len == 0 {
        return;
    }

    let carry0: Limb = 0;
    let carry1: Limb = 0;
    let pending1: Limb = 0;

    // Each iteration writes one finalized output limb. `pending1` is the low
    // limb of the preceding s1 row, while `carry0` and `carry1` are strictly
    // below B by the ordinary base-B multiplication invariant. Consequently
    // the two closing additions fit exactly in the advertised len+2 limbs.
    // SAFETY: the caller guarantees the source and destination spans. The
    // loop reads exactly src[0..len], writes dst[0..len], and the epilogue
    // writes dst[len..len+2].
    unsafe {
        asm!(
            "1:",
            "ldr {value}, [{src}], #8",

            "mul {lo0}, {value}, {s0}",
            "umulh {hi0}, {value}, {s0}",
            "adds {lo0}, {lo0}, {carry0}",
            "adc {hi0}, {hi0}, xzr",
            "adds {out}, {lo0}, {pending1}",
            "adc {carry0}, {hi0}, xzr",
            "str {out}, [{dst}], #8",

            "mul {lo1}, {value}, {s1}",
            "umulh {hi1}, {value}, {s1}",
            "adds {pending1}, {lo1}, {carry1}",
            "adc {carry1}, {hi1}, xzr",

            "subs {len}, {len}, #1",
            "b.ne 1b",

            "adds {pending1}, {pending1}, {carry0}",
            "adc {carry1}, {carry1}, xzr",
            "stp {pending1}, {carry1}, [{dst}]",

            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = inout(reg) len => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            carry0 = inout(reg) carry0 => _,
            carry1 = inout(reg) carry1 => _,
            pending1 = inout(reg) pending1 => _,
            value = out(reg) _,
            lo0 = out(reg) _,
            hi0 = out(reg) _,
            out = out(reg) _,
            lo1 = out(reg) _,
            hi1 = out(reg) _,
            options(nostack)
        );
    }
}
