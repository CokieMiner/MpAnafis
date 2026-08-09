//! RISC-V 64-bit write-only dual-row multiplication kernel.

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

    // `mulhu` supplies the exact high half of each B-by-B product. The `sltu`
    // results are single carry bits, so adding them to the corresponding high
    // halves maintains carry0, carry1 < B. The final pair therefore occupies
    // exactly dst[len..len+2].
    // SAFETY: the caller guarantees the source and destination spans. The
    // loop reads exactly src[0..len], writes dst[0..len], and the epilogue
    // writes dst[len..len+2].
    unsafe {
        asm!(
            "1:",
            "ld {value}, 0({src})",

            "mul {lo0}, {value}, {s0}",
            "mulhu {hi0}, {value}, {s0}",
            "add {sum0}, {lo0}, {carry0}",
            "sltu {carry_bit0}, {sum0}, {carry0}",
            "add {hi0}, {hi0}, {carry_bit0}",
            "add {out}, {sum0}, {pending1}",
            "sltu {carry_bit1}, {out}, {pending1}",
            "add {carry0}, {hi0}, {carry_bit1}",
            "sd {out}, 0({dst})",

            "mul {lo1}, {value}, {s1}",
            "mulhu {hi1}, {value}, {s1}",
            "add {pending1}, {lo1}, {carry1}",
            "sltu {carry_bit2}, {pending1}, {carry1}",
            "add {carry1}, {hi1}, {carry_bit2}",

            "addi {src}, {src}, 8",
            "addi {dst}, {dst}, 8",
            "addi {len}, {len}, -1",
            "bnez {len}, 1b",

            "add {out}, {pending1}, {carry0}",
            "sltu {carry_bit0}, {out}, {pending1}",
            "add {carry1}, {carry1}, {carry_bit0}",
            "sd {out}, 0({dst})",
            "sd {carry1}, 8({dst})",

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
            sum0 = out(reg) _,
            carry_bit0 = out(reg) _,
            out = out(reg) _,
            carry_bit1 = out(reg) _,
            lo1 = out(reg) _,
            hi1 = out(reg) _,
            carry_bit2 = out(reg) _,
            options(nostack)
        );
    }
}
