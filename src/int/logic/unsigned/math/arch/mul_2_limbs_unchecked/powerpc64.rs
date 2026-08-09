//! PowerPC 64-bit write-only dual-row multiplication kernel.

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

    // `mulld`/`mulhdu` form each exact two-limb product. `addc` records the
    // low-half carry in CA and `addze` incorporates that bit into the high
    // half, preserving the ordinary base-B row invariant carry0, carry1 < B.
    // SAFETY: the caller guarantees the source and destination spans. The
    // loop reads exactly src[0..len], writes dst[0..len], and the epilogue
    // writes dst[len..len+2].
    unsafe {
        asm!(
            "addi {src}, {src}, -8",
            "addi {dst}, {dst}, -8",
            "mtctr {len}",
            ".p2align 4",
            "1:",
            "ldu {value}, 8({src})",

            // Hoist all four multiplies above the carry chains
            "mulld {lo0}, {value}, {s0}",
            "mulhdu {hi0}, {value}, {s0}",
            "mulld {lo1}, {value}, {s1}",
            "mulhdu {hi1}, {value}, {s1}",

            // s0 chain + pending1 merge
            "addc {sum0}, {lo0}, {carry0}",
            "addze {hi0}, {hi0}",
            "addc {out}, {sum0}, {pending1}",
            "addze {carry0}, {hi0}",

            // s1 chain (no store — result becomes next pending1)
            "addc {pending1}, {lo1}, {carry1}",
            "addze {carry1}, {hi1}",

            "stdu {out}, 8({dst})",
            "bdnz 1b",

            "addc {pending1}, {pending1}, {carry0}",
            "addze {carry1}, {carry1}",
            "std {pending1}, 8({dst})",
            "std {carry1}, 16({dst})",

            src = inout(reg_nonzero) src => _,
            dst = inout(reg_nonzero) dst => _,
            len = in(reg) len,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            carry0 = inout(reg) carry0 => _,
            carry1 = inout(reg) carry1 => _,
            pending1 = inout(reg) pending1 => _,
            value = out(reg) _,
            lo0 = out(reg) _,
            hi0 = out(reg) _,
            sum0 = out(reg) _,
            out = out(reg) _,
            lo1 = out(reg) _,
            hi1 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            options(nostack)
        );
    }
}
