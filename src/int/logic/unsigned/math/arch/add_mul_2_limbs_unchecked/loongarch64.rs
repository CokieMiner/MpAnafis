//! `LoongArch64` fused dual-row multiply-add kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by two scalars `s0` and `s1` simultaneously,
/// accumulating each result into two overlapping rows of `dst`:
///
/// ```text
///   (c0, dst[0..len])   = dst[0..len]   + src[0..len] × s0 + c0_in
///   (c1, dst[1..len+1]) = dst[1..len+1] + src[0..len] × s1 + c1_in
/// ```
///
/// Returns the two final carry-out values `(c0, c1)`.
///
/// # Safety
///
/// `dst` must be valid for `len + 1` elements; `src` must be valid for `len` elements.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance: dual-scalar schoolbook inner loop"
)]
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
    // SAFETY: Caller guarantees dst and src are valid for `len` (+1) elements.
    // Uses mul.d/mulh.du + sltu carry tracking (LoongArch has no carry flag).
    // Two independent carry chains (c0, c1) are maintained simultaneously,
    // one for scalar s0 accumulating into dst[j] and one for s1 into dst[j+8].
    unsafe {
        asm!(
            "beqz {len}, 2f",

            "1:",
            "ld.d {s}, {src}, 0",

            "mul.d {p_lo0}, {s}, {s0}",
            "mulh.du {p_hi0}, {s}, {s0}",
            "mul.d {p_lo1}, {s}, {s1}",
            "mulh.du {p_hi1}, {s}, {s1}",

            "add.d {p_lo0}, {p_lo0}, {c0}",
            "sltu {t0}, {p_lo0}, {c0}",
            "add.d {p_lo1}, {p_lo1}, {c1}",
            "sltu {t1}, {p_lo1}, {c1}",

            "add.d {p_hi0}, {p_hi0}, {t0}",
            "add.d {p_hi1}, {p_hi1}, {t1}",

            "ld.d {d0}, {dst}, 0",
            "ld.d {d1}, {dst}, 8",

            "add.d {d0}, {d0}, {p_lo0}",
            "sltu {t0}, {d0}, {p_lo0}",
            "add.d {d1}, {d1}, {p_lo1}",
            "sltu {t1}, {d1}, {p_lo1}",

            "add.d {c0}, {p_hi0}, {t0}",
            "add.d {c1}, {p_hi1}, {t1}",

            "st.d {d0}, {dst}, 0",
            "st.d {d1}, {dst}, 8",

            "addi.d {src}, {src}, 8",
            "addi.d {dst}, {dst}, 8",
            "addi.d {len}, {len}, -1",
            "bnez {len}, 1b",
            "2:",

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = inout(reg) len => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            s = out(reg) _,
            d0 = out(reg) _,
            d1 = out(reg) _,
            p_lo0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi0 = out(reg) _,
            p_hi1 = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            options(nostack)
        );
    }
    (c0, c1)
}
