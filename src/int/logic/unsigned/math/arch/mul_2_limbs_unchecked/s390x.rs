//! IBM Z write-only dual-row multiplication kernel.

use core::arch::asm;

use super::Limb;

/// Write `src * (s0 + s1 * B)` into `dst` without reading old destination data.
///
/// # Safety
///
/// `src` must cover `len` readable limbs and `dst` must cover `len + 2`
/// writable limbs. The source and destination spans must not overlap. A zero
/// length returns without dereferencing either pointer.
#[allow(
    clippy::inline_always,
    reason = "The write-only two-row kernel initializes every basecase product and must not add a call boundary"
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
    let zero: Limb = 0;

    // `mlgr` returns each exact B-by-B product in r2:r3. Every `algr`
    // condition code is consumed immediately by `alcgr`, so carry0 and
    // carry1 remain the high base-B limbs of their respective rows. The low
    // limb of row one is delayed in pending1 until row zero reaches the same
    // output position; therefore each loop iteration finalizes exactly one
    // destination limb and the final pair fits in dst[len..len+2].
    // SAFETY: the caller proves the non-overlapping spans. The loop reads
    // src[0..len] and writes dst[0..len]; the epilogue writes the final two
    // advertised limbs.
    unsafe {
        asm!(
            ".p2align 4",
            "1:",
            "lg {value}, 0({src})",

            "lgr %r3, {value}",
            "mlgr %r2, {s0}",
            "algr %r3, {carry0}",
            "alcgr %r2, {zero}",
            "algr %r3, {pending1}",
            "alcgr %r2, {zero}",
            "lgr {carry0}, %r2",
            "stg %r3, 0({dst})",

            "lgr %r3, {value}",
            "mlgr %r2, {s1}",
            "algr %r3, {carry1}",
            "alcgr %r2, {zero}",
            "lgr {pending1}, %r3",
            "lgr {carry1}, %r2",

            "la {src}, 8({src})",
            "la {dst}, 8({dst})",
            "brctg {len}, 1b",

            "algr {pending1}, {carry0}",
            "alcgr {carry1}, {zero}",
            "stg {pending1}, 0({dst})",
            "stg {carry1}, 8({dst})",

            src = inout(reg_addr) src => _,
            dst = inout(reg_addr) dst => _,
            len = inout(reg) len => _,
            s0 = inout(reg) s0 => _,
            s1 = inout(reg) s1 => _,
            carry0 = inout(reg) carry0 => _,
            carry1 = inout(reg) carry1 => _,
            pending1 = inout(reg) pending1 => _,
            zero = inout(reg) zero => _,
            value = out(reg) _,
            out("r2") _,
            out("r3") _,
            options(nostack)
        );
    }
}
