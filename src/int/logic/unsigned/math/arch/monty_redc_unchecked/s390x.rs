//! IBM Z CIOS Montgomery reduction-step kernel.

use core::arch::asm;

use super::Limb;

/// Compute one fused Coarsely Integrated Operand Scanning reduction step.
///
/// # Safety
///
/// `out`, `b`, and `m` must each be valid for `len` limbs. The modulus must be
/// odd and `m_inv` must equal `-m[0]^-1 mod B`.
#[allow(
    clippy::inline_always,
    reason = "The CIOS reduction step is the inner loop of Montgomery multiplication"
)]
#[inline(always)]
pub unsafe fn monty_redc_step_unchecked(
    out: *mut Limb,
    b: *const Limb,
    m: *const Limb,
    mut len: usize,
    a_i: Limb,
    m_inv: Limb,
) -> Limb {
    if len == 0 {
        return 0;
    }

    let quotient = m_inv;
    let zero: Limb = 0;

    // `mlgr` forms the exact unsigned product in r2:r3. Each `algr` sets the
    // carry condition consumed by the following `alcgr`, so carry_b and
    // carry_m remain proper base-B high limbs. The inverse cancels the first
    // low limb, making each staggered store the exact CIOS division by B.
    // SAFETY: the caller-provided spans cover every load. Step zero advances
    // the pointers once; the loop handles len-1 remaining limbs and stores
    // out[0..len-1], then the epilogue stores out[len-1].
    unsafe {
        asm!(
            "lg {out_limb}, 0({out})",
            "lg {factor}, 0({b})",
            "lgr %r3, {factor}",
            "mlgr %r2, {a_i}",
            "algr %r3, {out_limb}",
            "alcgr %r2, {zero}",
            "lgr {carry_b}, %r2",
            "lgr {low_b}, %r3",

            "lgr %r3, {low_b}",
            "mlgr %r2, {quotient}",
            "lgr {quotient}, %r3",
            "lg {factor}, 0({m})",
            "lgr %r3, {factor}",
            "mlgr %r2, {quotient}",
            "algr %r3, {low_b}",
            "alcgr %r2, {zero}",
            "lgr {carry_m}, %r2",

            "la {out}, 8({out})",
            "la {b}, 8({b})",
            "la {m}, 8({m})",
            "aghi {len}, -1",
            "jz 2f",

            ".p2align 4",
            "1:",
            "lg {out_limb}, 0({out})",
            "lg {factor}, 0({b})",
            "lgr %r3, {factor}",
            "mlgr %r2, {a_i}",
            "algr %r3, {carry_b}",
            "alcgr %r2, {zero}",
            "algr %r3, {out_limb}",
            "alcgr %r2, {zero}",
            "lgr {carry_b}, %r2",
            "lgr {low_b}, %r3",

            "lg {factor}, 0({m})",
            "lgr %r3, {factor}",
            "mlgr %r2, {quotient}",
            "algr %r3, {carry_m}",
            "alcgr %r2, {zero}",
            "algr %r3, {low_b}",
            "alcgr %r2, {zero}",
            "lgr {carry_m}, %r2",
            "stg %r3, -8({out})",

            "la {out}, 8({out})",
            "la {b}, 8({b})",
            "la {m}, 8({m})",
            "brctg {len}, 1b",

            "2:",
            "lgr {out_limb}, {carry_b}",
            "algr {out_limb}, {carry_m}",
            "lghi {len}, 0",
            "alcgr {len}, {len}",
            "stg {out_limb}, -8({out})",

            out = inout(reg_addr) out => _,
            b = inout(reg_addr) b => _,
            m = inout(reg_addr) m => _,
            len = inout(reg) len,
            a_i = inout(reg) a_i => _,
            quotient = inout(reg) quotient => _,
            zero = inout(reg) zero => _,
            carry_b = out(reg) _,
            carry_m = out(reg) _,
            out_limb = out(reg) _,
            factor = out(reg) _,
            low_b = out(reg) _,
            out("r2") _,
            out("r3") _,
            options(nostack)
        );
    }
    len
}
