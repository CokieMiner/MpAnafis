//! IBM Z (s390x) CIOS Montgomery reduction-step kernel.
//!
//! Implements Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
//! using `mlgr` (even/odd register pair `%r2:%r3`) and `algr`/`alcgr` carry propagation.

use core::arch::asm;

use super::Limb;

/// Compute one fused Coarsely Integrated Operand Scanning (CIOS) reduction step.
///
/// Computes:
///
/// ```text
///   (out[0..len] + a_i * b[0..len] + q * m[0..len]) / 2^64
/// ```
///
/// # Microarchitectural Strategy
///
/// `mlgr` forms the exact unsigned product in `%r2:%r3`. Each `algr` sets the
/// carry condition consumed by the following `alcgr`, so `carry_b` and `carry_m`
/// remain proper base-B high limbs. The loop executes with 1-cycle `brctg` branching.
///
/// # Safety
///
/// - `out` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `b` and `m` must point to readable buffers of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of all buffers.
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

    // SAFETY:
    // 1. `out` is valid for reads and writes of `len` 64-bit `Limb` elements.
    // 2. `b` and `m` are valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            // Step 0: Prime the reduction pipeline with limb 0
            "lg {out_limb}, 0({out})",                   // Load out[0]
            "lg {factor}, 0({b})",                       // Load b[0]
            "lgr %r3, {factor}",                         // %r3 = b[0]
            "mlgr %r2, {a_i}",                           // %r2:%r3 = b[0] * a_i (128-bit product)
            "algr %r3, {out_limb}",                      // %r3 += out[0], set Condition Code (CC)
            "alcgr %r2, {zero}",                         // %r2 += CC carry + 0
            "lgr {carry_b}, %r2",                        // carry_b = %r2
            "lgr {low_b}, %r3",                          // low_b = %r3

            // Derive quotient multiplier q
            "lgr %r3, {low_b}",                          // %r3 = low_b
            "mlgr %r2, {quotient}",                      // %r2:%r3 = low_b * m_inv
            "lgr {quotient}, %r3",                       // quotient = (low_b * m_inv) mod 2^64
            "lg {factor}, 0({m})",                       // Load m[0]
            "lgr %r3, {factor}",                         // %r3 = m[0]
            "mlgr %r2, {quotient}",                      // %r2:%r3 = m[0] * quotient
            "algr %r3, {low_b}",                         // Cancel low word (0 mod 2^64)
            "alcgr %r2, {zero}",                         // Absorb carry into %r2
            "lgr {carry_m}, %r2",                        // carry_m = %r2

            // Advance pointers to limb 1
            "la {out}, 8({out})",                        // Advance out pointer
            "la {b}, 8({b})",                            // Advance b pointer
            "la {m}, 8({m})",                            // Advance m pointer
            "aghi {len}, -1",                            // Decrement len
            "jz 2f",                                     // If len == 1, skip to epilogue (2f)

            ".p2align 4",
            // Main reduction loop for j = 1 to len-1
            "1:",
            "lg {out_limb}, 0({out})",                   // Load out[j]
            "lg {factor}, 0({b})",                       // Load b[j]
            "lgr %r3, {factor}",                         // %r3 = b[j]
            "mlgr %r2, {a_i}",                           // %r2:%r3 = b[j] * a_i
            "algr %r3, {carry_b}",                       // %r3 += carry_b, set CC
            "alcgr %r2, {zero}",                         // %r2 += CC carry
            "algr %r3, {out_limb}",                      // %r3 += out[j], set CC
            "alcgr %r2, {zero}",                         // %r2 += CC carry
            "lgr {carry_b}, %r2",                        // Update carry_b
            "lgr {low_b}, %r3",                          // low_b = %r3

            "lg {factor}, 0({m})",                       // Load m[j]
            "lgr %r3, {factor}",                         // %r3 = m[j]
            "mlgr %r2, {quotient}",                      // %r2:%r3 = m[j] * quotient
            "algr %r3, {carry_m}",                       // %r3 += carry_m, set CC
            "alcgr %r2, {zero}",                         // %r2 += CC carry
            "algr %r3, {low_b}",                         // %r3 += low_b, set CC
            "alcgr %r2, {zero}",                         // %r2 += CC carry
            "lgr {carry_m}, %r2",                        // Update carry_m
            "stg %r3, -8({out})",                        // Store shifted limb into out[j-1]

            "la {out}, 8({out})",                        // Advance out
            "la {b}, 8({b})",                            // Advance b
            "la {m}, 8({m})",                            // Advance m
            "brctg {len}, 1b",                           // Decrement len and branch if > 0

            // Epilogue: Flush combined carries and return top overflow
            "2:",
            "lgr {out_limb}, {carry_b}",                 // out_limb = carry_b
            "algr {out_limb}, {carry_m}",                // out_limb += carry_m, set CC
            "lghi {len}, 0",                             // Clear len
            "alcgr {len}, {len}",                        // len = CC carry (top overflow: 0 or 1)
            "stg {out_limb}, -8({out})",                 // Store finalized out[len-1]

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
