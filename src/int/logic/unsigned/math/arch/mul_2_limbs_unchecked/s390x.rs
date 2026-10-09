//! IBM Z (s390x) write-only dual-row multiplication kernel.
//!
//! Evaluates `dst = src * (s0 + s1 * B)` in a single write-only pass using
//! `mlgr` (even/odd register pair `%r2:%r3`) and `algr`/`alcgr` carry propagation.

use core::arch::asm;

use super::Limb;

/// Write `src * (s0 + s1 * B)` into `dst` without reading old destination data.
///
/// Computes:
///
/// ```text
///   dst[0..len+2] = src[0..len] x (s0 + s1 x 2^64)
/// ```
///
/// Evaluates two simultaneous multiplication rows in registers without memory reads of `dst`.
/// Both products (`src[j] * s0` and `src[j] * s1`) are computed via `mlgr` into `%r2:%r3`,
/// and carries are chained using `algr`/`alcgr`.
///
/// # Safety
///
/// - For nonzero `len`, aligned `dst` must cover `len + 2` writable limbs,
///   which may be uninitialized; aligned `src` must cover `len` initialized limbs.
/// - `len + 2` must fit `usize`, and both byte spans must fit `isize::MAX`.
/// - Source and destination spans must be disjoint.
#[expect(
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

    // SAFETY: disjoint aligned spans provide len initialized inputs and len+2
    // writable outputs. BRCTG permits exactly len source loads and stores,
    // followed by two in-range final stores. Products plus two limbs are <=
    // B^2-1; the final high carry is <= s1. Output is never read. The product
    // pair r2:r3 and every modified general register are declared as outputs.
    unsafe {
        asm!(
            ".p2align 4",
            // Main dual-row multiplication loop
            "1:",
            "lg {value}, 0({src})",                      // Load src[j]

            // [Row 0 Multiplication & Carry Merge]
            "lgr %r3, {value}",                          // %r3 = src[j]
            "mlgr %r2, {s0}",                            // %r2:%r3 = %r3 * s0 (128-bit product)
            "algr %r3, {carry0}",                        // %r3 += carry0, set Condition Code (CC)
            "alcgr %r2, {zero}",                         // %r2 += CC carry + 0
            "algr %r3, {pending1}",                      // %r3 += pending1 (merge row 1 high limb), set CC
            "alcgr %r2, {zero}",                         // %r2 += CC carry + 0
            "lgr {carry0}, %r2",                         // carry0 = %r2 (row 0 carry)
            "stg %r3, 0({dst})",                         // Store finalized dst[j]

            // [Row 1 Multiplication & Pending Carry Tracking]
            "lgr %r3, {value}",                          // %r3 = src[j]
            "mlgr %r2, {s1}",                            // %r2:%r3 = %r3 * s1
            "algr %r3, {carry1}",                        // %r3 += carry1, set CC
            "alcgr %r2, {zero}",                         // %r2 += CC carry + 0
            "lgr {pending1}, %r3",                       // pending1 = low product (to merge in next step)
            "lgr {carry1}, %r2",                         // carry1 = high product (row 1 carry)

            // Advance pointers and decrement the remaining limb count.
            "la {src}, 8({src})",                        // Advance src pointer by 8 bytes
            "la {dst}, 8({dst})",                        // Advance dst pointer by 8 bytes
            "brctg {len}, 1b",                           // Decrement len and branch if > 0

            // [Epilogue: Store trailing high limbs]
            "algr {pending1}, {carry0}",                 // pending1 += carry0, set CC
            "alcgr {carry1}, {zero}",                    // carry1 += CC carry + 0
            "stg {pending1}, 0({dst})",                  // Store dst[len]
            "stg {carry1}, 8({dst})",                    // Store final high limb dst[len+1]

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
