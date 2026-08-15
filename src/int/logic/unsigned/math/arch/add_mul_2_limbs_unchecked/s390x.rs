//! `s390x` (IBM Z z/Architecture) fused dual-row multiply-add kernel.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 64)`)
//! using `mlgr` (even/odd register pair `%r2:%r3`) and `algr`/`alcgr` carry propagation.

use core::arch::asm;

use super::Limb;

/// Fused dual-row multiply-add kernel for `s390x` (IBM Z).
///
/// Computes:
///
/// ```text
///   dst[0..len] += src[0..len] * s0 + c0
///   dst[1..len+1] += src[0..len] * s1 + c1
/// ```
///
/// # Microarchitectural Strategy
///
/// `mlgr` requires the multiplicand in `%r3` and stores the 128-bit product in `%r2:%r3`.
/// The kernel evaluates both multiplication rows (`s0` and `s1`), chains additions using `algr`
/// (Add Logical) and `alcgr` (Add Logical with Carry), and loops via 1-cycle `brctg`.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len + 1` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical inner loop for 2-row multi-precision Karatsuba and basecase multiplication"
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
    let zero: Limb = 0;

    if len == 0 {
        return (0, 0);
    }

    // SAFETY:
    // 1. `dst` is valid for reads and writes of `len + 1` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer advances remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            ".p2align 4",
            // Main dual-row accumulation loop
            "1:",

            // [Load Source and Two Destination Limbs]
            "lg {val}, 0({src})",                        // Load src[j]
            "lg {dst0}, 0({dst})",                       // Load dst[j]
            "lg {dst1}, 8({dst})",                       // Load dst[j+1]

            // [Row 0 Carry Chain: dst[j] += val * s0 + c0]
            "lgr %r3, {val}",                            // %r3 = src[j]
            "mlgr %r2, {s0}",                            // %r2:%r3 = %r3 * s0 (128-bit product)
            "algr %r3, {c0}",                            // %r3 += c0, set Condition Code (CC)
            "alcgr %r2, {zero}",                         // %r2 += CC carry + 0
            "algr {dst0}, %r3",                          // dst0 += %r3, set CC
            "alcgr %r2, {zero}",                         // %r2 += CC carry + 0
            "lgr {c0}, %r2",                             // c0 = %r2 (row 0 carry)
            "stg {dst0}, 0({dst})",                      // Store finalized dst[j]

            // [Row 1 Carry Chain: dst[j+1] += val * s1 + c1]
            "lgr %r3, {val}",                            // %r3 = src[j]
            "mlgr %r2, {s1}",                            // %r2:%r3 = %r3 * s1
            "algr %r3, {c1}",                            // %r3 += c1, set CC
            "alcgr %r2, {zero}",                         // %r2 += CC carry + 0
            "algr {dst1}, %r3",                          // dst1 += %r3, set CC
            "alcgr %r2, {zero}",                         // %r2 += CC carry + 0
            "lgr {c1}, %r2",                             // c1 = %r2 (row 1 carry)
            "stg {dst1}, 8({dst})",                      // Store intermediate dst[j+1]

            // Advance pointers and loop via CTR
            "la {src}, 8({src})",                        // Advance src pointer by 8 bytes
            "la {dst}, 8({dst})",                        // Advance dst pointer by 8 bytes
            "brctg {len}, 1b",                           // Decrement len and branch if > 0

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg_addr) src => _,
            dst = inout(reg_addr) dst => _,
            len = inout(reg) len => _,
            s0 = inout(reg) s0 => _,
            s1 = inout(reg) s1 => _,
            zero = inout(reg) zero => _,
            val = out(reg) _,
            dst0 = out(reg) _,
            dst1 = out(reg) _,
            out("r2") _,
            out("r3") _,
            options(nostack)
        );
    }
    (c0, c1)
}
