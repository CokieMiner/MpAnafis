//! `s390x` multiply-add limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// This computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// This is the `s390x` (IBM Z) inline assembly implementation. It utilizes `mlgr`
/// for 64x64->128-bit multiplication (which strictly requires an even/odd register pair,
/// here mapped to `%r2` and `%r3`). Accumulation is performed via `algr` (add logical)
/// and `alcgr` (add logical with carry) to smoothly propagate carries into the high product.
///
/// The loop is **2-way unrolled** using `brctg` for zero-overhead loop control.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(clippy::inline_always, reason = "Performance critical inner loop")]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry: Limb = 0;
    let chunks = len >> 1;
    let rem = len & 1;
    let zero: Limb = 0;

    // SAFETY: Caller guarantees dst and src are valid for `len` elements.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",      // skip chunks loop if chunks == 0

            ".p2align 4",
            "2:",                           // --- Unrolled Loop x2 ---
            // Limb 0
            "lg {src_v}, 0({src})",
            "lg {dst_v}, 0({dst})",
            "lgr %r3, {src_v}",             // move src to r3
            "mlgr %r2, {scalar}",           // r2:r3 = r3 * scalar
            "algr %r3, {carry}",            // r3 += carry, sets CC
            "alcgr %r2, {zero}",            // r2 += 0 + CC (carry out 1)
            "algr {dst_v}, %r3",            // dst_v += r3, sets CC
            "alcgr %r2, {zero}",            // r2 += 0 + CC (carry out 2)
            "lgr {carry}, %r2",             // carry = r2
            "stg {dst_v}, 0({dst})",

            // Limb 1
            "lg {src_v}, 8({src})",
            "lg {dst_v}, 8({dst})",
            "lgr %r3, {src_v}",
            "mlgr %r2, {scalar}",
            "algr %r3, {carry}",
            "alcgr %r2, {zero}",
            "algr {dst_v}, %r3",
            "alcgr %r2, {zero}",
            "lgr {carry}, %r2",
            "stg {dst_v}, 8({dst})",

            "la {src}, 16({src})",
            "la {dst}, 16({dst})",
            "brctg {chunks}, 2b",           // chunks--, loop if != 0

            "1:",                           // --- Remainder Loop ---
            "cgij {rem}, 0, 8, 3f",         // skip tail if rem == 0

            "lg {src_v}, 0({src})",
            "lg {dst_v}, 0({dst})",
            "lgr %r3, {src_v}",
            "mlgr %r2, {scalar}",
            "algr %r3, {carry}",
            "alcgr %r2, {zero}",
            "algr {dst_v}, %r3",
            "alcgr %r2, {zero}",
            "lgr {carry}, %r2",
            "stg {dst_v}, 0({dst})",

            "3:",                           // --- End ---

            carry = inout(reg) carry,
            zero = inout(reg) zero => _,
            dst = inout(reg_addr) dst => _,
            src = inout(reg_addr) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            scalar = inout(reg) scalar => _,
            src_v = out(reg) _,
            dst_v = out(reg) _,
            out("r2") _,
            out("r3") _,
            options(nostack)
        );
    }
    carry
}
