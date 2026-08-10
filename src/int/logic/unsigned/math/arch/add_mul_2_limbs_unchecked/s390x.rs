//! `s390x` fused dual-row multiply-add kernel.

use core::arch::asm;

use super::Limb;

/// Fused `add_mul_2` kernel for `s390x` (IBM Z).
///
/// Computes:
/// ```text
/// dst[0..len] += src[0..len] * s0 + c0
/// dst[1..len+1] += src[0..len] * s1 + c1
/// ```
///
/// Utilizes `mlgr` for 64x64->128-bit multiplication (mapped to `%r2:%r3`)
/// and `algr`/`alcgr` for carry-propagating accumulation.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len + 1` elements.
/// - `src` must be valid for reads of `len` elements.
/// - **Aliasing**: `src[0..len]` must NOT overlap with `dst[1..len+1]`.  The
///   loop reads `src[j]`, then reads and writes `dst[j]` and `dst[j+1]`,
///   then advances both pointers by 1 limb.  If `dst` and `src` alias with
///   offset 1, the read of `src[j+1]` on the next iteration may see a
///   stale or partially updated value.
#[allow(clippy::inline_always, reason = "Performance critical inner loop")]
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

    // SAFETY: deferred to caller per the doc comment above.
    unsafe {
        asm!(
            ".p2align 4",
            "1:",
            // Read val = src[0]
            "lg {val}, 0({src})",
            // Read dst0 = dst[0]
            "lg {dst0}, 0({dst})",
            // Read dst1 = dst[1]
            "lg {dst1}, 8({dst})",

            // --- s0 chain: dst[j] += val * s0 + c0 ---
            "lgr %r3, {val}",
            "mlgr %r2, {s0}",           // r2:r3 = val * s0
            "algr %r3, {c0}",           // r3 += c0, sets CC
            "alcgr %r2, {zero}",        // r2 += 0 + CC
            "algr {dst0}, %r3",         // dst0 += r3, sets CC
            "alcgr %r2, {zero}",        // r2 += 0 + CC
            "lgr {c0}, %r2",            // c0 = r2
            "stg {dst0}, 0({dst})",

            // --- s1 chain: dst[j+1] += val * s1 + c1 ---
            "lgr %r3, {val}",
            "mlgr %r2, {s1}",           // r2:r3 = val * s1
            "algr %r3, {c1}",           // r3 += c1, sets CC
            "alcgr %r2, {zero}",        // r2 += 0 + CC
            "algr {dst1}, %r3",         // dst1 += r3, sets CC
            "alcgr %r2, {zero}",        // r2 += 0 + CC
            "lgr {c1}, %r2",            // c1 = r2
            "stg {dst1}, 8({dst})",

            "la {src}, 8({src})",
            "la {dst}, 8({dst})",
            "brctg {len}, 1b",          // len--, loop if != 0

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
