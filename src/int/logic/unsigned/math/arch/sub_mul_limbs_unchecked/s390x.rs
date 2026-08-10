//! `s390x` multiply-subtract limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from `dst`,
/// and return the final `(carry, borrow)` pair.
///
/// This computes:
///
/// ```text
///   (carry, borrow, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// The first element of the return tuple is the **high carry** from the
/// multiplication chain (the pending high half of `src[len-1] × scalar`).
/// The second element is the **final borrow** from the subtraction chain.
/// Both must be propagated by the caller.
///
/// This is the `s390x` (IBM Z) inline assembly implementation. It utilizes `mlgr`
/// for 64x64->128-bit multiplication (strictly requiring an even/odd register pair,
/// `%r2` and `%r3`). For subtraction, it uses `slgr` (subtract logical) and `slbgr`
/// (subtract logical with borrow) to propagate the borrow bit branchlessly.
///
/// The loop is **2-way unrolled** using `brctg` for zero-overhead loop control.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(clippy::inline_always, reason = "Performance critical inner loop")]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry: Limb = 0;
    let mut borrow: Limb = 0;
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
            "lgr %r3, {src_v}",
            "mlgr %r2, {scalar}",
            "algr %r3, {carry}",            // r3 += carry
            "alcgr %r2, {zero}",            // r2 += CC (multiplication carry out)
            "lgr {carry}, %r2",             // carry = r2

            "slgr {dst_v}, {borrow}",       // dst_v -= borrow
            "lghi {borrow_tmp}, 0",
            "slbgr {borrow_tmp}, {borrow_tmp}", // borrow_tmp = 0 or -1
            "slgr {dst_v}, %r3",            // dst_v -= low product
            "lghi {borrow}, 0",
            "slbgr {borrow}, {borrow}",     // borrow = 0 or -1
            "ogr {borrow}, {borrow_tmp}",   // combine borrows (-1 if any borrowed)
            "lcgr {borrow}, {borrow}",      // negate to make it 0 or 1
            "stg {dst_v}, 0({dst})",

            // Limb 1
            "lg {src_v}, 8({src})",
            "lg {dst_v}, 8({dst})",
            "lgr %r3, {src_v}",
            "mlgr %r2, {scalar}",
            "algr %r3, {carry}",
            "alcgr %r2, {zero}",
            "lgr {carry}, %r2",

            "slgr {dst_v}, {borrow}",
            "lghi {borrow_tmp}, 0",
            "slbgr {borrow_tmp}, {borrow_tmp}",
            "slgr {dst_v}, %r3",
            "lghi {borrow}, 0",
            "slbgr {borrow}, {borrow}",
            "ogr {borrow}, {borrow_tmp}",
            "lcgr {borrow}, {borrow}",
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
            "lgr {carry}, %r2",

            "slgr {dst_v}, {borrow}",
            "lghi {borrow_tmp}, 0",
            "slbgr {borrow_tmp}, {borrow_tmp}",
            "slgr {dst_v}, %r3",
            "lghi {borrow}, 0",
            "slbgr {borrow}, {borrow}",
            "ogr {borrow}, {borrow_tmp}",
            "lcgr {borrow}, {borrow}",
            "stg {dst_v}, 0({dst})",

            "3:",                           // --- End ---

            carry = inout(reg) carry,
            borrow = inout(reg) borrow,
            zero = inout(reg) zero => _,
            dst = inout(reg_addr) dst => _,
            src = inout(reg_addr) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            scalar = inout(reg) scalar => _,
            src_v = out(reg) _,
            dst_v = out(reg) _,
            borrow_tmp = out(reg) _,
            out("r2") _,
            out("r3") _,
            options(nostack)
        );
    }
    (carry, borrow)
}
