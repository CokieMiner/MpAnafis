//! `s390x` (IBM Z z/Architecture) fused multiply-subtract limb kernel.
//!
//! Uses 64x64->128-bit hardware multipliers (`mlgr` on even/odd register pair `%r2:%r3`),
//! logical addition with carry (`algr`/`alcgr`), and borrow capture (`slgr`/`slbgr`/`ogr`/`lcgr`).

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from `dst`,
/// and return the final `(carry, borrow)` pair.
///
/// For B = 2^64, `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
/// R2:R3 holds each product; two-limb blocks retain product carry and binary
/// borrow separately.
///
/// # Safety
///
/// Both pointers must cover `len` aligned, initialized limbs in disjoint spans;
/// `dst` requires exclusive access. Each span's byte length must fit in `isize`.
/// Zero length permits null pointers.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "short and unrolled assembly paths share one call boundary in the arithmetic hot path"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    if len < 4 {
        if len == 0 {
            return (0, 0);
        }
        let carry: Limb;
        let borrow: Limb;
        // SAFETY: 1 <= len <= 3 bounds all accesses to the caller's aligned initialized
        // disjoint spans. r2:r3 is the declared product pair. lghi preserves
        // the condition code when extracting addition carry and subtraction
        // borrow; low-word overflow and subtraction borrow are disjoint.
        unsafe {
            asm!(
                "lgr %r3, {scalar}",
                "mlg %r2, 0({src})",
                "lg {d}, 0({dst})",
                "lgr {carry}, %r2",
                "slgr {d}, %r3",
                "lghi {borrow}, 0",
                "slbgr {borrow}, {borrow}",
                "lcgr {borrow}, {borrow}",
                "stg {d}, 0({dst})",
                "brctg {count}, 2f",
                "j 3f",
                "2:",
                "la {src}, 8({src})",
                "la {dst}, 8({dst})",
                "lgr %r3, {scalar}",
                "mlg %r2, 0({src})",
                "lg {d}, 0({dst})",
                "lghi {temp}, 0",
                "algr %r3, {carry}",
                "alcgr %r2, {temp}",
                "lgr {carry}, %r2",
                "algr %r3, {borrow}",
                "lghi {borrow}, 0",
                "alcgr {borrow}, {borrow}",
                "slgr {d}, %r3",
                "lghi {temp}, 0",
                "slbgr {temp}, {temp}",
                "lcgr {temp}, {temp}",
                "ogr {borrow}, {temp}",
                "stg {d}, 0({dst})",
                "brctg {count}, 2b",
                "3:",
                src = inout(reg_addr) src => _,
                dst = inout(reg_addr) dst => _,
                count = inout(reg) len => _,
                scalar = in(reg) scalar,
                carry = out(reg) carry,
                borrow = out(reg) borrow,
                d = out(reg) _,
                temp = out(reg) _,
                out("r2") _,
                out("r3") _,
                options(nostack),
            );
        }
        return (carry, borrow);
    }
    let mut carry: Limb = 0;
    let mut borrow: Limb = 0;
    let chunks = len >> 1;
    let rem = len & 1;
    let zero: Limb = 0;

    // SAFETY: len >= 4 gives at least two complete pairs and at most one tail
    // limb in the aligned initialized disjoint spans. Product plus carry is
    // below B^2. LGHI preserves CC while SLBGR extracts each subtraction's
    // borrow mask; OR and negation convert their union to a binary borrow.
    // R2:R3 and early temporaries are distinct from all live inputs.
    unsafe {
        asm!(
            ".p2align 4",
            // Main 2-way unrolled loop body
            "2:",

            // [Limb 0 Multiply-Subtract]
            "lg {src_v}, 0({src})",                      // Load src[0]
            "lg {dst_v}, 0({dst})",                      // Load dst[0]
            "lgr %r3, {src_v}",                          // %r3 = src[0]
            "mlgr %r2, {scalar}",                        // %r2:%r3 = %r3 * scalar (128-bit product)
            "algr %r3, {carry}",                         // %r3 += carry, set Condition Code (CC)
            "alcgr %r2, {zero}",                         // %r2 += CC carry + 0
            "lgr {carry}, %r2",                          // Update running multiplication carry

            "slgr {dst_v}, {borrow}",                    // dst_v -= incoming borrow
            "lghi {borrow_tmp}, 0",                      // Clear borrow_tmp
            "slbgr {borrow_tmp}, {borrow_tmp}",          // borrow_tmp = 0 or -1 (first borrow mask)
            "slgr {dst_v}, %r3",                         // dst_v -= low product
            "lghi {borrow}, 0",                          // Clear borrow
            "slbgr {borrow}, {borrow}",                  // borrow = 0 or -1 (second borrow mask)
            "ogr {borrow}, {borrow_tmp}",                // Combine borrow masks (-1 if either borrowed)
            "lcgr {borrow}, {borrow}",                   // Negate mask: -1 -> 1, 0 -> 0
            "stg {dst_v}, 0({dst})",                     // Store updated dst[0]

            // [Limb 1 Multiply-Subtract]
            "lg {src_v}, 8({src})",                      // Load src[1]
            "lg {dst_v}, 8({dst})",                      // Load dst[1]
            "lgr %r3, {src_v}",                          // %r3 = src[1]
            "mlgr %r2, {scalar}",                        // %r2:%r3 = %r3 * scalar
            "algr %r3, {carry}",                         // %r3 += carry
            "alcgr %r2, {zero}",                         // %r2 += CC carry
            "lgr {carry}, %r2",                          // Update carry

            "slgr {dst_v}, {borrow}",                    // dst_v -= borrow
            "lghi {borrow_tmp}, 0",                      // Clear borrow_tmp
            "slbgr {borrow_tmp}, {borrow_tmp}",          // First borrow mask
            "slgr {dst_v}, %r3",                         // dst_v -= low product
            "lghi {borrow}, 0",                          // Clear borrow
            "slbgr {borrow}, {borrow}",                  // Second borrow mask
            "ogr {borrow}, {borrow_tmp}",                // Combine borrow masks
            "lcgr {borrow}, {borrow}",                   // Convert mask to 0 or 1
            "stg {dst_v}, 8({dst})",                     // Store updated dst[1]

            // Advance pointers by two limbs and decrement the loop register.
            "la {src}, 16({src})",                       // Advance src pointer by 16 bytes
            "la {dst}, 16({dst})",                       // Advance dst pointer by 16 bytes
            "brctg {chunks}, 2b",                        // Decrement chunks and branch if > 0

            // Remainder processing (0 or 1 limb)
            "1:",
            "cgij {rem}, 0, 8, 3f",                      // If rem == 0, skip to end (3f)

            // 1-limb tail
            "lg {src_v}, 0({src})",                      // Load single src limb
            "lg {dst_v}, 0({dst})",                      // Load single dst limb
            "lgr %r3, {src_v}",                          // Multiplicand
            "mlgr %r2, {scalar}",                        // 64x64->128 product
            "algr %r3, {carry}",                         // Add carry
            "alcgr %r2, {zero}",                         // Propagate carry
            "lgr {carry}, %r2",                          // Update carry

            "slgr {dst_v}, {borrow}",                    // Subtract incoming borrow
            "lghi {borrow_tmp}, 0",
            "slbgr {borrow_tmp}, {borrow_tmp}",          // Capture first borrow
            "slgr {dst_v}, %r3",                         // Subtract low product
            "lghi {borrow}, 0",
            "slbgr {borrow}, {borrow}",                  // Capture second borrow
            "ogr {borrow}, {borrow_tmp}",                // Combine borrows
            "lcgr {borrow}, {borrow}",                   // Convert to 0 or 1
            "stg {dst_v}, 0({dst})",                     // Store updated limb

            // Tail completion
            "3:",

            carry = inout(reg) carry,
            borrow = inout(reg) borrow,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg_addr) src => _,
            dst = inout(reg_addr) dst => _,
            scalar = in(reg) scalar,
            zero = in(reg) zero,
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
