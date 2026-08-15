//! `s390x` (IBM Z z/Architecture) fused multiply-subtract limb kernel.
//!
//! Uses 64×64→128-bit hardware multipliers (`mlgr` on even/odd register pair `%r2:%r3`),
//! logical addition with carry (`algr`/`alcgr`), and borrow capture (`slgr`/`slbgr`/`ogr`/`lcgr`).

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from `dst`,
/// and return the final `(carry, borrow)` pair.
///
/// Computes:
///
/// ```text
///   (carry, borrow, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// In IBM Z z/Architecture, `mlgr` requires an even/odd register pair (`%r2:%r3`), where `%r3` holds
/// the multiplicand and `%r2:%r3` receives the 128-bit product. Multiplicative carries are absorbed
/// using `algr` (Add Logical) and `alcgr` (Add Logical with Carry). Subtraction borrows are captured
/// using `slbgr` (Subtract Logical with Borrow) into branchless masks. Hardware loop control is driven
/// by `brctg` (Branch on Count 64-bit).
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in 64-bit s390x multi-precision hot paths"
)]
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

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",                   // If chunks == 0, skip to remainder loop (1f)

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

            // Advance pointers by 2 limbs (16 bytes) and loop via hardware CTR
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
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
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
