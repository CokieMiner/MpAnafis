//! `s390x` (IBM Z mainframe z14 / z15 / z16) fused multiply-add limb kernel.
//!
//! Uses 64×64→128-bit hardware multiplication (`mlgr` on even/odd register pair `%r2:%r3`),
//! logical addition with carry (`algr`/`alcgr`), and zero-overhead count branching (`brctg`).

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// Computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// IBM Z architecture uses the `mlgr` instruction requiring an even/odd register pair
/// (`%r2` = high 64 bits, `%r3` = low 64 bits). Addition uses logical additions (`algr`)
/// which set the 2-bit Condition Code (`CC`), immediately followed by `alcgr` (Add Logical
/// with Carry) against a zero register to absorb carry bits into the high product `%r2`.
/// The loop is 2-way unrolled (16 bytes per iteration) with `brctg` for hardware branch prediction.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in s390x multi-precision hot paths"
)]
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

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",      // Compare immediate and jump if chunks == 0 to remainder (1f)

            ".p2align 4",
            // Main 2-way unrolled loop body
            "2:",

            // [Limb 0 Multiply-Accumulate]
            "lg {src_v}, 0({src})",         // Load src[0] (64-bit Load Grand)
            "lg {dst_v}, 0({dst})",         // Load dst[0]
            "lgr %r3, {src_v}",             // Move src[0] into odd register %r3 (multiplicand)
            "mlgr %r2, {scalar}",           // %r2:%r3 = %r3 * scalar (128-bit product: %r2=hi, %r3=lo)
            "algr %r3, {carry}",            // %r3 += carry (Add Logical Grand, sets CC)
            "alcgr %r2, {zero}",            // %r2 += 0 + CC (Add Logical with Carry into high product)
            "algr {dst_v}, %r3",            // dst[0] += %r3, sets CC
            "alcgr %r2, {zero}",            // %r2 += 0 + CC (absorb destination carry into %r2)
            "lgr {carry}, %r2",             // carry = %r2 (updated running carry)
            "stg {dst_v}, 0({dst})",        // Store accumulated result to dst[0]

            // [Limb 1 Multiply-Accumulate]
            "lg {src_v}, 8({src})",         // Load src[1]
            "lg {dst_v}, 8({dst})",         // Load dst[1]
            "lgr %r3, {src_v}",             // Move src[1] into %r3
            "mlgr %r2, {scalar}",           // %r2:%r3 = %r3 * scalar
            "algr %r3, {carry}",            // %r3 += carry, sets CC
            "alcgr %r2, {zero}",            // %r2 += 0 + CC
            "algr {dst_v}, %r3",            // dst[1] += %r3, sets CC
            "alcgr %r2, {zero}",            // %r2 += 0 + CC
            "lgr {carry}, %r2",             // carry = %r2
            "stg {dst_v}, 8({dst})",        // Store accumulated result to dst[1]

            // Advance pointers by 2 limbs (16 bytes)
            "la {src}, 16({src})",          // Load Address (increment src by 16)
            "la {dst}, 16({dst})",          // Load Address (increment dst by 16)
            "brctg {chunks}, 2b",           // Branch on Count Grand (decrement chunks and loop if != 0)

            // Remainder processing (0 or 1 limb)
            "1:",
            "cgij {rem}, 0, 8, 3f",         // Jump to end if rem == 0

            // 1-limb tail
            "lg {src_v}, 0({src})",         // Load single src limb
            "lg {dst_v}, 0({dst})",         // Load single dst limb
            "lgr %r3, {src_v}",             // Move into %r3
            "mlgr %r2, {scalar}",           // %r2:%r3 = %r3 * scalar
            "algr %r3, {carry}",            // Add incoming carry, sets CC
            "alcgr %r2, {zero}",            // Propagate carry bit
            "algr {dst_v}, %r3",            // Accumulate into destination limb, sets CC
            "alcgr %r2, {zero}",            // Propagate carry bit
            "lgr {carry}, %r2",             // Update running carry
            "stg {dst_v}, 0({dst})",        // Store updated limb

            // Tail completion
            "3:",

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
