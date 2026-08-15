//! `x86_64` borrow propagation kernel (inline assembly).
//!
//! Resolves the overwhelmingly common first-limb stop before entering assembly,
//! then uses fast `subq $1`/`jnc` early-exit propagation for extended borrow chains.

use core::arch::asm;

use super::Limb;

/// Propagate a binary borrow through a raw limb pointer slice.
///
/// Computes:
///
/// ```text
///   (borrow_out, dst[0..len]) = dst[0..len] - borrow
/// ```
///
/// # Microarchitectural Strategy
///
/// Borrow propagation terminates at the first limb that does not underflow (i.e. where `dst[i] != 0`).
/// Statistically, $1 - 2^{-64}$ of all borrow propagations stop at limb 0 without entering assembly.
/// For rare borrow chains traversing multiple zero limbs, the assembly loop uses `subq $1` with
/// jump-if-not-carry (`jnc`) to break immediately on the first non-underflowing limb.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `borrow` must be a valid binary borrow ($\in \{0, 1\}$).
#[allow(
    unsafe_code,
    reason = "Hardware inline assembly natively requires unsafe code"
)]
#[allow(
    clippy::inline_always,
    reason = "Critical for peak performance in borrow propagation hot paths"
)]
#[inline(always)]
pub unsafe fn propagate_borrow_unchecked(dst: *mut Limb, len: usize, mut borrow: Limb) -> Limb {
    debug_assert!(borrow <= 1, "borrow propagation accepts a binary borrow");
    if len == 0 {
        return borrow;
    }
    // A borrow reaches the second limb iff dst[0] was zero. Handling this
    // scalar first step avoids entering assembly with probability 1 - 1/B,
    // where B = 2^Limb::BITS.
    // SAFETY: the caller guarantees dst is readable and writable for len > 0.
    let (first_difference, first_underflow) = unsafe { (*dst).overflowing_sub(borrow) };
    // SAFETY: the caller guarantees the first destination limb is writable.
    unsafe {
        *dst = first_difference;
    }
    if !first_underflow {
        return 0;
    }
    if len == 1 {
        return 1;
    }

    // SAFETY: len > 1 proves the one-limb offset remains within the allocation.
    let tail_dst = unsafe { dst.add(1) };
    let tail_len = len.wrapping_sub(1);
    // SAFETY: Caller guarantees `tail_dst` is valid for `tail_len` elements.
    unsafe {
        asm!(
            "1:",                                        // Loop head label
            "movq ({dst}), %rax",                        // Load dst[i]
            "subq $1, %rax",                             // Subtract 1 from limb (0 -> -1 sets CF=1)
            "movq %rax, ({dst})",                        // Write updated limb back to dst[i]
            "jnc 2f",                                    // If no borrow occurred (CF=0), borrow absorbed! Exit early
            "leaq 8({dst}), {dst}",                      // Advance pointer by 8 bytes
            "decq {len}",                                // Decrement remaining limb count
            "jnz 1b",                                    // Repeat while len != 0

            // All limbs underflowed from zero to MAX: final borrow out is 1
            "movq $1, {borrow}",                         // Set borrow out to 1
            "jmp 3f",                                    // Jump to completion label

            // Early exit: borrow absorbed by non-zero limb
            "2:",                                        // Early exit label
            "movq $0, {borrow}",                         // Set borrow out to 0

            // Completion
            "3:",                                        // Completion label
            borrow = out(reg) borrow,
            dst = inout(reg) tail_dst => _,
            len = inout(reg) tail_len => _,
            out("rax") _,
            options(nostack, att_syntax)
        );
    }
    borrow
}
