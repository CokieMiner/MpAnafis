//! `x86_64` carry propagation kernel (inline assembly).
//!
//! Resolves the overwhelmingly common first-limb stop before entering assembly,
//! then uses fast `incq`/`jnz` early-exit propagation for extended carry chains.

use core::arch::asm;

use super::Limb;

/// Propagate a binary carry through a raw limb pointer slice.
///
/// Computes:
///
/// ```text
///   (carry_out, dst[0..len]) = dst[0..len] + carry
/// ```
///
/// # Microarchitectural Strategy
///
/// Carry propagation terminates at the first limb that does not wrap (i.e. where `dst[i] != Limb::MAX`).
/// Statistically, $1 - 2^{-64}$ of all carry propagations stop at limb 0 without requiring loop entry.
/// For rare chains traversing multiple MAX limbs, the assembly loop uses single-uOp `incq` with
/// zero-flag conditional branching (`jnz`) to break immediately on the first non-wrapping limb.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `carry` must be a valid binary carry ($\in \{0, 1\}$).
#[allow(
    unsafe_code,
    reason = "Hardware inline assembly natively requires unsafe code"
)]
#[allow(
    clippy::inline_always,
    reason = "Critical for peak performance in carry propagation hot paths"
)]
#[inline(always)]
pub unsafe fn propagate_carry_unchecked(dst: *mut Limb, len: usize, mut carry: Limb) -> Limb {
    debug_assert!(carry <= 1, "carry propagation accepts a binary carry");
    if len == 0 {
        return carry;
    }
    // A carry reaches the second limb iff dst[0] was Limb::MAX. Handling this
    // scalar first step here avoids entering the unrolled assembly for the
    // common probability 1 - 1/B, where B = 2^Limb::BITS.
    // SAFETY: the caller guarantees dst is readable and writable for len > 0.
    let (first_sum, first_overflow) = unsafe { (*dst).overflowing_add(carry) };
    // SAFETY: the caller guarantees the first destination limb is writable.
    unsafe {
        *dst = first_sum;
    }
    if !first_overflow {
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
            "incq %rax",                                 // Increment limb (Limb::MAX -> 0 sets ZF=1)
            "movq %rax, ({dst})",                        // Write updated limb back to dst[i]
            "jnz 2f",                                    // If result != 0 (ZF=0), carry absorbed! Exit early
            "leaq 8({dst}), {dst}",                      // Advance pointer by 8 bytes
            "decq {len}",                                // Decrement remaining limb count
            "jnz 1b",                                    // Repeat while len != 0

            // All limbs wrapped to zero: final carry out is 1
            "movq $1, {carry}",                          // Set carry out to 1
            "jmp 3f",                                    // Jump to completion label

            // Early exit: carry absorbed by non-MAX limb
            "2:",                                        // Early exit label
            "movq $0, {carry}",                          // Set carry out to 0

            // Completion
            "3:",                                        // Completion label
            carry = out(reg) carry,
            dst = inout(reg) tail_dst => _,
            len = inout(reg) tail_len => _,
            out("rax") _,
            options(nostack, att_syntax)
        );
    }
    carry
}
