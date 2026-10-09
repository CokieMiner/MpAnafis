//! `x86_64` carry propagation kernel (inline assembly).
//!
//! Resolves first-limb termination before entering assembly, then uses
//! `incq`/`jnz` to stop when the carry is absorbed.

#![expect(unsafe_code, reason = "The kernel accesses caller-proven raw spans and x86-64 assembly")]

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
/// Carry propagation terminates at the first limb that does not wrap (i.e. where `dst[i] != Limb::MAX`).
/// For chains traversing multiple MAX limbs, the assembly loop uses `incq` with
/// zero-flag conditional branching (`jnz`) to break immediately on the first non-wrapping limb.
///
/// # Safety
///
/// - For nonzero `len`, `dst` covers `len` aligned, initialized, writable limbs
///   within `isize::MAX` bytes.
/// - `carry` must be a valid binary carry ($\in \{0, 1\}$).
#[expect(
    clippy::inline_always,
    reason = "Inlining retains the first-limb stopping path at its arithmetic caller"
)]
#[inline(always)]
pub unsafe fn propagate_carry_unchecked(dst: *mut Limb, len: usize, mut carry: Limb) -> Limb {
    debug_assert!(carry <= 1, "carry propagation accepts a binary carry");
    if len == 0 {
        return carry;
    }
    // A carry reaches the second limb iff dst[0] was Limb::MAX. Handling this
    // scalar first step resolves every chain absorbed at the first limb.
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
    // SAFETY: len > 1 makes the one-limb length reduction exact.
    let tail_len = unsafe { len.unchecked_sub(1) };
    // SAFETY: the writable initialized tail contains tail_len > 0 limbs.
    // Each iteration consumes one limb and stops on a cleared carry or the
    // exhausted count; pointer advances reach at most one past the span.
    // Every modified register is an output, and the stack is untouched.
    unsafe {
        asm!(
            "1:",                                        // Loop head label
            "movq ({dst}), %rax",                        // Load dst[i]
            "incq %rax",                                 // Increment limb (Limb::MAX -> 0 sets ZF=1)
            "movq %rax, ({dst})",                        // Write updated limb back to dst[i]
            "jnz 2f",                                    // Stop when the nonzero result absorbs the carry
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
