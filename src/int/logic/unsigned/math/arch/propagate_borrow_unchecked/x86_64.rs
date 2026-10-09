//! `x86_64` borrow propagation kernel (inline assembly).
//!
//! Resolves first-limb termination before entering assembly, then uses
//! `subq $1`/`jnc` to stop when the borrow is absorbed.

#![expect(unsafe_code, reason = "The kernel accesses caller-proven raw spans and x86-64 assembly")]

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
/// Borrow propagation terminates at the first limb that does not underflow (i.e. where `dst[i] != 0`).
/// For chains traversing multiple zero limbs, the assembly loop uses `subq $1` with
/// jump-if-not-carry (`jnc`) to break immediately on the first non-underflowing limb.
///
/// # Safety
///
/// - For nonzero `len`, `dst` covers `len` aligned, initialized, writable limbs
///   within `isize::MAX` bytes.
/// - `borrow` must be a valid binary borrow ($\in \{0, 1\}$).
#[expect(
    clippy::inline_always,
    reason = "Inlining retains the first-limb stopping path at its arithmetic caller"
)]
#[inline(always)]
pub unsafe fn propagate_borrow_unchecked(dst: *mut Limb, len: usize, mut borrow: Limb) -> Limb {
    debug_assert!(borrow <= 1, "borrow propagation accepts a binary borrow");
    if len == 0 {
        return borrow;
    }
    // A borrow reaches the second limb iff dst[0] was zero. Handling this
    // scalar first step resolves every chain absorbed at the first limb.
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
    // SAFETY: len > 1 makes the one-limb length reduction exact.
    let tail_len = unsafe { len.unchecked_sub(1) };
    // SAFETY: the writable initialized tail contains tail_len > 0 limbs.
    // Each iteration consumes one limb and stops on a cleared borrow or the
    // exhausted count; pointer advances reach at most one past the span.
    // Every modified register is an output, and the stack is untouched.
    unsafe {
        asm!(
            "1:",                                        // Loop head label
            "movq ({dst}), %rax",                        // Load dst[i]
            "subq $1, %rax",                             // Subtract 1 from limb (0 -> -1 sets CF=1)
            "movq %rax, ({dst})",                        // Write updated limb back to dst[i]
            "jnc 2f",                                    // Stop when CF=0 absorbs the borrow
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
