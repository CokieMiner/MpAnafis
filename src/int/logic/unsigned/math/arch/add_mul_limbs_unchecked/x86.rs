//! 32-bit x86 fused multiply-add limb kernel.
//!
//! Uses hardware 32-bit `mull` ($32 \times 32 \to 64$-bit into `%edx:%eax`)
//! and register carry propagation. The invariant scalar occupies one stack word,
//! leaving six general registers sufficient when EBP is a frame pointer.

use core::arch::asm;

use super::Limb;

/// Accumulates `src * scalar` into `dst` and returns the high carry limb.
///
/// Four limbs are processed per block, followed by a scalar tail.
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// Nonempty spans must cover `len` aligned, initialized limbs in disjoint live
/// allocations of at most `isize::MAX` bytes. The destination must be writable.
#[expect(
    clippy::inline_always,
    reason = "Keep the unrolled hardware row recurrence in the selected multiplication caller"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let carry: Limb;

    // SAFETY: each block consumes four aligned limbs within both initialized
    // spans; the remaining count bounds the scalar tail. Empty spans skip all
    // operand accesses. Stores target only the disjoint writable destination.
    // The scalar is saved before EAX changes, and the stack word is restored
    // on the sole exit. No instruction calls another function or escapes the block.
    unsafe {
        asm!(
            "pushl %eax",                                // Save the invariant scalar
            "xorl {carry}, {carry}",                     // Zero carry accumulator register
            "testl {len}, {len}",                        // Test if len == 0
            "jz 4f",                                     // If zero, jump to exit (4f)
            "cmpl $4, {len}",                            // Compare len with 4
            "jb 2f",                                     // If len < 4, jump to remainder loop (2f)

            // Main 4-way unrolled loop body
            "1:",                                        // Loop head label
            // [Limb 0]
            "movl 0({src}), %eax",                       // Load src[0] into %eax
            "mull 0(%esp)",                              // %edx:%eax = src[0] * scalar (64-bit product)
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 0({dst})",                       // dst[0] += low product sum
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // carry = %edx

            // [Limb 1]
            "movl 4({src}), %eax",                       // Load src[1]
            "mull 0(%esp)",                              // %edx:%eax = src[1] * scalar
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 4({dst})",                       // dst[1] += low product sum
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // carry = %edx

            // [Limb 2]
            "movl 8({src}), %eax",                       // Load src[2]
            "mull 0(%esp)",                              // %edx:%eax = src[2] * scalar
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 8({dst})",                       // dst[2] += low product sum
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // carry = %edx

            // [Limb 3]
            "movl 12({src}), %eax",                      // Load src[3]
            "mull 0(%esp)",                              // %edx:%eax = src[3] * scalar
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 12({dst})",                      // dst[3] += low product sum
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // carry = %edx

            "addl $16, {src}",                           // Advance src pointer by 16 bytes
            "addl $16, {dst}",                           // Advance dst pointer by 16 bytes
            "subl $4, {len}",                            // Decrement remaining length by 4
            "cmpl $4, {len}",                            // Check if len >= 4
            "jae 1b",                                    // Repeat while len >= 4

            // Remainder entry point (0 to 3 limbs)
            "2:",                                        // Remainder entry label
            "testl {len}, {len}",                        // Test if remaining len == 0
            "jz 4f",                                     // If zero, jump to exit (4f)

            // 1-limb unrolled tail loop
            "3:",                                        // Tail loop label
            "movl ({src}), %eax",                        // Load single src limb
            "mull 0(%esp)",                              // %edx:%eax = src * scalar
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, ({dst})",                        // dst += low product
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // carry = %edx
            "addl $4, {src}",                            // Advance src (+4)
            "addl $4, {dst}",                            // Advance dst (+4)
            "decl {len}",                                // Decrement len
            "jnz 3b",                                    // Repeat while len != 0

            // Tail completion
            "4:",                                        // Completion label
            "addl $4, %esp",                              // Restore the saved scalar word

            carry = out(reg) carry,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            len = inout(reg) len => _,
            inlateout("eax") scalar => _,
            out("edx") _,
            options(att_syntax)
        );
    }
    carry
}
