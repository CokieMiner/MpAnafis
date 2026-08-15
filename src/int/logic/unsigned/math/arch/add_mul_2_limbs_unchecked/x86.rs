//! 32-bit x86 fused dual-row multiply-add kernel.
//!
//! Accumulates two overlapping scalar-product rows in one source traversal
//! using 32-bit `mull` ($32 \times 32 \to 64$-bit into `%edx:%eax`) and 3-word stack state.

use core::arch::asm;

use super::Limb;

/// Accumulate two overlapping scalar-product rows in one source traversal.
///
/// Computes:
///
/// ```text
///   dst[0..len] += src[0..len] * s0 + c0
///   dst[1..len+1] += src[0..len] * s1 + c1
/// ```
///
/// # Microarchitectural Strategy
///
/// Under 32-bit x86 register pressure, `%eax` and `%edx` are reserved for `mull`, while `{dst}`,
/// `{src}`, `{carry0}`, and `{carry1}` consume the remaining 4 allocatable GPRs. The invariant scalars
/// (`s0`, `s1`) and the loop counter (`len`) reside on the top 3 words of the stack.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len + 1` initialized 32-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 32-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "This repeatedly invoked hot kernel must inline, and its dual-row assembly stays visibly unrolled so the carry-chain proof remains reviewable"
)]
#[inline(always)]
pub unsafe fn add_mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) -> (Limb, Limb) {
    if len == 0 {
        return (0, 0);
    }

    if len == 1 {
        let carry0: Limb;
        let carry1: Limb;
        // SAFETY:
        // 1. `dst` is valid for reads and writes of 2 limbs.
        // 2. `src` is valid for reads of 1 limb.
        // 3. Memory spans are non-overlapping.
        unsafe {
            asm!(
                "mull 0({src})",                         // %edx:%eax = src[0] * s0 (64-bit product)
                "addl %eax, 0({dst})",                   // dst[0] += %eax
                "adcl $0, %edx",                         // %edx += CF
                "movl %edx, {carry0}",                   // carry0 = %edx
                "movl {s1}, %eax",                       // %eax = s1
                "mull 0({src})",                         // %edx:%eax = src[0] * s1
                "addl %eax, 4({dst})",                   // dst[1] += %eax
                "adcl $0, %edx",                         // %edx += CF (carry1 in %edx)
                inlateout("eax") s0 => _,
                out("edx") carry1,
                dst = in(reg) dst,
                src = in(reg) src,
                s1 = in(reg) s1,
                carry0 = out(reg) carry0,
                options(nostack, att_syntax)
            );
        }
        return (carry0, carry1);
    }

    let carry0: Limb;
    let carry1: Limb;

    // SAFETY:
    // 1. `dst` is valid for reads and writes of `len + 1` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Stack pushes (`pushl`) are strictly restored before returning (`addl $12, %esp`).
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "pushl {s0}",                                // Save s0 at 8(%esp)
            "pushl {s1}",                                // Save s1 at 4(%esp)
            "pushl {len}",                               // Save len at 0(%esp)
            "xorl {carry0}, {carry0}",                   // Zero row 0 carry
            "xorl {carry1}, {carry1}",                   // Zero row 1 carry

            "cmpl $2, 0(%esp)",                          // Check if len < 2
            "jb 2f",                                     // If len < 2, skip to remainder (2f)

            // Main 2-way unrolled loop body
            "1:",

            // [Limb 0 - Row 0]
            "movl 8(%esp), %eax",                        // Load s0 from stack
            "mull 0({src})",                             // %edx:%eax = src[0] * s0
            "addl {carry0}, %eax",                       // %eax += carry0
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 0({dst})",                       // dst[0] += %eax
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry0}",                       // Update row 0 carry

            // [Limb 0 - Row 1]
            "movl 4(%esp), %eax",                        // Load s1 from stack
            "mull 0({src})",                             // %edx:%eax = src[0] * s1
            "addl {carry1}, %eax",                       // %eax += carry1
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 4({dst})",                       // dst[1] += %eax
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry1}",                       // Update row 1 carry

            // [Limb 1 - Row 0]
            "movl 8(%esp), %eax",                        // Load s0
            "mull 4({src})",                             // src[1] * s0
            "addl {carry0}, %eax",                       // %eax += carry0
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 4({dst})",                       // dst[1] += %eax
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry0}",                       // Update row 0 carry

            // [Limb 1 - Row 1]
            "movl 4(%esp), %eax",                        // Load s1
            "mull 4({src})",                             // src[1] * s1
            "addl {carry1}, %eax",                       // %eax += carry1
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 8({dst})",                       // dst[2] += %eax
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry1}",                       // Update row 1 carry

            // Advance pointers by 2 limbs (8 bytes)
            "addl $8, {src}",
            "addl $8, {dst}",
            "subl $2, 0(%esp)",                          // Decrement remaining count on stack
            "cmpl $2, 0(%esp)",                          // Check if remaining >= 2
            "jae 1b",                                    // Repeat loop

            // Remainder processing (0 or 1 limb)
            "2:",
            "cmpl $0, 0(%esp)",                          // Check if remaining == 0
            "je 4f",                                     // If 0, skip to cleanup (4f)

            // 1-limb tail
            "3:",
            "movl 8(%esp), %eax",                        // Load s0
            "mull 0({src})",                             // src[j] * s0
            "addl {carry0}, %eax",
            "adcl $0, %edx",
            "addl %eax, 0({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry0}",

            "movl 4(%esp), %eax",                        // Load s1
            "mull 0({src})",                             // src[j] * s1
            "addl {carry1}, %eax",
            "adcl $0, %edx",
            "addl %eax, 4({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry1}",

            // Cleanup stack
            "4:",
            "addl $12, %esp",                            // Restore 3 pushed words

            carry0 = lateout(reg) carry0,
            carry1 = lateout(reg) carry1,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            len = in(reg) len,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            out("eax") _,
            out("edx") _,
            options(att_syntax)
        );
    }
    (carry0, carry1)
}
