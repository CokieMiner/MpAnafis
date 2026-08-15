//! 32-bit x86 write-only dual-row multiplication kernel.
//!
//! Evaluates `dst = src * (s0 + s1 * B)` in a single write-only pass using
//! 32-bit `mull` ($32 \times 32 \to 64$-bit into `%edx:%eax`) and 3-word stack state.

use core::arch::asm;

use super::Limb;

/// Write `src * (s0 + s1 * B)` into `dst` without reading its old contents.
///
/// Computes:
///
/// ```text
///   dst[0..len+2] = src[0..len] × (s0 + s1 × 2^32)
/// ```
///
/// # Microarchitectural Strategy
///
/// Evaluates two simultaneous multiplication rows in registers without memory reads of `dst`.
/// Under 32-bit x86 register pressure, `%eax` and `%edx` are reserved for `mull`, while `{dst}`,
/// `{src}`, `{carry0}`, and `{carry1}` consume the remaining 4 allocatable GPRs.
///
/// # Safety
///
/// - `dst` must point to a writable buffer of at least `len + 2` initialized 32-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 32-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "This repeatedly invoked hot kernel must inline, and its write-only dual-row assembly stays visibly unrolled so the initialization invariant remains reviewable"
)]
#[inline(always)]
pub unsafe fn mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) {
    if len == 0 {
        return;
    }

    if len == 1 {
        // SAFETY:
        // 1. `dst` is valid for writes of 3 limbs.
        // 2. `src` is valid for reads of 1 limb.
        // 3. Memory spans are non-overlapping.
        unsafe {
            asm!(
                "mull 0({src})",                         // %edx:%eax = src[0] * s0 (64-bit product)
                "movl %eax, 0({dst})",                   // Store dst[0]
                "movl %edx, {carry0}",                   // carry0 = high product row 0
                "movl {s1}, %eax",                       // %eax = s1
                "mull 0({src})",                         // %edx:%eax = src[0] * s1 (64-bit product row 1)
                "addl {carry0}, %eax",                   // %eax += carry0
                "adcl $0, %edx",                         // %edx += CF
                "movl %eax, 4({dst})",                   // Store dst[1]
                "movl %edx, 8({dst})",                   // Store dst[2]
                inlateout("eax") s0 => _,
                out("edx") _,
                dst = in(reg) dst,
                src = in(reg) src,
                s1 = in(reg) s1,
                carry0 = out(reg) _,
                options(nostack, att_syntax)
            );
        }
        return;
    }

    // SAFETY:
    // 1. `dst` is valid for writes of `len + 2` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Pushed stack words are restored before return (`addl $12, %esp`).
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "pushl {s0}",                                // Save s0 at 8(%esp)
            "pushl {s1}",                                // Save s1 at 4(%esp)
            "pushl {len}",                               // Save len at 0(%esp)

            // Seed both rows from src[0]
            "movl 8(%esp), %eax",                        // Load s0 from stack
            "mull 0({src})",                             // %edx:%eax = src[0] * s0
            "movl %eax, 0({dst})",                       // Store dst[0]
            "movl %edx, {carry0}",                       // carry0 = high product row 0
            "movl 4(%esp), %eax",                        // Load s1 from stack
            "mull 0({src})",                             // %edx:%eax = src[0] * s1
            "movl %eax, 4({dst})",                       // Store intermediate dst[1]
            "movl %edx, {carry1}",                       // carry1 = high product row 1
            "addl $4, {src}",                            // Advance src pointer by 4 bytes
            "addl $4, {dst}",                            // Advance dst pointer by 4 bytes
            "decl 0(%esp)",                              // Decrement remaining count

            // Main 2-way unrolled loop body
            "cmpl $2, 0(%esp)",                          // Check if remaining >= 2
            "jb 2f",                                     // If not, skip to remainder (2f)

            "1:",
            // [Limb 0 - Row 0]
            "movl 8(%esp), %eax",                        // Load s0
            "mull 0({src})",                             // src[j] * s0
            "addl {carry0}, %eax",                       // %eax += carry0
            "adcl $0, %edx",                             // %edx += CF
            "addl 0({dst}), %eax",                       // %eax += dst[j]
            "adcl $0, %edx",                             // %edx += CF
            "movl %eax, 0({dst})",                       // Store finalized dst[j]
            "movl %edx, {carry0}",                       // Update carry0

            // [Limb 0 - Row 1]
            "movl 4(%esp), %eax",                        // Load s1
            "mull 0({src})",                             // src[j] * s1
            "addl {carry1}, %eax",                       // %eax += carry1
            "adcl $0, %edx",                             // %edx += CF
            "movl %eax, 4({dst})",                       // Store intermediate dst[j+1]
            "movl %edx, {carry1}",                       // Update carry1

            // [Limb 1 - Row 0]
            "movl 8(%esp), %eax",                        // Load s0
            "mull 4({src})",                             // src[j+1] * s0
            "addl {carry0}, %eax",                       // %eax += carry0
            "adcl $0, %edx",                             // %edx += CF
            "addl 4({dst}), %eax",                       // %eax += dst[j+1]
            "adcl $0, %edx",                             // %edx += CF
            "movl %eax, 4({dst})",                       // Store finalized dst[j+1]
            "movl %edx, {carry0}",                       // Update carry0

            // [Limb 1 - Row 1]
            "movl 4(%esp), %eax",                        // Load s1
            "mull 4({src})",                             // src[j+1] * s1
            "addl {carry1}, %eax",                       // %eax += carry1
            "adcl $0, %edx",                             // %edx += CF
            "movl %eax, 8({dst})",                       // Store intermediate dst[j+2]
            "movl %edx, {carry1}",                       // Update carry1

            // Advance pointers by 2 limbs (8 bytes)
            "addl $8, {src}",
            "addl $8, {dst}",
            "subl $2, 0(%esp)",                          // Decrement remaining count on stack
            "cmpl $2, 0(%esp)",                          // Check if remaining >= 2
            "jae 1b",                                    // Repeat loop

            // Remainder processing (0 or 1 limb)
            "2:",
            "cmpl $0, 0(%esp)",                          // Check if remaining == 0
            "je 4f",                                     // If 0, skip to epilogue (4f)

            // 1-limb tail
            "3:",
            "movl 8(%esp), %eax",                        // Load s0
            "mull 0({src})",                             // src[j] * s0
            "addl {carry0}, %eax",                       // Add carry0
            "adcl $0, %edx",                             // Propagate carry
            "addl 0({dst}), %eax",                       // Accumulate into dst
            "adcl $0, %edx",
            "movl %eax, 0({dst})",                       // Store finalized limb
            "movl %edx, {carry0}",                       // Update carry0

            "movl 4(%esp), %eax",                        // Load s1
            "mull 0({src})",                             // src[j] * s1
            "addl {carry1}, %eax",                       // Add carry1
            "adcl $0, %edx",                             // Propagate carry
            "movl %eax, 4({dst})",                       // Store intermediate limb
            "movl %edx, {carry1}",                       // Update carry1

            "addl $4, {src}",
            "addl $4, {dst}",

            // Epilogue: Flush trailing high row 1 limb + remaining carry
            "4:",
            "addl {carry0}, 0({dst})",                   // dst[len] += carry0
            "adcl $0, {carry1}",                         // carry1 += CF
            "movl {carry1}, 4({dst})",                   // Store final high limb dst[len+1]
            "addl $12, %esp",                            // Restore stack pointer

            carry0 = out(reg) _,
            carry1 = out(reg) _,
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
}
