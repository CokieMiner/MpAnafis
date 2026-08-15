//! Baseline generic x86-64 fused multiply-subtract limb kernel.
//!
//! Uses hardware `mulq` ($64 \times 64 \to 128$-bit in `%rdx:%rax`) with 4-way unrolling
//! and interleaved load latency hiding for legacy and pre-BMI2 x86-64 processors.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// Computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// Uses the vanilla `mulq` instruction with `%rax` as the implicit multiplicand and `%rdx:%rax`
/// receiving the 128-bit product. The loop is 4-way unrolled (32 bytes per iteration), hoisting
/// destination memory loads into early registers to hide memory-to-ALU latency bubbles.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in multi-precision hot paths"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry_hi: Limb;
    let mut borrow: Limb;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`, `24`, `32`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "xorl {carry_hi:e}, {carry_hi:e}",           // Zero carry_hi register
            "xorl {borrow:e}, {borrow:e}",               // Zero borrow register
            "decq {chunks}",                             // Pre-decrement chunk counter
            "js 1f",                                     // If chunks < 0, skip to remainder (1f)

            // Main 4-way unrolled loop body
            "2:",                                        // Loop head label
            // [Limb 0]
            "movq {scalar}, %rax",                       // Load scalar into %rax (implicit operand for mulq)
            "mulq 0({src})",                             // %rdx:%rax = src[0] * scalar
            "movq 0({dst}), %rcx",                       // Load dst[0] to hide multiplier latency
            "addq {carry_hi}, %rax",                     // %rax += carry_hi
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry_hi}",                     // Update carry_hi
            "addq {borrow}, %rax",                       // %rax += borrow
            "movq $0, {borrow}",                         // Clear borrow
            "adcq $0, {borrow}",                         // Capture add overflow
            "subq %rax, %rcx",                           // %rcx = dst[0] - %rax
            "movq %rcx, 0({dst})",                       // Store result back to dst[0]
            "adcq $0, {borrow}",                         // Capture subtraction borrow

            // [Limb 1]
            "movq {scalar}, %rax",                       // Reload scalar
            "mulq 8({src})",                             // %rdx:%rax = src[1] * scalar
            "movq 8({dst}), %rcx",                       // Load dst[1] to hide multiplier latency
            "addq {carry_hi}, %rax",                     // %rax += carry_hi
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry_hi}",                     // Update carry_hi
            "addq {borrow}, %rax",                       // %rax += borrow
            "movq $0, {borrow}",                         // Clear borrow
            "adcq $0, {borrow}",                         // Capture add overflow
            "subq %rax, %rcx",                           // %rcx = dst[1] - %rax
            "movq %rcx, 8({dst})",                       // Store result back to dst[1]
            "adcq $0, {borrow}",                         // Capture subtraction borrow

            // [Limb 2]
            "movq {scalar}, %rax",                       // Reload scalar
            "mulq 16({src})",                            // %rdx:%rax = src[2] * scalar
            "movq 16({dst}), %rcx",                      // Load dst[2] to hide multiplier latency
            "addq {carry_hi}, %rax",                     // %rax += carry_hi
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry_hi}",                     // Update carry_hi
            "addq {borrow}, %rax",                       // %rax += borrow
            "movq $0, {borrow}",                         // Clear borrow
            "adcq $0, {borrow}",                         // Capture add overflow
            "subq %rax, %rcx",                           // %rcx = dst[2] - %rax
            "movq %rcx, 16({dst})",                      // Store result back to dst[2]
            "adcq $0, {borrow}",                         // Capture subtraction borrow

            // [Limb 3]
            "movq {scalar}, %rax",                       // Reload scalar
            "mulq 24({src})",                            // %rdx:%rax = src[3] * scalar
            "movq 24({dst}), %rcx",                      // Load dst[3] to hide multiplier latency
            "addq {carry_hi}, %rax",                     // %rax += carry_hi
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry_hi}",                     // Update carry_hi
            "addq {borrow}, %rax",                       // %rax += borrow
            "movq $0, {borrow}",                         // Clear borrow
            "adcq $0, {borrow}",                         // Capture add overflow
            "subq %rax, %rcx",                           // %rcx = dst[3] - %rax
            "movq %rcx, 24({dst})",                      // Store result back to dst[3]
            "adcq $0, {borrow}",                         // Capture subtraction borrow

            "leaq 32({src}), {src}",                     // Advance src pointer by 32 bytes
            "leaq 32({dst}), {dst}",                     // Advance dst pointer by 32 bytes
            "decq {chunks}",                             // Decrement chunks
            "jns 2b",                                    // Repeat while chunks >= 0

            // Remainder processing entry point (0 to 3 limbs)
            "1:",                                        // Remainder entry label
            "decq {rem}",                                // Pre-decrement remainder counter
            "js 4f",                                     // If rem < 0, skip to finish (4f)

            // 1-limb unrolled tail loop
            "3:",                                        // Tail loop label
            "movq {scalar}, %rax",                       // Load scalar into %rax
            "mulq 0({src})",                             // Multiply single src limb
            "movq 0({dst}), %rcx",                       // Load single dst limb to hide multiplier latency
            "addq {carry_hi}, %rax",                     // %rax += carry_hi
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry_hi}",                     // Update carry_hi
            "addq {borrow}, %rax",                       // %rax += borrow
            "movq $0, {borrow}",                         // Clear borrow
            "adcq $0, {borrow}",                         // Capture add overflow
            "subq %rax, %rcx",                           // dst[0] - %rax
            "movq %rcx, 0({dst})",                       // Store updated limb
            "adcq $0, {borrow}",                         // Capture subtraction borrow
            "leaq 8({src}), {src}",                      // Advance src pointer (+8)
            "leaq 8({dst}), {dst}",                      // Advance dst pointer (+8)
            "decq {rem}",                                // Decrement remainder counter
            "jns 3b",                                    // Repeat while rem >= 0

            // Tail completion
            "4:",                                        // Completion label

            carry_hi = out(reg) carry_hi,
            borrow = out(reg) borrow,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            scalar = in(reg) scalar,
            out("rax") _,
            out("rdx") _,
            out("rcx") _,
            options(nostack, att_syntax)
        );
    }
    (carry_hi, borrow)
}
