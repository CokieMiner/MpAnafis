//! Baseline generic x86-64 fused multiply-add limb kernel.
//!
//! Uses standard 1-operand `mulq` ($64 \times 64 \to 128$-bit into `%rdx:%rax`)
//! and sequential `addq`/`adcq` carry chains. Serves as the universal fallback
//! kernel for all x86-64 microarchitectures (including early AMD Opteron / Intel Core 2).

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
/// Uses the hardware `mulq` instruction where `%rax` is the implicit multiplicand
/// and the 128-bit product is placed into `%rdx:%rax`. The loop is unrolled 4-way:
/// early memory loads of destination limbs `dst[i]` are interleaved with multiplier
/// latency bubbles to hide memory-to-ALU latency.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized limbs.
/// - `src` must point to a readable buffer of at least `len` initialized limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in multi-precision hot paths"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry: Limb;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` `Limb` elements.
    // 2. `src` is valid for reads of `len` `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`, `24`, `32`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "xorl {carry:e}, {carry:e}",                 // Zero carry accumulator register
            "decq {chunks}",                             // Pre-decrement chunk counter
            "js 1f",                                     // If chunks < 0, skip to tail (1f)

            // Main 4-way unrolled loop body
            "2:",                                        // Loop head label
            // [Limb 0]
            "movq {scalar}, %rax",                       // Load scalar into %rax (implicit operand for mulq)
            "mulq 0({src})",                             // %rdx:%rax = src[0] * scalar
            "movq 0({dst}), %rcx",                       // Load dst[0] to hide multiplier latency
            "addq {carry}, %rax",                        // %rax += carry
            "adcq $0, %rdx",                             // %rdx += CF
            "addq %rax, %rcx",                           // %rcx = dst[0] + low product
            "movq %rcx, 0({dst})",                       // Store accumulated sum to dst[0]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry}",                        // carry = %rdx

            // [Limb 1]
            "movq {scalar}, %rax",                       // Reload scalar into %rax
            "mulq 8({src})",                             // %rdx:%rax = src[1] * scalar
            "movq 8({dst}), %rcx",                       // Load dst[1] to hide multiplier latency
            "addq {carry}, %rax",                        // %rax += carry
            "adcq $0, %rdx",                             // %rdx += CF
            "addq %rax, %rcx",                           // %rcx = dst[1] + low product
            "movq %rcx, 8({dst})",                       // Store to dst[1]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry}",                        // carry = %rdx

            // [Limb 2]
            "movq {scalar}, %rax",                       // Reload scalar into %rax
            "mulq 16({src})",                            // %rdx:%rax = src[2] * scalar
            "movq 16({dst}), %rcx",                      // Load dst[2] to hide multiplier latency
            "addq {carry}, %rax",                        // %rax += carry
            "adcq $0, %rdx",                             // %rdx += CF
            "addq %rax, %rcx",                           // %rcx = dst[2] + low product
            "movq %rcx, 16({dst})",                      // Store to dst[2]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry}",                        // carry = %rdx

            // [Limb 3]
            "movq {scalar}, %rax",                       // Reload scalar into %rax
            "mulq 24({src})",                            // %rdx:%rax = src[3] * scalar
            "movq 24({dst}), %rcx",                      // Load dst[3] to hide multiplier latency
            "addq {carry}, %rax",                        // %rax += carry
            "adcq $0, %rdx",                             // %rdx += CF
            "addq %rax, %rcx",                           // %rcx = dst[3] + low product
            "movq %rcx, 24({dst})",                      // Store to dst[3]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry}",                        // carry = %rdx

            "leaq 32({src}), {src}",                     // Advance src pointer by 32 bytes
            "leaq 32({dst}), {dst}",                     // Advance dst pointer by 32 bytes
            "decq {chunks}",                             // Decrement chunk counter
            "jns 2b",                                    // Repeat while chunks >= 0

            // Tail processing entry point (0 to 3 limbs remaining)
            "1:",                                        // Tail entry label
            "decq {rem}",                                // Pre-decrement remainder counter
            "js 4f",                                     // If rem < 0, skip to finish (4f)

            // 1-limb unrolled tail loop
            "3:",                                        // Tail loop label
            "movq {scalar}, %rax",                       // Load scalar into %rax
            "mulq 0({src})",                             // %rdx:%rax = src[0] * scalar
            "movq 0({dst}), %rcx",                       // Load dst[0] to hide multiplier latency
            "addq {carry}, %rax",                        // %rax += carry
            "adcq $0, %rdx",                             // %rdx += CF
            "addq %rax, %rcx",                           // %rcx = dst[0] + low product
            "movq %rcx, 0({dst})",                       // Store to dst[0]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry}",                        // carry = %rdx
            "leaq 8({src}), {src}",                      // Advance src pointer (+8)
            "leaq 8({dst}), {dst}",                      // Advance dst pointer (+8)
            "decq {rem}",                                // Decrement remainder counter
            "jns 3b",                                    // Repeat while rem >= 0

            // Tail completion
            "4:",                                        // Completion label

            carry = out(reg) carry,
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
    carry
}
