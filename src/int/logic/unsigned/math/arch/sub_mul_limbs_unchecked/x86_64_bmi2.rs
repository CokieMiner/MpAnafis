//! BMI2-only `mulx` fused multiply-subtract limb kernel for `x86_64` (without ADX).
//!
//! Uses `mulxq` (BMI2) for flag-free multiplication and explicit register tracking
//! for concurrent multiplication carry and subtraction borrow chains.

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
/// On processors with BMI2 but lacking ADX, `mulxq` computes $64 \times 64 \to 128$-bit unsigned
/// products into GPRs without touching flags. The multiplication carry (`carry_hi`) and subtraction
/// borrow (`borrow`) are tracked and accumulated in 64-bit registers, and propagated across
/// 4-way unrolled iterations.
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
            "1:",                                        // Entry label
            "decq {chunks}",                             // Pre-decrement chunk counter
            "js 3f",                                     // If chunks < 0, skip to remainder (3f)

            // Main 4-way unrolled loop body
            "2:",                                        // Loop head label
            // [Limb 0]
            "mulxq 0({src}), %rax, %r8",                  // %rdx * src[0] -> (%r8:%rax)
            "mulxq 8({src}), %r10, %r11",                 // %rdx * src[1] -> (%r11:%r10)
            "addq {carry_hi}, %rax",                      // %rax += carry_hi
            "adcq $0, %r8",                               // %r8 += CF
            "addq {borrow}, %rax",                        // %rax += borrow
            "movq $0, {borrow}",                          // Clear borrow
            "adcq $0, {borrow}",                          // Capture add overflow into borrow
            "movq 0({dst}), %rcx",                        // Load dst[0]
            "subq %rax, %rcx",                            // %rcx = dst[0] - %rax
            "movq %rcx, 0({dst})",                        // Store result back to dst[0]
            "adcq $0, {borrow}",                          // Capture subtraction borrow

            // [Limb 1]
            "addq %r8, %r10",                             // %r10 += %r8 (high product of limb 0)
            "adcq $0, %r11",                              // %r11 += CF
            "movq %r11, {carry_hi}",                      // Update running carry_hi
            "addq {borrow}, %r10",                        // %r10 += borrow
            "movq $0, {borrow}",                          // Clear borrow
            "adcq $0, {borrow}",                          // Capture add overflow
            "movq 8({dst}), %rcx",                        // Load dst[1]
            "subq %r10, %rcx",                            // %rcx = dst[1] - %r10
            "movq %rcx, 8({dst})",                        // Store result back to dst[1]
            "adcq $0, {borrow}",                          // Capture subtraction borrow

            // [Limb 2]
            "mulxq 16({src}), %rax, %r8",                 // %rdx * src[2] -> (%r8:%rax)
            "mulxq 24({src}), %r10, %r11",                // %rdx * src[3] -> (%r11:%r10)
            "addq {carry_hi}, %rax",                      // %rax += carry_hi
            "adcq $0, %r8",                               // %r8 += CF
            "addq {borrow}, %rax",                        // %rax += borrow
            "movq $0, {borrow}",                          // Clear borrow
            "adcq $0, {borrow}",                          // Capture add overflow
            "movq 16({dst}), %rcx",                       // Load dst[2]
            "subq %rax, %rcx",                            // %rcx = dst[2] - %rax
            "movq %rcx, 16({dst})",                       // Store result back to dst[2]
            "adcq $0, {borrow}",                          // Capture subtraction borrow

            // [Limb 3]
            "addq %r8, %r10",                             // %r10 += %r8
            "adcq $0, %r11",                              // %r11 += CF
            "movq %r11, {carry_hi}",                      // Update running carry_hi
            "addq {borrow}, %r10",                        // %r10 += borrow
            "movq $0, {borrow}",                          // Clear borrow
            "adcq $0, {borrow}",                          // Capture add overflow
            "movq 24({dst}), %rcx",                       // Load dst[3]
            "subq %r10, %rcx",                            // %rcx = dst[3] - %r10
            "movq %rcx, 24({dst})",                       // Store result back to dst[3]
            "adcq $0, {borrow}",                          // Capture subtraction borrow

            "leaq 32({src}), {src}",                     // Advance src pointer by 32 bytes
            "leaq 32({dst}), {dst}",                     // Advance dst pointer by 32 bytes
            "decq {chunks}",                             // Decrement chunks
            "jns 2b",                                    // Repeat while chunks >= 0

            // Remainder processing entry point (0 to 3 limbs)
            "3:",                                        // Remainder entry label
            "decq {rem}",                                // Pre-decrement remainder counter
            "js 5f",                                     // If rem < 0, skip to finish (5f)

            // 1-limb unrolled tail loop
            "4:",                                        // Tail loop label
            "mulxq 0({src}), %rax, %r8",                  // Multiply single limb
            "addq {carry_hi}, %rax",                      // %rax += carry_hi
            "adcq $0, %r8",                               // %r8 += CF
            "movq %r8, {carry_hi}",                       // Update carry_hi
            "addq {borrow}, %rax",                        // %rax += borrow
            "movq $0, {borrow}",                          // Clear borrow
            "adcq $0, {borrow}",                          // Capture add overflow
            "movq 0({dst}), %rcx",                        // Load dst[0]
            "subq %rax, %rcx",                            // dst[0] - %rax
            "movq %rcx, 0({dst})",                        // Store updated limb
            "adcq $0, {borrow}",                          // Capture subtraction borrow
            "leaq 8({src}), {src}",                      // Advance src pointer (+8)
            "leaq 8({dst}), {dst}",                      // Advance dst pointer (+8)
            "decq {rem}",                                // Decrement remainder counter
            "jns 4b",                                    // Repeat while rem >= 0

            // Tail completion
            "5:",                                        // Completion label

            carry_hi = out(reg) carry_hi,
            borrow = out(reg) borrow,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            in("rdx") scalar,
            out("rax") _,
            out("rcx") _,
            out("r8") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
    (carry_hi, borrow)
}
