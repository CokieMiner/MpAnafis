//! BMI2-only `mulx` fused multiply-add limb kernel for `x86_64` (without ADX).
//!
//! Uses `mulxq` (BMI2) for flag-free multiplication and standard `addq`/`adcq`
//! for single-chain carry propagation. Dispatched at runtime on Haswell and early
//! AMD Zen architectures supporting BMI2 without ADX.

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
/// On processors with BMI2 but lacking ADX (e.g. Intel Haswell / Broadwell without ADX),
/// `mulxq` computes $64 \times 64 \to 128$-bit unsigned products into arbitrary destination
/// registers without altering flags, breaking the architectural bottleneck of legacy `mulq`.
/// Carry propagation is performed via an unrolled single-chain sequence using `addq`/`adcq`.
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
            "xorl {carry:e}, {carry:e}",                 // Zero carry register
            "decq {chunks}",                             // Pre-decrement chunk counter
            "js 1f",                                     // If chunks < 0, skip to tail (1f)

            // Main 4-way unrolled loop body
            "2:",                                        // Loop head label
            "mulxq 0({src}), %rax, %r8",                 // (%r8:%rax) = scalar * src[0]
            "mulxq 8({src}), %r10, %r11",                // (%r11:%r10) = scalar * src[1]
            "addq {carry}, %rax",                        // rax += carry, set CF
            "adcq $0, %r8",                              // r8 += CF
            "movq 0({dst}), %rcx",                       // rcx = dst[0]
            "addq %rax, %rcx",                           // rcx += rax, set CF
            "movq %rcx, 0({dst})",                       // Store updated dst[0]
            "adcq $0, %r8",                              // r8 += CF
            "addq %r8, %r10",                            // r10 += r8 (hi0), set CF
            "adcq $0, %r11",                             // r11 += CF
            "movq 8({dst}), %rcx",                       // rcx = dst[1]
            "addq %r10, %rcx",                           // rcx += r10, set CF
            "movq %rcx, 8({dst})",                       // Store updated dst[1]
            "adcq $0, %r11",                             // r11 += CF
            "movq %r11, {carry}",                        // Save running carry

            "mulxq 16({src}), %rax, %r8",                // (%r8:%rax) = scalar * src[2]
            "mulxq 24({src}), %r10, %r11",               // (%r11:%r10) = scalar * src[3]
            "addq {carry}, %rax",                        // rax += carry, set CF
            "adcq $0, %r8",                              // r8 += CF
            "movq 16({dst}), %rcx",                      // rcx = dst[2]
            "addq %rax, %rcx",                           // rcx += rax, set CF
            "movq %rcx, 16({dst})",                      // Store updated dst[2]
            "adcq $0, %r8",                              // r8 += CF
            "addq %r8, %r10",                            // r10 += r8 (hi2), set CF
            "adcq $0, %r11",                             // r11 += CF
            "movq 24({dst}), %rcx",                      // rcx = dst[3]
            "addq %r10, %rcx",                           // rcx += r10, set CF
            "movq %rcx, 24({dst})",                      // Store updated dst[3]
            "adcq $0, %r11",                             // r11 += CF
            "movq %r11, {carry}",                        // Save running carry

            "leaq 32({src}), {src}",                     // Advance src pointer by 32 bytes
            "leaq 32({dst}), {dst}",                     // Advance dst pointer by 32 bytes
            "decq {chunks}",                             // Decrement chunks
            "jns 2b",                                    // Repeat while chunks >= 0

            // Tail processing entry point (0 to 3 limbs remaining)
            "1:",                                        // Tail entry label
            "decq {rem}",                                // Pre-decrement remainder counter
            "js 4f",                                     // If rem < 0, skip to finish (4f)

            // 1-limb unrolled tail loop
            "3:",                                        // Tail loop label
            "mulxq 0({src}), %rax, %r8",                 // (%r8:%rax) = scalar * src[0]
            "addq {carry}, %rax",                        // rax += carry, set CF
            "adcq $0, %r8",                              // r8 += CF
            "movq 0({dst}), %rcx",                       // rcx = dst[0]
            "addq %rax, %rcx",                           // rcx += rax, set CF
            "movq %rcx, 0({dst})",                       // Store updated dst[0]
            "adcq $0, %r8",                              // r8 += CF
            "movq %r8, {carry}",                         // Update running carry
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
            in("rdx") scalar,
            out("rax") _,
            out("rcx") _,
            out("r8") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
    carry
}
