//! BMI2 fused dual-row multiply-add kernel for x86-64.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 64)`)
//! using flag-free `mulxq` to share source operand loads across both multiplier pipelines.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by two scalars `s0` and `s1` simultaneously,
/// accumulating each result into two overlapping rows of `dst`:
///
/// ```text
///   (c0, dst[0..len])   = dst[0..len]   + src[0..len] × s0   [row i]
///   (c1, dst[1..len+1]) = dst[1..len+1] + src[0..len] × s1   [row i+1]
/// ```
///
/// Returns `(c0, c1)` — the carry-out words for row 0 and row 1.
///
/// # Microarchitectural Strategy
///
/// `mulxq` computes $64 \times 64 \to 128$-bit multiplication without modifying CPU flags.
/// By loading `src[j]` into `%rdx` once, both products (`src[j] * s0` into `%r9:%r8` and
/// `src[j] * s1` into `%r11:%r10`) execute back-to-back without reloading memory or clobbering flags.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len + 1` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "Critical for peak assembly performance: dual-row schoolbook inner loop"
)]
#[inline(always)]
pub unsafe fn add_mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) -> (Limb, Limb) {
    let mut c0: Limb = 0;
    let mut c1: Limb = 0;
    if len == 0 {
        return (0, 0);
    }
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` is valid for reads and writes of `len + 1` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`, `24`, `32`) remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "decq {chunks}",                             // Pre-decrement chunk counter for sign-flag check
            "js 3f",                                     // If chunks < 0 (len < 4), skip to remainder (3f)

            // Main 4-way unrolled loop body
            "1:",                                        // Loop head label
            // [Limb 0: Shared Source Load in %rdx]
            "movq ({src}), %rdx",                        // %rdx = src[0] (shared multiplier operand)
            "mulxq {s0}, %r8, %r9",                      // %r9:%r8 = src[0] * s0 (flag-free 128-bit product)
            "mulxq {s1}, %r10, %r11",                    // %r11:%r10 = src[0] * s1 (flag-free product)
            "addq {c0}, %r8",                            // %r8 += c0
            "adcq $0, %r9",                              // %r9 += CF
            "addq ({dst}), %r8",                         // %r8 += dst[0]
            "adcq $0, %r9",                              // %r9 += CF
            "movq %r8, ({dst})",                         // Store finalized dst[0]
            "movq %r9, {c0}",                            // Update row 0 carry
            "addq {c1}, %r10",                           // %r10 += c1
            "adcq $0, %r11",                             // %r11 += CF
            "addq 8({dst}), %r10",                       // %r10 += dst[1]
            "adcq $0, %r11",                             // %r11 += CF
            "movq %r10, 8({dst})",                       // Store intermediate dst[1]
            "movq %r11, {c1}",                           // Update row 1 carry

            // [Limb 1: Shared Source Load in %rdx]
            "movq 8({src}), %rdx",                       // %rdx = src[1]
            "mulxq {s0}, %r8, %r9",                      // src[1] * s0
            "mulxq {s1}, %r10, %r11",                    // src[1] * s1
            "addq {c0}, %r8",                            // %r8 += c0
            "adcq $0, %r9",                              // %r9 += CF
            "addq 8({dst}), %r8",                        // %r8 += dst[1]
            "adcq $0, %r9",                              // %r9 += CF
            "movq %r8, 8({dst})",                        // Store finalized dst[1]
            "movq %r9, {c0}",                            // Update row 0 carry
            "addq {c1}, %r10",                           // %r10 += c1
            "adcq $0, %r11",                             // %r11 += CF
            "addq 16({dst}), %r10",                      // %r10 += dst[2]
            "adcq $0, %r11",                             // %r11 += CF
            "movq %r10, 16({dst})",                      // Store intermediate dst[2]
            "movq %r11, {c1}",                           // Update row 1 carry

            // [Limb 2: Shared Source Load in %rdx]
            "movq 16({src}), %rdx",                      // %rdx = src[2]
            "mulxq {s0}, %r8, %r9",                      // src[2] * s0
            "mulxq {s1}, %r10, %r11",                    // src[2] * s1
            "addq {c0}, %r8",                            // %r8 += c0
            "adcq $0, %r9",                              // %r9 += CF
            "addq 16({dst}), %r8",                       // %r8 += dst[2]
            "adcq $0, %r9",                              // %r9 += CF
            "movq %r8, 16({dst})",                       // Store finalized dst[2]
            "movq %r9, {c0}",                            // Update row 0 carry
            "addq {c1}, %r10",                           // %r10 += c1
            "adcq $0, %r11",                             // %r11 += CF
            "addq 24({dst}), %r10",                      // %r10 += dst[3]
            "adcq $0, %r11",                             // %r11 += CF
            "movq %r10, 24({dst})",                      // Store intermediate dst[3]
            "movq %r11, {c1}",                           // Update row 1 carry

            // [Limb 3: Shared Source Load in %rdx]
            "movq 24({src}), %rdx",                      // %rdx = src[3]
            "mulxq {s0}, %r8, %r9",                      // src[3] * s0
            "mulxq {s1}, %r10, %r11",                    // src[3] * s1
            "addq {c0}, %r8",                            // %r8 += c0
            "adcq $0, %r9",                              // %r9 += CF
            "addq 24({dst}), %r8",                       // %r8 += dst[3]
            "adcq $0, %r9",                              // %r9 += CF
            "movq %r8, 24({dst})",                       // Store finalized dst[3]
            "movq %r9, {c0}",                            // Update row 0 carry
            "addq {c1}, %r10",                           // %r10 += c1
            "adcq $0, %r11",                             // %r11 += CF
            "addq 32({dst}), %r10",                      // %r10 += dst[4]
            "adcq $0, %r11",                             // %r11 += CF
            "movq %r10, 32({dst})",                      // Store intermediate dst[4]
            "movq %r11, {c1}",                           // Update row 1 carry

            "addq $32, {src}",                           // Advance src pointer by 32 bytes
            "addq $32, {dst}",                           // Advance dst pointer by 32 bytes
            "decq {chunks}",                             // Decrement chunk counter
            "jns 1b",                                    // Repeat while chunks >= 0

            // Remainder entry point (0 to 3 limbs)
            "3:",                                        // Remainder entry label
            "testq {rem}, {rem}",                        // Test if remainder count == 0
            "jz 5f",                                     // If zero, skip to completion (5f)

            // 1-limb tail loop
            "4:",                                        // Tail loop label
            "movq ({src}), %rdx",                        // Load single src limb into %rdx
            "mulxq {s0}, %r8, %r9",                      // src[j] * s0
            "mulxq {s1}, %r10, %r11",                    // src[j] * s1
            "addq {c0}, %r8",                            // r8 += c0
            "adcq $0, %r9",                              // r9 += CF
            "addq ({dst}), %r8",                         // r8 += dst[j]
            "adcq $0, %r9",                              // r9 += CF
            "movq %r8, ({dst})",                         // Finalize dst[j]
            "movq %r9, {c0}",                            // Update c0
            "addq {c1}, %r10",                           // r10 += c1
            "adcq $0, %r11",                             // r11 += CF
            "addq 8({dst}), %r10",                       // r10 += dst[j+1]
            "adcq $0, %r11",                             // r11 += CF
            "movq %r10, 8({dst})",                       // Intermediate dst[j+1]
            "movq %r11, {c1}",                           // Update c1
            "addq $8, {src}",                            // Advance src pointer (+8)
            "addq $8, {dst}",                            // Advance dst pointer (+8)
            "decq {rem}",                                // Decrement remainder counter
            "jnz 4b",                                    // Repeat while rem != 0

            // Tail completion
            "5:",                                        // Completion label

            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            out("rdx") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
    (c0, c1)
}
