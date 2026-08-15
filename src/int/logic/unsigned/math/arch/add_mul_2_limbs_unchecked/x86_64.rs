//! Baseline x86-64 fused dual-row multiply-add kernel.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 64)`)
//! using standard `mulq` ($64 \times 64 \to 128$-bit into `%rdx:%rax`) and `addq`/`adcq`.

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
/// Multi-precision basecase multiplication processes two multiplier scalars (`s0` and `s1`)
/// simultaneously. The 4-way unrolled body keeps both carry chains (`c0` and `c1`) in registers,
/// using standard `mulq` and paired `addq`/`adcq` additions to assemble both rows in a single pass.
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
            "js 3f",                                     // If chunks < 0 (i.e. len < 4), skip to remainder (3f)

            // Main 4-way unrolled loop body
            "1:",                                        // Loop head label
            // [Limb 0 - Row 0]
            "movq ({src}), %r8",                         // Load src[0] into %r8
            "movq {s0}, %rax",                           // %rax = s0
            "mulq %r8",                                  // %rdx:%rax = src[0] * s0 (128-bit product)
            "addq {c0}, %rax",                           // %rax += c0
            "adcq $0, %rdx",                             // %rdx += CF
            "addq ({dst}), %rax",                        // %rax += dst[0]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rax, ({dst})",                        // Store finalized dst[0]
            "movq %rdx, {c0}",                           // Update row 0 carry

            // [Limb 0 - Row 1]
            "movq {s1}, %rax",                           // %rax = s1
            "mulq %r8",                                  // %rdx:%rax = src[0] * s1
            "addq {c1}, %rax",                           // %rax += c1
            "adcq $0, %rdx",                             // %rdx += CF
            "addq 8({dst}), %rax",                       // %rax += dst[1]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rax, 8({dst})",                       // Store intermediate dst[1]
            "movq %rdx, {c1}",                           // Update row 1 carry

            // [Limb 1 - Row 0]
            "movq 8({src}), %r8",                        // Load src[1]
            "movq {s0}, %rax",                           // %rax = s0
            "mulq %r8",                                  // src[1] * s0
            "addq {c0}, %rax",                           // %rax += c0
            "adcq $0, %rdx",                             // %rdx += CF
            "addq 8({dst}), %rax",                       // %rax += dst[1]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rax, 8({dst})",                       // Store finalized dst[1]
            "movq %rdx, {c0}",                           // Update row 0 carry

            // [Limb 1 - Row 1]
            "movq {s1}, %rax",                           // %rax = s1
            "mulq %r8",                                  // src[1] * s1
            "addq {c1}, %rax",                           // %rax += c1
            "adcq $0, %rdx",                             // %rdx += CF
            "addq 16({dst}), %rax",                      // %rax += dst[2]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rax, 16({dst})",                      // Store intermediate dst[2]
            "movq %rdx, {c1}",                           // Update row 1 carry

            // [Limb 2 - Row 0]
            "movq 16({src}), %r8",                       // Load src[2]
            "movq {s0}, %rax",                           // %rax = s0
            "mulq %r8",                                  // src[2] * s0
            "addq {c0}, %rax",                           // %rax += c0
            "adcq $0, %rdx",                             // %rdx += CF
            "addq 16({dst}), %rax",                      // %rax += dst[2]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rax, 16({dst})",                      // Store finalized dst[2]
            "movq %rdx, {c0}",                           // Update row 0 carry

            // [Limb 2 - Row 1]
            "movq {s1}, %rax",                           // %rax = s1
            "mulq %r8",                                  // src[2] * s1
            "addq {c1}, %rax",                           // %rax += c1
            "adcq $0, %rdx",                             // %rdx += CF
            "addq 24({dst}), %rax",                      // %rax += dst[3]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rax, 24({dst})",                      // Store intermediate dst[3]
            "movq %rdx, {c1}",                           // Update row 1 carry

            // [Limb 3 - Row 0]
            "movq 24({src}), %r8",                       // Load src[3]
            "movq {s0}, %rax",                           // %rax = s0
            "mulq %r8",                                  // src[3] * s0
            "addq {c0}, %rax",                           // %rax += c0
            "adcq $0, %rdx",                             // %rdx += CF
            "addq 24({dst}), %rax",                      // %rax += dst[3]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rax, 24({dst})",                      // Store finalized dst[3]
            "movq %rdx, {c0}",                           // Update row 0 carry

            // [Limb 3 - Row 1]
            "movq {s1}, %rax",                           // %rax = s1
            "mulq %r8",                                  // src[3] * s1
            "addq {c1}, %rax",                           // %rax += c1
            "adcq $0, %rdx",                             // %rdx += CF
            "addq 32({dst}), %rax",                      // %rax += dst[4]
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rax, 32({dst})",                      // Store intermediate dst[4]
            "movq %rdx, {c1}",                           // Update row 1 carry

            "addq $32, {src}",                           // Advance src pointer by 32 bytes
            "addq $32, {dst}",                           // Advance dst pointer by 32 bytes
            "decq {chunks}",                             // Decrement chunk counter
            "jns 1b",                                    // Repeat while chunks >= 0

            // Remainder entry point (0 to 3 limbs)
            "3:",                                        // Remainder entry label
            "testq {rem}, {rem}",                        // Test if remainder count == 0
            "jz 5f",                                     // If zero, jump to completion (5f)

            // 1-limb tail loop
            "4:",                                        // Tail loop label
            "movq ({src}), %r8",                         // Load single src limb
            "movq {s0}, %rax",                           // Load s0 into rax
            "mulq %r8",                                  // src[j] * s0
            "addq {c0}, %rax",                           // rax += c0
            "adcq $0, %rdx",                             // rdx += CF
            "addq ({dst}), %rax",                        // rax += dst[j]
            "adcq $0, %rdx",                             // rdx += CF
            "movq %rax, ({dst})",                        // Finalize dst[j]
            "movq %rdx, {c0}",                           // Update c0
            "movq {s1}, %rax",                           // Load s1 into rax
            "mulq %r8",                                  // src[j] * s1
            "addq {c1}, %rax",                           // rax += c1
            "adcq $0, %rdx",                             // rdx += CF
            "addq 8({dst}), %rax",                       // rax += dst[j+1]
            "adcq $0, %rdx",                             // rdx += CF
            "movq %rax, 8({dst})",                       // Store intermediate dst[j+1]
            "movq %rdx, {c1}",                           // Update c1
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
            out("rax") _,
            out("rdx") _,
            out("r8") _,
            options(nostack, att_syntax)
        );
    }
    (c0, c1)
}
