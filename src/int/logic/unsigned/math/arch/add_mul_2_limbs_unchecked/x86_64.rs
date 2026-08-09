//! Baseline x86-64 fused dual-row multiply-add kernel.

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
/// This is the vanilla `x86_64` implementation utilizing the standard `mulq`
/// instruction for 64x64->128-bit multiplication and `addq`/`adcq` for
/// multi-precision accumulation.
///
/// Returns `(c0, c1)` — the carry-out words for row 0 and row 1.
///
/// # Safety
///
/// `dst` must be valid for `len + 1` elements (row 1 writes one limb ahead
/// of row 0); `src` must be valid for `len` elements. `len` may be 0.
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
    // SAFETY: caller guarantees dst valid for len+1 and src valid for len elements.
    unsafe {
        asm!(
            "decq {chunks}",
            "js 3f",
            ".p2align 4",
            "1:",
            // Limb 0 - Row 0
            "movq {s0}, %rax",
            "mulq ({src})",
            "addq {c0}, %rax",
            "adcq $0, %rdx",
            "addq %rax, ({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {c0}",

            // Limb 0 - Row 1
            "movq {s1}, %rax",
            "mulq ({src})",
            "addq {c1}, %rax",
            "adcq $0, %rdx",
            "addq %rax, 8({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {c1}",

            // Limb 1 - Row 0
            "movq {s0}, %rax",
            "mulq 8({src})",
            "addq {c0}, %rax",
            "adcq $0, %rdx",
            "addq %rax, 8({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {c0}",

            // Limb 1 - Row 1
            "movq {s1}, %rax",
            "mulq 8({src})",
            "addq {c1}, %rax",
            "adcq $0, %rdx",
            "addq %rax, 16({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {c1}",

            // Limb 2 - Row 0
            "movq {s0}, %rax",
            "mulq 16({src})",
            "addq {c0}, %rax",
            "adcq $0, %rdx",
            "addq %rax, 16({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {c0}",

            // Limb 2 - Row 1
            "movq {s1}, %rax",
            "mulq 16({src})",
            "addq {c1}, %rax",
            "adcq $0, %rdx",
            "addq %rax, 24({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {c1}",

            // Limb 3 - Row 0
            "movq {s0}, %rax",
            "mulq 24({src})",
            "addq {c0}, %rax",
            "adcq $0, %rdx",
            "addq %rax, 24({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {c0}",

            // Limb 3 - Row 1
            "movq {s1}, %rax",
            "mulq 24({src})",
            "addq {c1}, %rax",
            "adcq $0, %rdx",
            "addq %rax, 32({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {c1}",

            "leaq 32({src}), {src}",
            "leaq 32({dst}), {dst}",
            "decq {chunks}",
            "jns 1b",
            "3:",
            "decq {rem}",
            "js 2f",
            ".p2align 4",
            "4:",
            // Remainder - Row 0
            "movq {s0}, %rax",
            "mulq ({src})",
            "addq {c0}, %rax",
            "adcq $0, %rdx",
            "addq %rax, ({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {c0}",

            // Remainder - Row 1
            "movq {s1}, %rax",
            "mulq ({src})",
            "addq {c1}, %rax",
            "adcq $0, %rdx",
            "addq %rax, 8({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {c1}",

            "leaq 8({src}), {src}",
            "leaq 8({dst}), {dst}",
            "decq {rem}",
            "jns 4b",

            "2:",

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            out("rax") _,
            out("rdx") _,
            options(nostack, att_syntax)
        );
    }
    (c0, c1)
}
