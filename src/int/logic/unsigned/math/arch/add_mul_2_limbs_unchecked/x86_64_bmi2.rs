//! BMI2 fused dual-row multiply-add kernel for x86-64.

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
/// This implementation utilizes the `x86_64` BMI2 `mulxq` instruction for
/// flag-free multiplication, allowing back-to-back dual scalar multiplication
/// sharing the same source load in `%rdx`.
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
            // Limb 0
            "movq ({src}), %rdx",
            "mulxq {s0}, %r8, %r9",
            "mulxq {s1}, %r10, %r11",
            "addq {c0}, %r8",
            "adcq $0, %r9",
            "addq %r8, ({dst})",
            "adcq $0, %r9",
            "movq %r9, {c0}",
            "addq {c1}, %r10",
            "adcq $0, %r11",
            "addq %r10, 8({dst})",
            "adcq $0, %r11",
            "movq %r11, {c1}",

            // Limb 1
            "movq 8({src}), %rdx",
            "mulxq {s0}, %r8, %r9",
            "mulxq {s1}, %r10, %r11",
            "addq {c0}, %r8",
            "adcq $0, %r9",
            "addq %r8, 8({dst})",
            "adcq $0, %r9",
            "movq %r9, {c0}",
            "addq {c1}, %r10",
            "adcq $0, %r11",
            "addq %r10, 16({dst})",
            "adcq $0, %r11",
            "movq %r11, {c1}",

            // Limb 2
            "movq 16({src}), %rdx",
            "mulxq {s0}, %r8, %r9",
            "mulxq {s1}, %r10, %r11",
            "addq {c0}, %r8",
            "adcq $0, %r9",
            "addq %r8, 16({dst})",
            "adcq $0, %r9",
            "movq %r9, {c0}",
            "addq {c1}, %r10",
            "adcq $0, %r11",
            "addq %r10, 24({dst})",
            "adcq $0, %r11",
            "movq %r11, {c1}",

            // Limb 3
            "movq 24({src}), %rdx",
            "mulxq {s0}, %r8, %r9",
            "mulxq {s1}, %r10, %r11",
            "addq {c0}, %r8",
            "adcq $0, %r9",
            "addq %r8, 24({dst})",
            "adcq $0, %r9",
            "movq %r9, {c0}",
            "addq {c1}, %r10",
            "adcq $0, %r11",
            "addq %r10, 32({dst})",
            "adcq $0, %r11",
            "movq %r11, {c1}",

            "leaq 32({src}), {src}",
            "leaq 32({dst}), {dst}",
            "decq {chunks}",
            "jns 1b",
            "3:",
            "decq {rem}",
            "js 2f",
            ".p2align 4",
            "4:",
            "movq ({src}), %rdx",
            "mulxq {s0}, %r8, %r9",
            "mulxq {s1}, %r10, %r11",
            "addq {c0}, %r8",
            "adcq $0, %r9",
            "addq %r8, ({dst})",
            "adcq $0, %r9",
            "movq %r9, {c0}",
            "addq {c1}, %r10",
            "adcq $0, %r11",
            "addq %r10, 8({dst})",
            "adcq $0, %r11",
            "movq %r11, {c1}",

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
