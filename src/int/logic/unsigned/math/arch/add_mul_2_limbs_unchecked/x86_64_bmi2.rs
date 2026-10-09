//! BMI2 fused dual-row multiply-add kernel for x86-64.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 64)`)
//! using flag-free `mulxq` to share source loads and retain overlapping destination limbs.

use core::arch::asm;

use super::Limb;

/// Accumulates two interleaved scalar-product rows and returns their separate carries.
///
/// Each `src[j]` is loaded into `%rdx` once for both flag-preserving products. The row carries
/// occupy `%r9` and `%r11`; each is consumed before its next product overwrites it.
/// `%rcx` retains the intermediate `dst[j + 1]` from row 1 for the next row-0
/// update. Each destination limb is loaded and stored once, including a final
/// store of the retained limb to `dst[len]`.
/// Empty spans return `(0, 0)` without accessing pointers.
///
/// # Safety
///
/// For nonzero `len`, `src` must cover `len` aligned, initialized limbs and
/// `dst` must cover `len + 1` aligned, initialized, writable limbs. The spans
/// must be disjoint and remain within live allocations of at most `isize::MAX` bytes.
/// The executing CPU must support BMI2.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "Keep both unrolled row recurrences in the selected multiplication caller"
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
    let chunks = len >> 2;
    let rem = len & 3;
    let c0: Limb;
    let c1: Limb;

    // SAFETY: the caller supplies len+1 initialized destination limbs, len
    // initialized source limbs, disjoint spans, and BMI2 availability. Since
    // len > 0, the initial destination load is valid. Each iteration consumes
    // one source limb and finalizes one destination limb; RCX retains row 1's
    // intermediate limb for the next row-0 update. The final store writes
    // dst[len]. All main and tail offsets lie within those supplied spans.
    // Each incoming row carry is consumed before MULX replaces its high word.
    // For B = 2^64 and carry <= scalar, each row sum is bounded by
    // (B-1)*scalar + (B-1) + scalar <= B^2-1, so both ADC steps fit its high
    // word. RAX and RDX receive the two carries after all inputs are consumed.
    unsafe {
        asm!(
            "xorl %r9d, %r9d",                           // Initial row-0 carry
            "xorl %r11d, %r11d",                         // Initial row-1 carry
            "movq ({dst}), %rcx",                        // Initial overlapping destination limb
            "decq {chunks}",                             // Pre-decrement chunk counter for sign-flag check
            "js 3f",                                     // If chunks < 0 (len < 4), skip to remainder (3f)

            // Main 4-way unrolled loop body
            "1:",                                        // Loop head label
            // Limb 0: finalize row 0, retain row 1 for the next column
            "addq %r9, %rcx",                            // Consume incoming row-0 carry
            "movq ({src}), %rdx",                        // %rdx = src[0] (shared multiplier operand)
            "mulxq {s0}, %rax, %r9",                     // %r9:%rax = src[0] * s0; preserves incoming CF
            "adcq $0, %r9",
            "addq %rax, %rcx",
            "adcq $0, %r9",
            "movq %rcx, ({dst})",                        // Finalized dst[0]
            "movq 8({dst}), %rcx",                       // Original dst[1]
            "addq %r11, %rcx",                           // Consume incoming row-1 carry
            "mulxq {s1}, %rax, %r11",
            "adcq $0, %r11",
            "addq %rax, %rcx",
            "adcq $0, %r11",                             // Retain updated dst[1] in RCX

            // Limb 1
            "addq %r9, %rcx",
            "movq 8({src}), %rdx",                       // %rdx = src[1]
            "mulxq {s0}, %rax, %r9",
            "adcq $0, %r9",
            "addq %rax, %rcx",
            "adcq $0, %r9",
            "movq %rcx, 8({dst})",
            "movq 16({dst}), %rcx",
            "addq %r11, %rcx",
            "mulxq {s1}, %rax, %r11",
            "adcq $0, %r11",
            "addq %rax, %rcx",
            "adcq $0, %r11",

            // Limb 2
            "addq %r9, %rcx",
            "movq 16({src}), %rdx",                      // %rdx = src[2]
            "mulxq {s0}, %rax, %r9",
            "adcq $0, %r9",
            "addq %rax, %rcx",
            "adcq $0, %r9",
            "movq %rcx, 16({dst})",
            "movq 24({dst}), %rcx",
            "addq %r11, %rcx",
            "mulxq {s1}, %rax, %r11",
            "adcq $0, %r11",
            "addq %rax, %rcx",
            "adcq $0, %r11",

            // Limb 3
            "addq %r9, %rcx",
            "movq 24({src}), %rdx",                      // %rdx = src[3]
            "mulxq {s0}, %rax, %r9",
            "adcq $0, %r9",
            "addq %rax, %rcx",
            "adcq $0, %r9",
            "movq %rcx, 24({dst})",
            "movq 32({dst}), %rcx",
            "addq %r11, %rcx",
            "mulxq {s1}, %rax, %r11",
            "adcq $0, %r11",
            "addq %rax, %rcx",
            "adcq $0, %r11",

            "leaq 32({src}), {src}",                     // Advance src pointer by 32 bytes
            "leaq 32({dst}), {dst}",                     // Advance dst pointer by 32 bytes
            "decq {chunks}",                             // Decrement chunk counter
            "jns 1b",                                    // Repeat while chunks >= 0

            // Remainder entry point (0 to 3 limbs)
            "3:",                                        // Remainder entry label
            "testq {rem}, {rem}",                        // Test if remainder count == 0
            "jz 5f",                                     // If zero, skip to completion (5f)

            // 1-limb tail loop
            "4:",                                        // Tail loop label
            "addq %r9, %rcx",
            "movq ({src}), %rdx",                        // Load single src limb into %rdx
            "mulxq {s0}, %rax, %r9",
            "adcq $0, %r9",
            "addq %rax, %rcx",
            "adcq $0, %r9",
            "movq %rcx, ({dst})",
            "movq 8({dst}), %rcx",
            "addq %r11, %rcx",
            "mulxq {s1}, %rax, %r11",
            "adcq $0, %r11",
            "addq %rax, %rcx",
            "adcq $0, %r11",
            "leaq 8({src}), {src}",
            "leaq 8({dst}), {dst}",
            "decq {rem}",                                // Decrement remainder counter
            "jnz 4b",                                    // Repeat while rem != 0

            // Tail completion
            "5:",                                        // Completion label
            "movq %rcx, ({dst})",                        // Retained final limb at dst[len]
            "movq %r9, %rax",                            // Row-0 carry
            "movq %r11, %rdx",                           // Row-1 carry

            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            out("rax") c0,
            out("rdx") c1,
            out("rcx") _,
            out("r9") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
    (c0, c1)
}
