//! Baseline x86-64 fused dual-row multiply-add kernel.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 64)`)
//! using standard `mulq` ($64 \times 64 \to 128$-bit into `%rdx:%rax`) and `addq`/`adcq`.

use core::arch::asm;

use super::Limb;

/// Accumulates two interleaved scalar-product rows and returns their separate carries.
///
/// Four source limbs are processed per block, followed by a scalar tail.
/// Empty spans return `(0, 0)` without accessing pointers.
///
/// # Safety
///
/// For nonzero `len`, `src` must cover `len` aligned, initialized limbs and
/// `dst` must cover `len + 1` aligned, initialized, writable limbs. The spans
/// must be disjoint and remain within live allocations of at most `isize::MAX` bytes.
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
    let mut c0: Limb = 0;
    let mut c1: Limb = 0;
    if len == 0 {
        return (0, 0);
    }
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: 4*chunks + rem == len bounds the aligned source[j] and
    // destination[j..=j+1] accesses. Counts fit signed registers because spans
    // have at most isize::MAX bytes. All writes stay in the initialized,
    // writable destination, which is disjoint from the source.
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
