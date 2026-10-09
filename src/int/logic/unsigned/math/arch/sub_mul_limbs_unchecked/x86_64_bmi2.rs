//! BMI2-only `mulx` fused multiply-subtract limb kernel for `x86_64` (without ADX).
//!
//! Uses `mulxq` (BMI2) for flag-free multiplication and explicit register tracking
//! for concurrent multiplication carry and subtraction borrow chains.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// For B = 2^64, `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
/// Four-limb blocks retain product carry and subtraction borrow separately.
///
/// # Safety
///
/// Both pointers must cover `len` aligned, initialized limbs in disjoint spans;
/// `dst` requires exclusive access. Each span's byte length must fit in `isize`.
/// Zero length permits null pointers. The CPU must support BMI2.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "short and unrolled assembly paths share one call boundary in the arithmetic hot path"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    if len < 4 {
        if len == 0 {
            return (0, 0);
        }
        let carry: Limb;
        let borrow: Limb;
        // SAFETY: 1 <= len <= 3 bounds all accesses within the aligned initialized,
        // disjoint spans. Backend selection establishes BMI2 support. Separate
        // registers retain product carry and boolean subtraction borrow; only
        // caller-saved registers are used and no chunk setup is necessary.
        unsafe {
            asm!(
                "mulxq (%rsi), %r8, %rax",
                "movl $0, %r10d",
                "subq %r8, (%rdi)",
                "adcq $0, %r10",
                "decq %r11",
                "jz 3f",
                "2:",
                "mulxq 8(%rsi), %r8, %r9",
                "leaq 8(%rsi), %rsi",
                "leaq 8(%rdi), %rdi",
                "addq %rax, %r8",
                "adcq $0, %r9",
                "addq %r10, %r8",
                "movl $0, %r10d",
                "adcq $0, %r10",
                "subq %r8, (%rdi)",
                "adcq $0, %r10",
                "movq %r9, %rax",
                "decq %r11",
                "jnz 2b",
                "3:",
                "movq %r10, %rdx",
                inout("rdi") dst => _,
                inout("rsi") src => _,
                inout("r11") len => _,
                inout("rdx") scalar => borrow,
                out("rax") carry,
                out("r8") _,
                out("r9") _,
                out("r10") _,
                options(nostack, att_syntax),
            );
        }
        return (carry, borrow);
    }
    let carry: Limb;
    let borrow_out: Limb;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: len >= 4 gives at least one four-limb block and a shorter tail
    // within the aligned initialized disjoint spans. The byte bound keeps
    // signed counters representable through their terminating -1. Product
    // plus carry is below B^2, and low-digit addition overflow excludes
    // subtraction underflow, preserving a binary borrow. The caller proves
    // BMI2; RAX and RDX hold return values only after their last product use.
    unsafe {
        asm!(
            "xorl {carry_hi:e}, {carry_hi:e}",           // Zero carry_hi register
            "xorl {borrow:e}, {borrow:e}",               // Zero borrow register
            "decq {chunks}",                             // Pre-decrement chunk counter

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
            "movq {carry_hi}, %rax",                     // Return the high product carry
            "movq {borrow}, %rdx",                       // Scalar is dead; return subtraction borrow

            carry_hi = out(reg) _,
            borrow = out(reg) _,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            inout("rdx") scalar => borrow_out,
            out("rax") carry,
            out("rcx") _,
            out("r8") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
    (carry, borrow_out)
}
