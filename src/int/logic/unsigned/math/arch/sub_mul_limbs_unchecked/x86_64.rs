//! Baseline x86-64 multiply-subtract with four-limb blocks.
//!
//! `mulq` forms each 128-bit product in RDX:RAX. Product carry and subtraction
//! borrow are preserved separately across limbs.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// For B = 2^64, `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
///
/// # Safety
///
/// Both pointers must cover `len` aligned, initialized limbs in disjoint spans;
/// `dst` requires exclusive access. Each span's byte length must fit in `isize`.
/// Zero length permits null pointers.
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
        // SAFETY: 1 <= len <= 3 bounds every access to the caller's aligned,
        // initialized disjoint spans, with exclusive writes to dst. All
        // changed registers are declared. The first
        // product has no incoming carry or borrow; subsequent low-word
        // addition overflow and subtraction underflow are mutually exclusive.
        unsafe {
            asm!(
                "movq %rcx, %rax",
                "mulq (%rsi)",
                "movq %rdx, %r9",
                "subq %rax, (%rdi)",
                "movl $0, %r10d",
                "adcq $0, %r10",
                "decq %r8",
                "jz 3f",
                "2:",
                "leaq 8(%rsi), %rsi",
                "leaq 8(%rdi), %rdi",
                "movq %rcx, %rax",
                "mulq (%rsi)",
                "addq %r9, %rax",
                "adcq $0, %rdx",
                "movq %rdx, %r9",
                "addq %r10, %rax",
                "movl $0, %r10d",
                "adcq $0, %r10",
                "subq %rax, (%rdi)",
                "adcq $0, %r10",
                "decq %r8",
                "jnz 2b",
                "3:",
                "movq %r9, %rax",
                "movq %r10, %rdx",
                inout("rdi") dst => _,
                inout("rsi") src => _,
                in("rcx") scalar,
                inout("r8") len => _,
                out("rax") carry,
                out("rdx") borrow,
                out("r9") _,
                out("r10") _,
                options(nostack, att_syntax),
            );
        }
        return (carry, borrow);
    }
    let mut carry_hi: Limb;
    let mut borrow: Limb;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: len >= 4 gives at least one complete four-limb block, followed
    // by len % 4 limbs, within the aligned initialized disjoint spans. The
    // byte bound keeps signed counters representable; only their terminating
    // decrement reaches -1. Product plus carry is below B^2. Adding binary
    // borrow to the low digit and subtracting it cannot both overflow, so the
    // two captured flags still sum to a binary borrow. All clobbers are listed.
    unsafe {
        asm!(
            "xorl {carry_hi:e}, {carry_hi:e}",           // Zero carry_hi register
            "xorl {borrow:e}, {borrow:e}",               // Zero borrow register
            "decq {chunks}",                             // Pre-decrement chunk counter

            // Main 4-way unrolled loop body
            "2:",                                        // Loop head label
            // [Limb 0]
            "movq {scalar}, %rax",                       // Load scalar into %rax (implicit operand for mulq)
            "mulq 0({src})",                             // %rdx:%rax = src[0] * scalar
            "movq 0({dst}), %rcx",                       // Load dst[0]
            "addq {carry_hi}, %rax",                     // %rax += carry_hi
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry_hi}",                     // Update carry_hi
            "addq {borrow}, %rax",                       // %rax += borrow
            "movq $0, {borrow}",                         // Clear borrow
            "adcq $0, {borrow}",                         // Capture add overflow
            "subq %rax, %rcx",                           // %rcx = dst[0] - %rax
            "movq %rcx, 0({dst})",                       // Store result back to dst[0]
            "adcq $0, {borrow}",                         // Capture subtraction borrow

            // [Limb 1]
            "movq {scalar}, %rax",                       // Reload scalar
            "mulq 8({src})",                             // %rdx:%rax = src[1] * scalar
            "movq 8({dst}), %rcx",                       // Load dst[1]
            "addq {carry_hi}, %rax",                     // %rax += carry_hi
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry_hi}",                     // Update carry_hi
            "addq {borrow}, %rax",                       // %rax += borrow
            "movq $0, {borrow}",                         // Clear borrow
            "adcq $0, {borrow}",                         // Capture add overflow
            "subq %rax, %rcx",                           // %rcx = dst[1] - %rax
            "movq %rcx, 8({dst})",                       // Store result back to dst[1]
            "adcq $0, {borrow}",                         // Capture subtraction borrow

            // [Limb 2]
            "movq {scalar}, %rax",                       // Reload scalar
            "mulq 16({src})",                            // %rdx:%rax = src[2] * scalar
            "movq 16({dst}), %rcx",                      // Load dst[2]
            "addq {carry_hi}, %rax",                     // %rax += carry_hi
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry_hi}",                     // Update carry_hi
            "addq {borrow}, %rax",                       // %rax += borrow
            "movq $0, {borrow}",                         // Clear borrow
            "adcq $0, {borrow}",                         // Capture add overflow
            "subq %rax, %rcx",                           // %rcx = dst[2] - %rax
            "movq %rcx, 16({dst})",                      // Store result back to dst[2]
            "adcq $0, {borrow}",                         // Capture subtraction borrow

            // [Limb 3]
            "movq {scalar}, %rax",                       // Reload scalar
            "mulq 24({src})",                            // %rdx:%rax = src[3] * scalar
            "movq 24({dst}), %rcx",                      // Load dst[3]
            "addq {carry_hi}, %rax",                     // %rax += carry_hi
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry_hi}",                     // Update carry_hi
            "addq {borrow}, %rax",                       // %rax += borrow
            "movq $0, {borrow}",                         // Clear borrow
            "adcq $0, {borrow}",                         // Capture add overflow
            "subq %rax, %rcx",                           // %rcx = dst[3] - %rax
            "movq %rcx, 24({dst})",                      // Store result back to dst[3]
            "adcq $0, {borrow}",                         // Capture subtraction borrow

            "leaq 32({src}), {src}",                     // Advance src pointer by 32 bytes
            "leaq 32({dst}), {dst}",                     // Advance dst pointer by 32 bytes
            "decq {chunks}",                             // Decrement chunks
            "jns 2b",                                    // Repeat while chunks >= 0

            // Remainder processing entry point (0 to 3 limbs)
            "1:",                                        // Remainder entry label
            "decq {rem}",                                // Pre-decrement remainder counter
            "js 4f",                                     // If rem < 0, skip to finish (4f)

            // 1-limb unrolled tail loop
            "3:",                                        // Tail loop label
            "movq {scalar}, %rax",                       // Load scalar into %rax
            "mulq 0({src})",                             // Multiply single src limb
            "movq 0({dst}), %rcx",                       // Load the destination limb
            "addq {carry_hi}, %rax",                     // %rax += carry_hi
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rdx, {carry_hi}",                     // Update carry_hi
            "addq {borrow}, %rax",                       // %rax += borrow
            "movq $0, {borrow}",                         // Clear borrow
            "adcq $0, {borrow}",                         // Capture add overflow
            "subq %rax, %rcx",                           // dst[0] - %rax
            "movq %rcx, 0({dst})",                       // Store updated limb
            "adcq $0, {borrow}",                         // Capture subtraction borrow
            "leaq 8({src}), {src}",                      // Advance src pointer (+8)
            "leaq 8({dst}), {dst}",                      // Advance dst pointer (+8)
            "decq {rem}",                                // Decrement remainder counter
            "jns 3b",                                    // Repeat while rem >= 0

            // Tail completion
            "4:",                                        // Completion label

            carry_hi = out(reg) carry_hi,
            borrow = out(reg) borrow,
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
    (carry_hi, borrow)
}
