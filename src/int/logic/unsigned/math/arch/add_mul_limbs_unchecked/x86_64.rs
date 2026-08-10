//! Baseline x86-64 multiply-add limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// This computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// This is the vanilla `x86_64` implementation utilizing the standard `mulq`
/// instruction for 64x64->128-bit multiplication and `addq`/`adcq` for
/// multi-precision accumulation.
///
/// The loop is unrolled by a factor of 4 for better instruction-level
/// parallelism and pipeline utilization.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(clippy::inline_always, reason = "Performance critical inner loop")]
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

    // SAFETY: Caller guarantees dst and src are valid for `len` elements
    unsafe {
        asm!(
            "xorl {carry:e}, {carry:e}", // Zero out carry register

            "decq {chunks}",             // Decrement loop counter, sets SF if negative
            "js 1f",                     // Jump to remainder if chunks < 0
            ".p2align 4",
            "2:",                        // --- Unrolled Loop x4 ---

            // Limb 0
            "movq {scalar}, %rax",       // Load scalar into RAX (implied operand for mulq)
            "mulq 0({src})",             // RDX:RAX = src[0] * scalar
            "addq {carry}, %rax",        // RAX += carry
            "adcq $0, %rdx",             // RDX += CF (propagate carry to high word)
            "movq 0({dst}), %rcx",       // load dst[0]
            "addq %rax, %rcx",           // rcx = dst[0] + RAX (low word)
            "movq %rcx, 0({dst})",       // store to dst[0]
            "adcq $0, %rdx",             // RDX += CF (propagate carry to next iteration)
            "movq %rdx, {carry}",        // carry = RDX

            // Limb 1
            "movq {scalar}, %rax",
            "mulq 8({src})",
            "addq {carry}, %rax",
            "adcq $0, %rdx",
            "movq 8({dst}), %rcx",
            "addq %rax, %rcx",
            "movq %rcx, 8({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {carry}",

            // Limb 2
            "movq {scalar}, %rax",
            "mulq 16({src})",
            "addq {carry}, %rax",
            "adcq $0, %rdx",
            "movq 16({dst}), %rcx",
            "addq %rax, %rcx",
            "movq %rcx, 16({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {carry}",

            // Limb 3
            "movq {scalar}, %rax",
            "mulq 24({src})",
            "addq {carry}, %rax",
            "adcq $0, %rdx",
            "movq 24({dst}), %rcx",
            "addq %rax, %rcx",
            "movq %rcx, 24({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {carry}",

            "leaq 32({src}), {src}",     // src += 4 limbs
            "leaq 32({dst}), {dst}",     // dst += 4 limbs
            "decq {chunks}",             // chunks -= 1
            "jns 2b",                    // loop if chunks >= 0

            "1:",                        // --- Remainder Loop ---
            "decq {rem}",                // rem -= 1
            "js 4f",                     // jump to end if rem < 0
            ".p2align 4",
            "3:",
            "movq {scalar}, %rax",
            "mulq 0({src})",
            "addq {carry}, %rax",
            "adcq $0, %rdx",
            "movq 0({dst}), %rcx",
            "addq %rax, %rcx",
            "movq %rcx, 0({dst})",
            "adcq $0, %rdx",
            "movq %rdx, {carry}",

            "leaq 8({src}), {src}",      // src += 1 limb
            "leaq 8({dst}), {dst}",      // dst += 1 limb
            "decq {rem}",
            "jns 3b",                    // loop if rem >= 0
            "4:",                        // --- End ---

            carry = out(reg) carry,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            scalar = in(reg) scalar,
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            options(nostack, att_syntax)
        );
    }
    carry
}
