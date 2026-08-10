//! Baseline x86-64 multiply-subtract limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// This computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// This is the vanilla `x86_64` inline assembly implementation using `mulq`,
/// `addq`, `adcq`, `subq` and `sbbq`.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(clippy::inline_always, reason = "Performance critical inner loop")]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry_hi: Limb;
    let mut borrow: Limb;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: Caller guarantees dst and src are valid for `len` elements
    unsafe {
        asm!(
            "xorl {carry_hi:e}, {carry_hi:e}",
            "xorl {borrow:e}, {borrow:e}",

            "decq {chunks}",
            "js 1f",
            ".p2align 4",
            "2:",
            "movq {scalar}, %rax",
            "mulq 0({src})",
            "addq {carry_hi}, %rax",
            "adcq $0, %rdx",
            "movq %rdx, {carry_hi}",
            "addq {borrow}, %rax",
            "movq $0, {borrow}",
            "adcq $0, {borrow}",
            "movq 0({dst}), %rcx",
            "subq %rax, %rcx",
            "movq %rcx, 0({dst})",
            "adcq $0, {borrow}",

            "movq {scalar}, %rax",
            "mulq 8({src})",
            "addq {carry_hi}, %rax",
            "adcq $0, %rdx",
            "movq %rdx, {carry_hi}",
            "addq {borrow}, %rax",
            "movq $0, {borrow}",
            "adcq $0, {borrow}",
            "movq 8({dst}), %rcx",
            "subq %rax, %rcx",
            "movq %rcx, 8({dst})",
            "adcq $0, {borrow}",

            "movq {scalar}, %rax",
            "mulq 16({src})",
            "addq {carry_hi}, %rax",
            "adcq $0, %rdx",
            "movq %rdx, {carry_hi}",
            "addq {borrow}, %rax",
            "movq $0, {borrow}",
            "adcq $0, {borrow}",
            "movq 16({dst}), %rcx",
            "subq %rax, %rcx",
            "movq %rcx, 16({dst})",
            "adcq $0, {borrow}",

            "movq {scalar}, %rax",
            "mulq 24({src})",
            "addq {carry_hi}, %rax",
            "adcq $0, %rdx",
            "movq %rdx, {carry_hi}",
            "addq {borrow}, %rax",
            "movq $0, {borrow}",
            "adcq $0, {borrow}",
            "movq 24({dst}), %rcx",
            "subq %rax, %rcx",
            "movq %rcx, 24({dst})",
            "adcq $0, {borrow}",

            "leaq 32({src}), {src}",
            "leaq 32({dst}), {dst}",
            "decq {chunks}",
            "jns 2b",

            "1:",
            "decq {rem}",
            "js 4f",
            ".p2align 4",
            "3:",
            "movq {scalar}, %rax",
            "mulq 0({src})",
            "addq {carry_hi}, %rax",
            "adcq $0, %rdx",
            "movq %rdx, {carry_hi}",
            "addq {borrow}, %rax",
            "movq $0, {borrow}",
            "adcq $0, {borrow}",
            "movq 0({dst}), %rcx",
            "subq %rax, %rcx",
            "movq %rcx, 0({dst})",
            "adcq $0, {borrow}",

            "leaq 8({src}), {src}",
            "leaq 8({dst}), {dst}",
            "decq {rem}",
            "jns 3b",
            "4:",

            carry_hi = out(reg) carry_hi,
            borrow = out(reg) borrow,
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
    (carry_hi, borrow)
}
