//! BMI2-only `mulx` variants for `x86_64` without ADX.
//!
//! Uses `mulx` (BMI2) for flag-free multiplication but standard `adc`/`sbb`
//! for carry/borrow chains. Dispatched at runtime when BMI2 is available
//! but ADX is not.

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
/// This implementation utilizes the `x86_64` BMI2 `mulxq` instruction for
/// 64x64->128-bit multiplication, avoiding flag clobbers.
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
            "2:",
            "mulxq 0({src}), %rax, %r8",
            "addq {carry_hi}, %rax",
            "adcq $0, %r8",
            "movq %r8, {carry_hi}",
            "addq {borrow}, %rax",
            "movq $0, {borrow}",
            "adcq $0, {borrow}",
            "subq %rax, 0({dst})",
            "adcq $0, {borrow}",

            "mulxq 8({src}), %rax, %r8",
            "addq {carry_hi}, %rax",
            "adcq $0, %r8",
            "movq %r8, {carry_hi}",
            "addq {borrow}, %rax",
            "movq $0, {borrow}",
            "adcq $0, {borrow}",
            "subq %rax, 8({dst})",
            "adcq $0, {borrow}",

            "mulxq 16({src}), %rax, %r8",
            "addq {carry_hi}, %rax",
            "adcq $0, %r8",
            "movq %r8, {carry_hi}",
            "addq {borrow}, %rax",
            "movq $0, {borrow}",
            "adcq $0, {borrow}",
            "subq %rax, 16({dst})",
            "adcq $0, {borrow}",

            "mulxq 24({src}), %rax, %r8",
            "addq {carry_hi}, %rax",
            "adcq $0, %r8",
            "movq %r8, {carry_hi}",
            "addq {borrow}, %rax",
            "movq $0, {borrow}",
            "adcq $0, {borrow}",
            "subq %rax, 24({dst})",
            "adcq $0, {borrow}",

            "leaq 32({src}), {src}",
            "leaq 32({dst}), {dst}",
            "decq {chunks}",
            "jns 2b",

            "1:",
            "decq {rem}",
            "js 4f",
            "3:",
            "mulxq 0({src}), %rax, %r8",
            "addq {carry_hi}, %rax",
            "adcq $0, %r8",
            "movq %r8, {carry_hi}",
            "addq {borrow}, %rax",
            "movq $0, {borrow}",
            "adcq $0, {borrow}",
            "subq %rax, 0({dst})",
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
            in("rdx") scalar,
            out("rax") _,
            out("r8") _,
            options(nostack, att_syntax)
        );
    }
    (carry_hi, borrow)
}
