//! 32-bit x86 fused dual-row multiply-add kernel.

use core::arch::asm;

use super::Limb;

/// Accumulate two overlapping scalar-product rows in one source traversal.
///
/// # Safety
///
/// `src` must be valid for `len` readable limbs, `dst` must be valid for
/// `len + 1` readable and writable limbs, and the regions must not overlap.
#[allow(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "This repeatedly invoked hot kernel must inline, and its dual-row assembly stays visibly unrolled so the carry-chain proof remains reviewable"
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

    if len == 1 {
        let carry0: Limb;
        let carry1: Limb;
        // With no loop-carried state, both rows fit entirely in registers and
        // avoid the invariant spills required by the general 32-bit loop.
        // SAFETY: the caller provides one source limb and two destination
        // limbs. Each two-step ADD/ADC chain exactly accumulates one 64-bit
        // product, one zero initial carry, and one 32-bit destination limb.
        unsafe {
            asm!(
                "mull 0({src})",
                "addl %eax, 0({dst})",
                "adcl $0, %edx",
                "movl %edx, {carry0}",
                "movl {s1}, %eax",
                "mull 0({src})",
                "addl %eax, 4({dst})",
                "adcl $0, %edx",
                inlateout("eax") s0 => _,
                out("edx") carry1,
                dst = in(reg) dst,
                src = in(reg) src,
                s1 = in(reg) s1,
                carry0 = out(reg) carry0,
                options(nostack, att_syntax)
            );
        }
        return (carry0, carry1);
    }

    let carry0: Limb;
    let carry1: Limb;
    // EAX:EDX holds each 32x32->64 product. Both row carries and both pointers
    // occupy the four remaining allocatable GPRs, so the invariant scalars and
    // remaining length use three balanced stack words. Each row's ADD/ADC
    // sequence proves carry_j = high(src[i]*sj + old_carry_j + dst[i+j]).
    // Two source limbs per body amortize the stack counter and loop branch.
    // SAFETY: the caller provides `len` source limbs and `len+1` destination
    // limbs. The unrolled and tail bodies together advance exactly `len`
    // positions, every product/addition wraps at its documented limb boundary,
    // and all three pushes are restored before returning.
    unsafe {
        asm!(
            "pushl {s0}",
            "pushl {s1}",
            "pushl {len}",
            "xorl {carry0}, {carry0}",
            "xorl {carry1}, {carry1}",

            "cmpl $2, 0(%esp)",
            "jb 2f",
            "1:",
            "movl 8(%esp), %eax",
            "mull 0({src})",
            "addl {carry0}, %eax",
            "adcl $0, %edx",
            "addl %eax, 0({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry0}",
            "movl 4(%esp), %eax",
            "mull 0({src})",
            "addl {carry1}, %eax",
            "adcl $0, %edx",
            "addl %eax, 4({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry1}",

            "movl 8(%esp), %eax",
            "mull 4({src})",
            "addl {carry0}, %eax",
            "adcl $0, %edx",
            "addl %eax, 4({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry0}",
            "movl 4(%esp), %eax",
            "mull 4({src})",
            "addl {carry1}, %eax",
            "adcl $0, %edx",
            "addl %eax, 8({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry1}",

            "addl $8, {src}",
            "addl $8, {dst}",
            "subl $2, 0(%esp)",
            "cmpl $2, 0(%esp)",
            "jae 1b",

            "2:",
            "cmpl $0, 0(%esp)",
            "je 3f",
            "movl 8(%esp), %eax",
            "mull 0({src})",
            "addl {carry0}, %eax",
            "adcl $0, %edx",
            "addl %eax, 0({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry0}",
            "movl 4(%esp), %eax",
            "mull 0({src})",
            "addl {carry1}, %eax",
            "adcl $0, %edx",
            "addl %eax, 4({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry1}",
            "3:",
            "addl $12, %esp",

            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            len = in(reg) len,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            carry0 = lateout(reg) carry0,
            carry1 = lateout(reg) carry1,
            lateout("eax") _,
            lateout("edx") _,
            options(att_syntax)
        );
    }
    (carry0, carry1)
}
