//! 32-bit x86 multiply-add limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `src` by one limb, add it into `dst`, and return the carry.
///
/// The complete dependency chain stays in registers. This avoids the stack
/// carry spill produced by LLVM's generic `u64` lowering on 32-bit x86.
///
/// # Safety
///
/// `src` and `dst` must cover `len` limbs and must not overlap.
#[allow(
    clippy::inline_always,
    reason = "Basecase multiplication repeatedly invokes this register-resident inner loop"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let carry: Limb;

    // Each iteration forms p = src[i]*scalar + carry + dst[i]. The two ADC
    // instructions add the only possible low-limb overflows to EDX, so EDX is
    // exactly floor(p/2^32) and remains one limb. The loop touches precisely
    // the caller-provided spans and the zero-length path dereferences nothing.
    // SAFETY: the caller supplies both len-limb spans; x86 MUL is total for all
    // limb values, and every pointer advance is guarded by the remaining count.
    unsafe {
        asm!(
            "xorl {carry}, {carry}",
            "testl {len}, {len}",
            "jz 4f",
            "cmpl $4, {len}",
            "jb 2f",
            ".p2align 4",
            "1:",
            "movl 0({src}), %eax",
            "mull {scalar}",
            "addl {carry}, %eax",
            "adcl $0, %edx",
            "addl %eax, 0({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry}",

            "movl 4({src}), %eax",
            "mull {scalar}",
            "addl {carry}, %eax",
            "adcl $0, %edx",
            "addl %eax, 4({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry}",

            "movl 8({src}), %eax",
            "mull {scalar}",
            "addl {carry}, %eax",
            "adcl $0, %edx",
            "addl %eax, 8({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry}",

            "movl 12({src}), %eax",
            "mull {scalar}",
            "addl {carry}, %eax",
            "adcl $0, %edx",
            "addl %eax, 12({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry}",

            "addl $16, {src}",
            "addl $16, {dst}",
            "subl $4, {len}",
            "cmpl $4, {len}",
            "jae 1b",

            "2:",
            "testl {len}, {len}",
            "jz 4f",
            ".p2align 4",
            "3:",
            "movl ({src}), %eax",
            "mull {scalar}",
            "addl {carry}, %eax",
            "adcl $0, %edx",
            "addl %eax, ({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry}",
            "addl $4, {src}",
            "addl $4, {dst}",
            "decl {len}",
            "jnz 3b",
            "4:",
            carry = out(reg) carry,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            len = inout(reg) len => _,
            scalar = in(reg) scalar,
            out("eax") _,
            out("edx") _,
            options(nostack, att_syntax)
        );
    }
    carry
}
