//! 32-bit x86 multiply-subtract limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `src` by one limb, subtract it from `dst`, and return the final
/// multiplication carry and subtraction borrow.
///
/// # Safety
///
/// `src` and `dst` must cover `len` limbs and must not overlap.
#[allow(
    clippy::inline_always,
    reason = "Knuth division and Toom interpolation repeatedly invoke this inner loop"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let carry: Limb;
    let borrow: Limb;

    // EAX:EDX forms src[i]*scalar + carry. The scalar and remaining length
    // occupy two temporary stack words: 32-bit x86 has only four allocatable
    // non-implicit GPRs once EAX:EDX are reserved by MUL, exactly enough for
    // dst, src, carry, and borrow. Borrow is encoded as 0 or -1; adding one
    // restores the prior borrow to CF, SBB applies it, and the following
    // `sbb borrow, borrow` recreates the mask from the new CF. Saving both
    // inputs before any output is written makes the late-output overlaps sound.
    // The four-limb body amortizes the stack-resident loop counter.
    // SAFETY: the caller supplies both len-limb spans. Every memory access is
    // guarded by len, MUL is total, and both temporary stack words are balanced.
    unsafe {
        asm!(
            "pushl {scalar}",
            "pushl {len}",
            "xorl {carry}, {carry}",
            "xorl {borrow}, {borrow}",
            "cmpl $4, (%esp)",
            "jb 2f",
            ".p2align 4",
            "1:",
            "movl 4(%esp), %eax",
            "mull 0({src})",
            "addl {carry}, %eax",
            "adcl $0, %edx",
            "movl %edx, {carry}",
            "addl $1, {borrow}",
            "sbbl %eax, 0({dst})",
            "sbbl {borrow}, {borrow}",

            "movl 4(%esp), %eax",
            "mull 4({src})",
            "addl {carry}, %eax",
            "adcl $0, %edx",
            "movl %edx, {carry}",
            "addl $1, {borrow}",
            "sbbl %eax, 4({dst})",
            "sbbl {borrow}, {borrow}",

            "movl 4(%esp), %eax",
            "mull 8({src})",
            "addl {carry}, %eax",
            "adcl $0, %edx",
            "movl %edx, {carry}",
            "addl $1, {borrow}",
            "sbbl %eax, 8({dst})",
            "sbbl {borrow}, {borrow}",

            "movl 4(%esp), %eax",
            "mull 12({src})",
            "addl {carry}, %eax",
            "adcl $0, %edx",
            "movl %edx, {carry}",
            "addl $1, {borrow}",
            "sbbl %eax, 12({dst})",
            "sbbl {borrow}, {borrow}",

            "addl $16, {src}",
            "addl $16, {dst}",
            "subl $4, (%esp)",
            "cmpl $4, (%esp)",
            "jae 1b",
            "2:",
            "cmpl $0, (%esp)",
            "je 4f",
            ".p2align 4",
            "3:",
            "movl 4(%esp), %eax",
            "mull 0({src})",
            "addl {carry}, %eax",
            "adcl $0, %edx",
            "movl %edx, {carry}",
            "addl $1, {borrow}",
            "sbbl %eax, 0({dst})",
            "sbbl {borrow}, {borrow}",
            "addl $4, {src}",
            "addl $4, {dst}",
            "decl (%esp)",
            "jnz 3b",
            "4:",
            "negl {borrow}",
            "addl $8, %esp",
            carry = lateout(reg) carry,
            borrow = lateout(reg) borrow,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            len = in(reg) len,
            scalar = in(reg) scalar,
            lateout("eax") _,
            lateout("edx") _,
            options(att_syntax)
        );
    }
    (carry, borrow)
}
