//! 32-bit x86 write-only dual-row multiplication kernel.

use core::arch::asm;

use super::Limb;

/// Write `src * (s0 + s1 * B)` into `dst` without reading its old contents.
///
/// # Safety
///
/// `src` must be valid for `len` limbs, `dst` must be valid for `len + 2`
/// limbs, and the two regions must not overlap.
#[allow(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "This repeatedly invoked hot kernel must inline, and its write-only dual-row assembly stays visibly unrolled so the initialization invariant remains reviewable"
)]
#[inline(always)]
pub unsafe fn mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) {
    if len == 0 {
        return;
    }

    if len == 1 {
        // The general loop must spill three invariants because both row carries
        // are live. At one limb no loop state exists, so a register-only path
        // is both smaller and measurably faster than paying that setup cost.
        // SAFETY: the caller provides one readable source limb and three
        // writable destination limbs. EAX:EDX contains each exact 32x32-bit
        // product; adding product0.high to product1.low can carry at most one
        // into product1.high, which still fits the third result limb.
        unsafe {
            asm!(
                "mull 0({src})",
                "movl %eax, 0({dst})",
                "movl %edx, {carry0}",
                "movl {s1}, %eax",
                "mull 0({src})",
                "addl {carry0}, %eax",
                "adcl $0, %edx",
                "movl %eax, 4({dst})",
                "movl %edx, 8({dst})",
                inlateout("eax") s0 => _,
                out("edx") _,
                dst = in(reg) dst,
                src = in(reg) src,
                s1 = in(reg) s1,
                carry0 = out(reg) _,
                options(nostack, att_syntax)
            );
        }
        return;
    }

    // EAX:EDX holds each 32x32->64 product. The two row carries and both
    // pointers consume all four remaining allocatable GPRs, so the invariant
    // scalars and remaining length live in three balanced stack words. For
    // every current output position, dst[0] already contains the preceding
    // high-scalar row; the low-scalar row is added to it, while the next
    // high-scalar limb is written to dst[1]. Thus each initialized destination
    // limb is read only after it has been written by this kernel. The two-limb
    // body amortizes the stack-resident counter without extending live state.
    // SAFETY: len is nonzero and the caller provides `len` readable source
    // limbs plus `len+2` writable destination limbs. Pointer advances total
    // exactly len limbs, MUL is total, and all three pushes are restored.
    unsafe {
        asm!(
            "pushl {s0}",
            "pushl {s1}",
            "pushl {len}",

            // Seed both rows from src[0].
            "movl 8(%esp), %eax",
            "mull 0({src})",
            "movl %eax, 0({dst})",
            "movl %edx, {carry0}",
            "movl 4(%esp), %eax",
            "mull 0({src})",
            "movl %eax, 4({dst})",
            "movl %edx, {carry1}",
            "addl $4, {src}",
            "addl $4, {dst}",
            "decl 0(%esp)",

            // Consume pairs while at least two source limbs remain.
            "cmpl $2, 0(%esp)",
            "jb 2f",
            "1:",
            "movl 8(%esp), %eax",
            "mull 0({src})",
            "addl {carry0}, %eax",
            "adcl $0, %edx",
            "addl 0({dst}), %eax",
            "adcl $0, %edx",
            "movl %eax, 0({dst})",
            "movl %edx, {carry0}",
            "movl 4(%esp), %eax",
            "mull 0({src})",
            "addl {carry1}, %eax",
            "adcl $0, %edx",
            "movl %eax, 4({dst})",
            "movl %edx, {carry1}",

            "movl 8(%esp), %eax",
            "mull 4({src})",
            "addl {carry0}, %eax",
            "adcl $0, %edx",
            "addl 4({dst}), %eax",
            "adcl $0, %edx",
            "movl %eax, 4({dst})",
            "movl %edx, {carry0}",
            "movl 4(%esp), %eax",
            "mull 4({src})",
            "addl {carry1}, %eax",
            "adcl $0, %edx",
            "movl %eax, 8({dst})",
            "movl %edx, {carry1}",

            "addl $8, {src}",
            "addl $8, {dst}",
            "subl $2, 0(%esp)",
            "cmpl $2, 0(%esp)",
            "jae 1b",

            // At most one source limb remains.
            "2:",
            "cmpl $0, 0(%esp)",
            "je 3f",
            "movl 8(%esp), %eax",
            "mull 0({src})",
            "addl {carry0}, %eax",
            "adcl $0, %edx",
            "addl 0({dst}), %eax",
            "adcl $0, %edx",
            "movl %eax, 0({dst})",
            "movl %edx, {carry0}",
            "movl 4(%esp), %eax",
            "mull 0({src})",
            "addl {carry1}, %eax",
            "adcl $0, %edx",
            "movl %eax, 4({dst})",
            "movl %edx, {carry1}",
            "addl $4, {dst}",

            // Close the overlap between the final two row carries.
            "3:",
            "addl {carry0}, 0({dst})",
            "adcl $0, {carry1}",
            "movl {carry1}, 4({dst})",
            "addl $12, %esp",

            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            len = in(reg) len,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            carry0 = lateout(reg) _,
            carry1 = lateout(reg) _,
            lateout("eax") _,
            lateout("edx") _,
            options(att_syntax)
        );
    }
}
