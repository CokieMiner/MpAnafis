//! 32-bit x86 fused dual-row multiply-add kernel.
//!
//! Accumulates two overlapping scalar-product rows in one source traversal
//! using 32-bit `mull` ($32 \times 32 \to 64$-bit into `%edx:%eax`) and 3-word stack state.

use core::arch::asm;

use super::Limb;

/// Accumulates two interleaved scalar-product rows and returns their separate carries.
///
/// EAX and EDX initially carry the scalars, then hold each widened product.
/// The scalars and count remain on the stack during the loop; four further
/// registers hold pointers and row carries. Empty spans access no pointers.
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
    if len == 0 {
        return (0, 0);
    }

    if len == 1 {
        let carry0: Limb;
        let carry1: Limb;
        // SAFETY: len == 1 bounds both aligned destination accesses and the
        // source read. The destination is initialized, writable, and disjoint.
        // EAX is an early output because the first product overwrites it
        // before s1 is consumed, including when both scalars have equal values.
        unsafe {
            asm!(
                "mull 0({src})",                         // %edx:%eax = src[0] * s0 (64-bit product)
                "addl %eax, 0({dst})",                   // dst[0] += %eax
                "adcl $0, %edx",                         // %edx += CF
                "movl %edx, {carry0}",                   // carry0 = %edx
                "movl {s1}, %eax",                       // %eax = s1
                "mull 0({src})",                         // %edx:%eax = src[0] * s1
                "addl %eax, 4({dst})",                   // dst[1] += %eax
                "adcl $0, %edx",                         // %edx += CF (carry1 in %edx)
                inout("eax") s0 => _,
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

    // SAFETY: the preceding returns establish len >= 2. Each two-limb block
    // reads two source limbs and updates three initialized destination limbs;
    // a final odd limb uses the remaining source and destination pair. The
    // disjoint live spans bound every aligned access. The three pushes are
    // restored on the sole exit. All scalar and count inputs are saved before
    // the product registers or overlapping late-output carry registers change.
    unsafe {
        asm!(
            "pushl %eax",                                // Save s0 at 8(%esp)
            "pushl %edx",                                // Save s1 at 4(%esp)
            "pushl {len}",                               // Save len at 0(%esp)
            "xorl {carry0}, {carry0}",                   // Zero row 0 carry
            "xorl {carry1}, {carry1}",                   // Zero row 1 carry

            // Main 2-way unrolled loop body
            "1:",

            // [Limb 0 - Row 0]
            "movl 8(%esp), %eax",                        // Load s0 from stack
            "mull 0({src})",                             // %edx:%eax = src[0] * s0
            "addl {carry0}, %eax",                       // %eax += carry0
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 0({dst})",                       // dst[0] += %eax
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry0}",                       // Update row 0 carry

            // [Limb 0 - Row 1]
            "movl 4(%esp), %eax",                        // Load s1 from stack
            "mull 0({src})",                             // %edx:%eax = src[0] * s1
            "addl {carry1}, %eax",                       // %eax += carry1
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 4({dst})",                       // dst[1] += %eax
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry1}",                       // Update row 1 carry

            // [Limb 1 - Row 0]
            "movl 8(%esp), %eax",                        // Load s0
            "mull 4({src})",                             // src[1] * s0
            "addl {carry0}, %eax",                       // %eax += carry0
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 4({dst})",                       // dst[1] += %eax
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry0}",                       // Update row 0 carry

            // [Limb 1 - Row 1]
            "movl 4(%esp), %eax",                        // Load s1
            "mull 4({src})",                             // src[1] * s1
            "addl {carry1}, %eax",                       // %eax += carry1
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 8({dst})",                       // dst[2] += %eax
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry1}",                       // Update row 1 carry

            // Advance pointers by 2 limbs (8 bytes)
            "addl $8, {src}",
            "addl $8, {dst}",
            "subl $2, 0(%esp)",                          // Decrement remaining count on stack
            "cmpl $2, 0(%esp)",                          // Check if remaining >= 2
            "jae 1b",                                    // Repeat loop

            // Remainder processing (0 or 1 limb)
            "2:",
            "cmpl $0, 0(%esp)",                          // Check if remaining == 0
            "je 4f",                                     // If 0, skip to cleanup (4f)

            // 1-limb tail
            "3:",
            "movl 8(%esp), %eax",                        // Load s0
            "mull 0({src})",                             // src[j] * s0
            "addl {carry0}, %eax",
            "adcl $0, %edx",
            "addl %eax, 0({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry0}",

            "movl 4(%esp), %eax",                        // Load s1
            "mull 0({src})",                             // src[j] * s1
            "addl {carry1}, %eax",
            "adcl $0, %edx",
            "addl %eax, 4({dst})",
            "adcl $0, %edx",
            "movl %edx, {carry1}",

            // Cleanup stack
            "4:",
            "addl $12, %esp",                            // Restore 3 pushed words

            carry0 = lateout(reg) carry0,
            carry1 = lateout(reg) carry1,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            len = in(reg) len,
            inlateout("eax") s0 => _,
            inlateout("edx") s1 => _,
            options(att_syntax)
        );
    }
    (carry0, carry1)
}
