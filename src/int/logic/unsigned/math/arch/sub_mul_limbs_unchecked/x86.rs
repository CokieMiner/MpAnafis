//! 32-bit x86 fused multiply-subtract limb kernel.
//!
//! `mull` forms each 64-bit product in EDX:EAX. Two stack words hold the scalar
//! and loop count; a register holds the subtraction borrow mask.

use core::arch::asm;

use super::Limb;

/// Multiply `src` by one limb, subtract it from `dst`, and return the final
/// multiplication carry and subtraction borrow.
///
/// For B = 2^32, `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
/// `addl $1` restores CF from the saved zero or all-ones borrow mask.
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
        // SAFETY: 1 <= len <= 3 bounds each access to the caller's aligned, initialized
        // disjoint spans. Both pushed words are removed before leaving asm.
        // len and scalar are consumed before the late outputs may overwrite
        // their registers; the pointers remain live in distinct registers.
        unsafe {
            asm!(
                "pushl {scalar}",
                "pushl {len}",
                "movl 4(%esp), %eax",
                "mull ({src})",
                "movl %edx, {carry}",
                "subl %eax, ({dst})",
                "sbbl {borrow}, {borrow}",
                "decl (%esp)",
                "jz 3f",
                "2:",
                "addl $4, {src}",
                "addl $4, {dst}",
                "movl 4(%esp), %eax",
                "mull ({src})",
                "addl {carry}, %eax",
                "adcl $0, %edx",
                "movl %edx, {carry}",
                "addl $1, {borrow}",
                "sbbl %eax, ({dst})",
                "sbbl {borrow}, {borrow}",
                "decl (%esp)",
                "jnz 2b",
                "3:",
                "negl {borrow}",
                "addl $8, %esp",
                carry = lateout(reg) carry,
                borrow = lateout(reg) borrow,
                dst = inout(reg) dst => _,
                src = inout(reg) src => _,
                len = in(reg) len,
                scalar = in(reg) scalar,
                out("eax") _,
                out("edx") _,
                options(att_syntax),
            );
        }
        return (carry, borrow);
    }
    let carry: Limb;
    let borrow: Limb;

    // SAFETY: len >= 4 gives complete four-limb blocks and a tail shorter than
    // four limbs in the aligned initialized disjoint spans. Writes are
    // exclusive. Product plus carry is below B^2; each SBB captures a binary
    // borrow as an all-ones mask. Both inputs are pushed before late outputs
    // reuse their registers, and the final stack adjustment removes both words.
    unsafe {
        asm!(
            "pushl {scalar}",                            // Save scalar at 4(%esp)
            "pushl {len}",                               // Save loop counter at 0(%esp)
            "xorl {carry}, {carry}",                     // Zero carry register
            "xorl {borrow}, {borrow}",                   // Zero borrow mask (0 = no borrow)

            // Main 4-way unrolled loop body
            "1:",

            // [Limb 0]
            "movl 4(%esp), %eax",                        // Load scalar from stack into %eax
            "mull 0({src})",                             // %edx:%eax = src[0] * scalar (64-bit product)
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF (propagate carry to high product)
            "movl %edx, {carry}",                        // Update running multiplication carry
            "addl $1, {borrow}",                         // Restore borrow mask to CF: (-1 + 1 -> CF=1; 0 + 1 -> CF=0)
            "sbbl %eax, 0({dst})",                       // dst[0] = dst[0] - low product - CF
            "sbbl {borrow}, {borrow}",                   // Recapture CF into borrow mask (CF=1 -> -1, CF=0 -> 0)

            // [Limb 1]
            "movl 4(%esp), %eax",                        // Load scalar
            "mull 4({src})",                             // %edx:%eax = src[1] * scalar
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // Update carry
            "addl $1, {borrow}",                         // Restore borrow to CF
            "sbbl %eax, 4({dst})",                       // dst[1] -= product + CF
            "sbbl {borrow}, {borrow}",                   // Recapture borrow mask

            // [Limb 2]
            "movl 4(%esp), %eax",                        // Load scalar
            "mull 8({src})",                             // %edx:%eax = src[2] * scalar
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // Update carry
            "addl $1, {borrow}",                         // Restore borrow to CF
            "sbbl %eax, 8({dst})",                       // dst[2] -= product + CF
            "sbbl {borrow}, {borrow}",                   // Recapture borrow mask

            // [Limb 3]
            "movl 4(%esp), %eax",                        // Load scalar
            "mull 12({src})",                            // %edx:%eax = src[3] * scalar
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // Update carry
            "addl $1, {borrow}",                         // Restore borrow to CF
            "sbbl %eax, 12({dst})",                      // dst[3] -= product + CF
            "sbbl {borrow}, {borrow}",                   // Recapture borrow mask

            // Advance pointers by 4 limbs (16 bytes)
            "addl $16, {src}",
            "addl $16, {dst}",
            "subl $4, (%esp)",                           // Decrement remaining counter on stack
            "cmpl $4, (%esp)",                           // Check if remaining >= 4
            "jae 1b",                                    // Repeat loop

            // Remainder processing (0 to 3 limbs)
            "2:",
            "cmpl $0, (%esp)",                           // Check if remaining == 0
            "je 4f",                                     // If 0, skip to cleanup (4f)

            // 1-limb unrolled tail loop
            "3:",
            "movl 4(%esp), %eax",                        // Load scalar
            "mull 0({src})",                             // Multiply single limb
            "addl {carry}, %eax",                        // Add carry
            "adcl $0, %edx",                             // Propagate carry
            "movl %edx, {carry}",                        // Update carry
            "addl $1, {borrow}",                         // Restore borrow to CF
            "sbbl %eax, 0({dst})",                       // dst[0] -= product + CF
            "sbbl {borrow}, {borrow}",                   // Recapture borrow mask
            "addl $4, {src}",                            // Advance src by 4 bytes
            "addl $4, {dst}",                            // Advance dst by 4 bytes
            "decl (%esp)",                               // Decrement remainder on stack
            "jnz 3b",                                    // Repeat while != 0

            // Cleanup stack and convert borrow mask
            "4:",
            "negl {borrow}",                             // Convert borrow mask: -1 -> 1, 0 -> 0
            "addl $8, %esp",                             // Restore stack pointer

            carry = lateout(reg) carry,
            borrow = lateout(reg) borrow,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            len = in(reg) len,
            scalar = in(reg) scalar,
            out("eax") _,
            out("edx") _,
            options(att_syntax)
        );
    }
    (carry, borrow)
}
