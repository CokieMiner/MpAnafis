//! 32-bit x86 fused multiply-subtract limb kernel.
//!
//! Uses hardware 32-bit `mull` ($32 \times 32 \to 64$-bit into `%edx:%eax`),
//! 2-word temporary stack storage for registers under extreme 32-bit GPR pressure,
//! and branchless borrow-mask toggling via `addl $1` and `sbbl`.

use core::arch::asm;

use super::Limb;

/// Multiply `src` by one limb, subtract it from `dst`, and return the final
/// multiplication carry and subtraction borrow.
///
/// Computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// 32-bit x86 provides only 6 allocatable general-purpose registers (`%eax`, `%edx`, `%ecx`, `%ebx`,
/// `%esi`, `%edi`). Once `%eax` and `%edx` are allocated for `mull`, exactly 4 GPRs remain for `dst`,
/// `src`, `carry`, and `borrow`. The loop counter and `scalar` are kept in the top two stack words
/// (`0(%esp)` and `4(%esp)`). Borrow is maintained as a $-1$/$0$ bitmask: `addl $1, {borrow}` restores
/// the borrow bit to `CF`, `sbbl` performs the multi-precision subtraction, and `sbbl {borrow}, {borrow}`
/// recaptures `CF` into the mask without branches.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 32-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 32-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical inner loop for 32-bit multi-precision Knuth division and basecase subtraction"
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

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Stack pushes (`pushl`) are strictly matched with `addl $8, %esp` before return.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "pushl {scalar}",                            // Save scalar at 4(%esp)
            "pushl {len}",                               // Save loop counter at 0(%esp)
            "xorl {carry}, {carry}",                     // Zero carry register
            "xorl {borrow}, {borrow}",                   // Zero borrow mask (0 = no borrow)

            "cmpl $4, (%esp)",                           // Check if len < 4
            "jb 2f",                                     // If len < 4, jump to remainder handler (2f)

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
