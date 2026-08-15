//! 32-bit x86 fused multiply-add limb kernel.
//!
//! Uses hardware 32-bit `mull` ($32 \times 32 \to 64$-bit into `%edx:%eax`)
//! and registers for complete carry propagation, preventing LLVM stack spills on 32-bit targets.

use core::arch::asm;

use super::Limb;

/// Multiply `len` 32-bit limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// Computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// 32-bit x86 has only 6 allocatable general-purpose registers (excluding `%esp` and `%ebp`).
/// This implementation maintains the entire inner multiply-accumulate dependency loop strictly
/// inside `%eax`, `%edx`, and caller registers, avoiding catastrophic stack spills.
/// The loop is 4-way unrolled (16 bytes per iteration).
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized limbs.
/// - `src` must point to a readable buffer of at least `len` initialized limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in 32-bit multi-precision hot paths"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let carry: Limb;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `4`, `8`, `12`, `16`) remain within `len * 4` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "xorl {carry}, {carry}",                     // Zero carry accumulator register
            "testl {len}, {len}",                        // Test if len == 0
            "jz 4f",                                     // If zero, jump to exit (4f)
            "cmpl $4, {len}",                            // Compare len with 4
            "jb 2f",                                     // If len < 4, jump to remainder loop (2f)

            // Main 4-way unrolled loop body
            "1:",                                        // Loop head label
            // [Limb 0]
            "movl 0({src}), %eax",                       // Load src[0] into %eax
            "mull {scalar}",                             // %edx:%eax = src[0] * scalar (64-bit product)
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 0({dst})",                       // dst[0] += low product sum
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // carry = %edx

            // [Limb 1]
            "movl 4({src}), %eax",                       // Load src[1]
            "mull {scalar}",                             // %edx:%eax = src[1] * scalar
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 4({dst})",                       // dst[1] += low product sum
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // carry = %edx

            // [Limb 2]
            "movl 8({src}), %eax",                       // Load src[2]
            "mull {scalar}",                             // %edx:%eax = src[2] * scalar
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 8({dst})",                       // dst[2] += low product sum
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // carry = %edx

            // [Limb 3]
            "movl 12({src}), %eax",                      // Load src[3]
            "mull {scalar}",                             // %edx:%eax = src[3] * scalar
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, 12({dst})",                      // dst[3] += low product sum
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // carry = %edx

            "addl $16, {src}",                           // Advance src pointer by 16 bytes
            "addl $16, {dst}",                           // Advance dst pointer by 16 bytes
            "subl $4, {len}",                            // Decrement remaining length by 4
            "cmpl $4, {len}",                            // Check if len >= 4
            "jae 1b",                                    // Repeat while len >= 4

            // Remainder entry point (0 to 3 limbs)
            "2:",                                        // Remainder entry label
            "testl {len}, {len}",                        // Test if remaining len == 0
            "jz 4f",                                     // If zero, jump to exit (4f)

            // 1-limb unrolled tail loop
            "3:",                                        // Tail loop label
            "movl ({src}), %eax",                        // Load single src limb
            "mull {scalar}",                             // %edx:%eax = src * scalar
            "addl {carry}, %eax",                        // %eax += carry
            "adcl $0, %edx",                             // %edx += CF
            "addl %eax, ({dst})",                        // dst += low product
            "adcl $0, %edx",                             // %edx += CF
            "movl %edx, {carry}",                        // carry = %edx
            "addl $4, {src}",                            // Advance src (+4)
            "addl $4, {dst}",                            // Advance dst (+4)
            "decl {len}",                                // Decrement len
            "jnz 3b",                                    // Repeat while len != 0

            // Tail completion
            "4:",                                        // Completion label

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
