//! `x86_64` architecture-specific shift kernels.
//!
//! Uses `shldq` for left-shift and `shrdq` for right-shift.
//! These instructions merge bits from two registers in a single operation,
//! replacing the separate shift+or+and pattern the compiler generates.

use core::arch::asm;

use super::Limb;

/// Left-shift `len` limbs in-place by `shift` bits (0 < shift < `LIMB_BITS`).
/// Returns the bits shifted out of the top limb.
///
/// # Safety
///
/// - `limbs` must be valid for reads and writes of `len` elements.
/// - `shift` must satisfy `0 < shift < LIMB_BITS`: the kernel computes
///   `LIMB_BITS - shift` and applies both shift amounts to each element, so
///   an out-of-range amount is undefined behavior.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn lshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    if len == 0 {
        return 0;
    }
    let mut carry_out: Limb;
    // SAFETY: Caller guarantees `limbs` has `len` elements, shift in 1..63
    unsafe {
        asm!(
            // Extract carry: top_limb >> (64 - shift)
            "movq -8({limbs},{len},8), {carry_out}",
            "movl {shift:e}, %ecx",
            "negb %cl",                         // cl = 64-shift (mod 64)
            "shrq %cl, {carry_out}",
            "movl {shift:e}, %ecx",

            "movq {len}, %rdx",
            "subq $1, %rdx",                    // rdx = len - 1
            "jz 2f",                             // skip if len == 1

            "leaq -8({limbs},{len},8), {ptr}",   // ptr = &limbs[len-1]

            "movq %rdx, %r8",
            "shrq $2, %r8",                     // r8 = iterations / 4
            "jz 3f",                            // if < 4, go to scalar tail

            ".p2align 4",
            "4:",
            "movq -8({ptr}), %rax",              // load limbs[i-1]
            "shldq %cl, %rax, ({ptr})",          // merge into limbs[i]
            "movq -16({ptr}), %rax",
            "shldq %cl, %rax, -8({ptr})",
            "movq -24({ptr}), %rax",
            "shldq %cl, %rax, -16({ptr})",
            "movq -32({ptr}), %rax",
            "shldq %cl, %rax, -24({ptr})",
            "leaq -32({ptr}), {ptr}",
            "decq %r8",
            "jnz 4b",

            "3:",
            "andq $3, %rdx",                    // rdx = iterations % 4
            "jz 2f",

            ".p2align 4",
            "1:",
            "movq -8({ptr}), %rax",
            "shldq %cl, %rax, ({ptr})",
            "leaq -8({ptr}), {ptr}",
            "decq %rdx",
            "jnz 1b",

            // Shift bottom limb (no source below)
            "2:",
            "shlq %cl, ({limbs})",

            carry_out = out(reg) carry_out,
            limbs = in(reg) limbs,
            len = in(reg) len,
            shift = in(reg) shift,
            ptr = out(reg) _,
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            out("r8") _,
            options(nostack, att_syntax)
        );
    }
    carry_out
}
