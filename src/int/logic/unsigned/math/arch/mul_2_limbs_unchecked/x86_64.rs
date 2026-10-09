//! Baseline generic x86-64 write-only 2-by-N limb multiplication kernel.
//!
//! Evaluates `dst = src * (s0 + s1 * B)` in a single write-only pass using
//! standard `mulq`, eliminating memory zeroing during basecase multiplication.

use core::arch::asm;

use super::Limb;

/// Write `src * (s0 + s1 * B)` into `dst` without reading its prior contents.
///
/// Computes:
///
/// ```text
///   dst[0..len+2] = src[0..len] x (s0 + s1 x 2^64)
/// ```
///
/// Computes two simultaneous multiplication rows (`s0 * src` and `s1 * src`)
/// using standard `mulq` ($64 \times 64 \to 128$-bit into `%rdx:%rax`). By keeping both
/// row carry chains (`%r8` for row 0, `%r9` for row 1) in registers, this kernel writes
/// directly into destination memory without requiring prior memory zeroing.
///
/// # Safety
///
/// - For nonzero `len`, aligned `dst` must cover `len + 2` writable limbs,
///   which may be uninitialized; aligned `src` must cover `len` initialized limbs.
/// - `len + 2` must fit `usize`, and both byte spans must fit `isize::MAX`.
/// - Source and destination spans must be disjoint.
#[expect(
    clippy::inline_always,
    reason = "inlining joins the two-row initializer to complete basecase multiplication"
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

    // SAFETY: disjoint aligned spans provide len initialized inputs and len+2
    // writable outputs. Limb zero is consumed before the len-1 remaining
    // iterations; two final stores complete the output. Products plus two limbs
    // are <= B^2-1; the final high carry is <= s1. No destination is read.
    // All modified registers are outputs, and the stack is untouched.
    unsafe {
        asm!(
            // [Initial Step: Limb 0 of src]
            "movq ({src}), %rcx",                        // Load src[0] into %rcx
            "movq {s0}, %rax",                           // %rax = s0
            "mulq %rcx",                                 // %rdx:%rax = src[0] * s0
            "movq %rax, ({dst})",                        // Write low product directly to dst[0]
            "movq %rdx, %r8",                            // %r8 = row 0 carry
            "movq {s1}, %rax",                           // %rax = s1
            "mulq %rcx",                                 // %rdx:%rax = src[0] * s1
            "movq %rax, %rsi",                           // %rsi = prev_s1_lo
            "movq %rdx, %r9",                            // %r9 = row 1 carry
            "leaq 8({src}), {src}",                      // Advance src pointer by 8 bytes
            "leaq 8({dst}), {dst}",                      // Advance dst pointer by 8 bytes
            "decq {len}",                                // Decrement counter
            "jz 2f",                                     // If len was 1, skip to final flush (2f)

            // Main loop processing src[1..len]
            "1:",
            "movq ({src}), %rcx",                        // Load src[i] into %rcx
            "movq {s0}, %rax",                           // %rax = s0
            "leaq 8({src}), {src}",                      // Advance src by 8 bytes
            "mulq %rcx",                                 // %rdx:%rax = src[i] * s0
            "addq %rsi, %rax",                           // Merge the preceding row-one low limb
            "adcq $0, %rdx",                             // %rdx += CF
            "addq %r8, %rax",                            // %rax += row 0 carry
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rax, ({dst})",                        // Store fully resolved limb to dst[i]

            "movq {s1}, %rax",                           // %rax = s1
            "movq %rdx, %r8",                            // %r8 = updated row 0 carry
            "mulq %rcx",                                 // %rdx:%rax = src[i] * s1
            "addq %r9, %rax",                            // %rax += row 1 carry
            "adcq $0, %rdx",                             // %rdx += CF
            "decq {len}",                                // Decrement counter

            "leaq 8({dst}), {dst}",                      // Advance dst by 8 bytes
            "movq %rax, %rsi",                           // %rsi = updated prev_s1_lo
            "movq %rdx, %r9",                            // %r9 = updated row 1 carry
            "jnz 1b",                                    // Repeat while len != 0

            // [Final Carry Flush into dst[len] and dst[len+1]]
            "2:",
            "addq %r8, %rsi",                            // Accumulate remaining row 0 carry into %rsi
            "adcq $0, %r9",                              // Propagate overflow into row 1 carry
            "movq %rsi, ({dst})",                        // Store final dst[len]
            "movq %r9, 8({dst})",                        // Store final dst[len+1]

            len = inout(reg) len => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            out("rsi") _,
            out("r8") _,
            out("r9") _,
            options(nostack, att_syntax)
        );
    }
}
