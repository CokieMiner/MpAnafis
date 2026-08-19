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
///   dst[0..len+2] = src[0..len] × (s0 + s1 × 2^64)
/// ```
///
/// # Microarchitectural Strategy
///
/// Computes two simultaneous multiplication rows (`s0 * src` and `s1 * src`)
/// using standard `mulq` ($64 \times 64 \to 128$-bit into `%rdx:%rax`). By keeping both
/// row carry chains (`%r8` for row 0, `%r9` for row 1) in registers, this kernel writes
/// directly into destination memory without requiring prior memory zeroing.
///
/// # Safety
///
/// - `dst` must point to a writable buffer of at least `len + 2` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of the `src` slice.
#[allow(
    clippy::inline_always,
    reason = "Critical basecase initialization kernel; inlined directly into basecase multiplication loops"
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

    // SAFETY:
    // 1. `dst` is valid for writes of `len + 2` `Limb` elements.
    // 2. `src` is valid for reads of `len` `Limb` elements.
    // 3. Pointer offsets (`0`, `8`) remain within `(len + 2) * 8` bytes.
    // 4. Memory spans are non-overlapping.
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
            "mulq %rcx",                                 // %rdx:%rax = src[i] * s0
            "addq %rsi, %rax",                           // %rax += prev_s1_lo (pure register, no reload!)
            "adcq $0, %rdx",                             // %rdx += CF
            "addq %r8, %rax",                            // %rax += row 0 carry
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rax, ({dst})",                        // Store fully resolved limb to dst[i]
            "movq %rdx, %r8",                            // %r8 = updated row 0 carry

            "movq {s1}, %rax",                           // %rax = s1
            "mulq %rcx",                                 // %rdx:%rax = src[i] * s1
            "addq %r9, %rax",                            // %rax += row 1 carry
            "adcq $0, %rdx",                             // %rdx += CF
            "movq %rax, %rsi",                           // %rsi = updated prev_s1_lo
            "movq %rdx, %r9",                            // %r9 = updated row 1 carry

            "leaq 8({src}), {src}",                      // Advance src by 8 bytes
            "leaq 8({dst}), {dst}",                      // Advance dst by 8 bytes
            "decq {len}",                                // Decrement counter
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
