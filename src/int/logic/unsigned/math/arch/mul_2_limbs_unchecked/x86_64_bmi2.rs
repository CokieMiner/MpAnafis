//! BMI2 write-only 2-by-N limb multiplication kernel for `x86_64`.
//!
//! Evaluates `dst = src * (s0 + s1 * B)` in a single write-only pass,
//! eliminating memory zeroing and initialization passes during basecase multiplication.

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
/// in registers using `mulxq`. By keeping both row carry chains (`%r8` for row 0, `%r9` for row 1)
/// in registers, this kernel writes directly into destination memory without needing
/// a separate zeroing step.
///
/// # Safety
///
/// - For nonzero `len`, aligned `dst` must cover `len + 2` writable limbs,
///   which may be uninitialized; aligned `src` must cover `len` initialized limbs.
/// - `len + 2` must fit `usize`, and both byte spans must fit `isize::MAX`.
/// - Source and destination spans must be disjoint.
/// - The CPU must support BMI2.
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
    // writable outputs; the caller also guarantees BMI2. Limb zero precedes
    // the len-1 remaining iterations, then two stores complete the output.
    // Products plus two limbs are <= B^2-1; the final high carry is <= s1.
    // No destination is read. All modified registers are declared as outputs.
    unsafe {
        asm!(
            // [Initial Step: Limb 0 of src]
            "movq ({src}), %rdx",                        // Load src[0] into %rdx for mulxq
            "mulxq {s0}, %r10, %r8",                     // %r8:%r10 = src[0] * s0 (%r8 = row 0 carry)
            "mulxq {s1}, %rsi, %r9",                     // %r9:%rsi = src[0] * s1 (%r9 = row 1 carry, %rsi = prev_s1_lo)
            "movq %r10, ({dst})",                        // Write low product directly to dst[0]
            "leaq 8({src}), {src}",                      // Advance src pointer by 8 bytes
            "leaq 8({dst}), {dst}",                      // Advance dst pointer by 8 bytes
            "decq {len}",                                // Decrement limb counter
            "jz 2f",                                     // If len was 1, jump directly to final carry flush (2f)

            // Main loop processing src[1..len]
            "1:",
            "movq ({src}), %rdx",                        // Load src[i] into %rdx
            "mulxq {s0}, %r10, %r11",                    // %r11:%r10 = src[i] * s0
            "mulxq {s1}, %rax, %rcx",                    // %rcx:%rax = src[i] * s1

            // [Row 0 Carry & Destination Accumulation]
            "addq %rsi, %r10",                           // Merge the preceding row-one low limb
            "adcq $0, %r11",                             // %r11 += CF
            "addq %r8, %r10",                            // %r10 += row 0 running carry
            "adcq $0, %r11",                             // %r11 += CF
            "movq %r10, ({dst})",                        // Store fully resolved limb to dst[i]
            "movq %r11, %r8",                            // %r8 = updated row 0 carry

            // [Row 1 Carry Accumulation]
            "addq %r9, %rax",                            // %rax += row 1 running carry
            "adcq $0, %rcx",                             // %rcx += CF
            "movq %rax, %rsi",                           // %rsi = updated prev_s1_lo for next iteration
            "movq %rcx, %r9",                            // %r9 = updated row 1 carry

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
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
}
