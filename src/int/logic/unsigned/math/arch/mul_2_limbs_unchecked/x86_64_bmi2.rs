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
///   dst[0..len+2] = src[0..len] × (s0 + s1 × 2^64)
/// ```
///
/// # Microarchitectural Strategy
///
/// Computes two simultaneous multiplication rows (`s0 * src` and `s1 * src`)
/// in registers using `mulxq`. By keeping both row carry chains (`%r8` for row 0, `%r9` for row 1)
/// in registers, this kernel writes directly into destination memory without needing
/// a separate zeroing/allocation step.
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
            "movq ({src}), %rdx",                        // Load src[0] into %rdx for mulxq
            "mulxq {s0}, %r10, %r11",                    // %r11:%r10 = src[0] * s0
            "mulxq {s1}, %rax, %rcx",                    // %rcx:%rax = src[0] * s1
            "movq %r10, ({dst})",                        // Write low product directly to dst[0]
            "movq %r11, %r8",                            // %r8 = row 0 running carry
            "movq %rax, 8({dst})",                       // Write (src[0]*s1).lo directly to dst[1]
            "movq %rcx, %r9",                            // %r9 = row 1 running carry
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
            "addq %r8, %r10",                            // %r10 += row 0 running carry
            "adcq $0, %r11",                             // %r11 += CF (high product carry)
            "addq ({dst}), %r10",                        // %r10 += dst[i] (accumulate previous s1 row)
            "adcq $0, %r11",                             // %r11 += CF
            "movq %r10, ({dst})",                        // Store combined limb to dst[i]
            "movq %r11, %r8",                            // %r8 = updated row 0 carry

            // [Row 1 Carry Accumulation]
            "addq %r9, %rax",                            // %rax += row 1 running carry
            "adcq $0, %rcx",                             // %rcx += CF
            "movq %rax, 8({dst})",                       // Store row 1 limb to dst[i+1]
            "movq %rcx, %r9",                            // %r9 = updated row 1 carry

            "leaq 8({src}), {src}",                      // Advance src by 8 bytes
            "leaq 8({dst}), {dst}",                      // Advance dst by 8 bytes
            "decq {len}",                                // Decrement counter
            "jnz 1b",                                    // Repeat while len != 0

            // [Final Carry Flush into dst[len] and dst[len+1]]
            "2:",
            "movq ({dst}), %r10",                        // Load dst[len]
            "addq %r8, %r10",                            // Accumulate remaining row 0 carry
            "movq %r10, ({dst})",                        // Store final dst[len]
            "adcq $0, %r9",                              // Propagate overflow into row 1 carry
            "movq %r9, 8({dst})",                        // Store final dst[len+1]

            len = inout(reg) len => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
}
