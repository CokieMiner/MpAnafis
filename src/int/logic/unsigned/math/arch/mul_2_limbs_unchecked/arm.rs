//! ARM 32-bit (`ARMv6` / ARMv7-A) write-only dual-row multiplication kernel using `umaal`.
//!
//! Evaluates `dst = src * (s0 + s1 * B)` in a single write-only pass using
//! the single-cycle hardware 4-operand fused MAC instruction `umaal`.

use core::arch::asm;

use super::Limb;

/// Write `src * (s0 + s1 * B)` into `dst` without reading its old contents.
///
/// Computes:
///
/// ```text
///   dst[0..len+2] = src[0..len] × (s0 + s1 × 2^32)
/// ```
///
/// # Microarchitectural Strategy
///
/// Utilizes `ARMv6`/v7 `umaal` to accumulate products and carries directly into output registers
/// without condition-flag dependencies. Unrolled 2-way for improved instruction pipelining
/// and dual-issue efficiency.
///
/// # Safety
///
/// - `dst` must point to a writable buffer of at least `len + 2` initialized 32-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 32-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical basecase initialization kernel; removing its call boundary matters for small products"
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

    let carry0: Limb = 0;
    let carry1: Limb = 0;
    let pending1: Limb = 0;
    let chunks = len >> 1;

    // SAFETY:
    // 1. `dst` is valid for writes of `len + 2` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Pointer post-increments remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cmp {chunks}, #0",                          // Compare 2-limb chunk counter with 0
            "beq 2f",                                    // If chunks == 0, skip to remainder (2f)

            // Main 2-way unrolled loop body
            "1:",

            // [Limb 0: Fused MAC across both rows]
            "ldr {s}, [{src}], #4",                      // Load src[0] and advance pointer by 4 bytes
            "umaal {pending1}, {carry0}, {s}, {s0}",     // (carry0:pending1) = pending1 + carry0 + (s * s0)
            "str {pending1}, [{dst}], #4",               // Store finalized dst[0] and advance dst pointer
            "mov {pending1}, #0",                        // Clear pending1 register for row 1
            "umaal {pending1}, {carry1}, {s}, {s1}",     // (carry1:pending1) = 0 + carry1 + (s * s1)

            // [Limb 1: Fused MAC across both rows]
            "ldr {s}, [{src}], #4",                      // Load src[1] and advance pointer by 4 bytes
            "umaal {pending1}, {carry0}, {s}, {s0}",     // (carry0:pending1) = pending1 + carry0 + (s * s0)
            "str {pending1}, [{dst}], #4",               // Store finalized dst[1] and advance dst pointer
            "mov {pending1}, #0",                        // Clear pending1 register for row 1
            "umaal {pending1}, {carry1}, {s}, {s1}",     // (carry1:pending1) = 0 + carry1 + (s * s1)

            "subs {chunks}, {chunks}, #1",               // Decrement chunk counter
            "bne 1b",                                    // Repeat while chunks != 0

            // Remainder processing (0 or 1 limb)
            "2:",
            "tst {len}, #1",                             // Test if len is odd
            "beq 3f",                                    // If even, skip remainder (3f)

            // 1-limb tail
            "ldr {s}, [{src}], #4",                      // Load single src limb
            "umaal {pending1}, {carry0}, {s}, {s0}",     // Row 0 fused MAC
            "str {pending1}, [{dst}], #4",               // Store finalized dst limb
            "mov {pending1}, #0",                        // Clear pending1
            "umaal {pending1}, {carry1}, {s}, {s1}",     // Row 1 fused MAC

            // Epilogue: Flush trailing high row 1 limb + remaining carry
            "3:",
            "adds {pending1}, {pending1}, {carry0}",     // pending1 += carry0, set C flag
            "adc {carry1}, {carry1}, #0",                // carry1 += C flag + 0
            "str {pending1}, [{dst}], #4",               // Store dst[len]
            "str {carry1}, [{dst}]",                     // Store final high limb dst[len+1]

            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = in(reg) len,
            chunks = inout(reg) chunks => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            carry0 = inout(reg) carry0 => _,
            carry1 = inout(reg) carry1 => _,
            pending1 = inout(reg) pending1 => _,
            s = out(reg) _,
            options(nostack)
        );
    }
}
