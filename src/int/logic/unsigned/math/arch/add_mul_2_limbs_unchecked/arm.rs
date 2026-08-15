//! ARM 32-bit (`ARMv6` / ARMv7-A) fused dual-row multiply-add kernel using `umaal`.
//!
//! Utilizes the single-cycle hardware 4-operand fused MAC instruction `umaal`
//! to compute `(c:dst) = dst + c + (src * scalar)` directly in registers.

use core::arch::asm;

use super::Limb;

/// Fused dual-row multiply-add kernel for ARM 32-bit targets.
///
/// Computes:
///
/// ```text
///   dst[0..len] += src[0..len] * s0 + c0
///   dst[1..len+1] += src[0..len] * s1 + c1
/// ```
///
/// # Microarchitectural Strategy
///
/// `ARMv6`+ architectures feature `umaal` (Unsigned Multiply Accumulate Accumulate Long),
/// which computes $(R_{hi}:R_{lo}) = R_{lo} + R_{hi} + (`R_n` \times `R_m`)$ in a single execution cycle.
/// This kernel evaluates both multiplication rows (`s0` and `s1`) in lockstep using `umaal`,
/// eliminating separate carry addition and propagation instructions.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len + 1` initialized 32-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 32-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in 32-bit ARM dual-row multiplication"
)]
#[inline(always)]
pub unsafe fn add_mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) -> (Limb, Limb) {
    let mut c0: Limb = 0;
    let mut c1: Limb = 0;

    if len == 0 {
        return (0, 0);
    }

    let chunks = len >> 1;
    let rem = len & 1;

    // SAFETY:
    // 1. `dst` is valid for reads and writes of `len + 1` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Pointer post-increments remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cmp {chunks}, #0",                          // Compare 2-limb chunk counter with 0
            "beq 2f",                                    // If chunks == 0, skip to remainder loop (2f)
            ".p2align 2",

            // Main 2-way unrolled loop body
            "1:",

            // [Limb 0: Fused 4-operand MAC across both rows]
            "ldr {s}, [{src}], #4",                      // Load src[0] and advance pointer by 4 bytes
            "ldr {d0}, [{dst}]",                         // Load dst[0]
            "ldr {d1}, [{dst}, #4]",                     // Load dst[1]
            "umaal {d0}, {c0}, {s}, {s0}",               // (c0:d0) = d0 + c0 + (s * s0)
            "umaal {d1}, {c1}, {s}, {s1}",               // (c1:d1) = d1 + c1 + (s * s1)
            "str {d0}, [{dst}], #4",                     // Store finalized dst[0] and advance dst pointer
            "str {d1}, [{dst}]",                         // Store intermediate dst[1]

            // [Limb 1: Fused 4-operand MAC across both rows]
            "ldr {s}, [{src}], #4",                      // Load src[1]
            "ldr {d0}, [{dst}]",                         // Load dst[1] (updated in previous step)
            "ldr {d1}, [{dst}, #4]",                     // Load dst[2]
            "umaal {d0}, {c0}, {s}, {s0}",               // (c0:d0) = d0 + c0 + (s * s0)
            "umaal {d1}, {c1}, {s}, {s1}",               // (c1:d1) = d1 + c1 + (s * s1)
            "str {d0}, [{dst}], #4",                     // Store finalized dst[1] and advance pointer
            "str {d1}, [{dst}]",                         // Store intermediate dst[2]

            "subs {chunks}, {chunks}, #1",               // Decrement chunk counter
            "bne 1b",                                    // Repeat loop while chunks != 0

            // Remainder processing (0 or 1 limb)
            "2:",
            "cmp {rem}, #0",                             // Compare remainder count with 0
            "beq 4f",                                    // If rem == 0, skip to end (4f)

            // 1-limb tail
            "3:",
            "ldr {s}, [{src}], #4",                      // Load single src limb
            "ldr {d0}, [{dst}]",                         // Load single dst limb
            "ldr {d1}, [{dst}, #4]",                     // Load next dst limb
            "umaal {d0}, {c0}, {s}, {s0}",               // Row 0 fused MAC
            "umaal {d1}, {c1}, {s}, {s1}",               // Row 1 fused MAC
            "str {d0}, [{dst}], #4",                     // Store dst limb
            "str {d1}, [{dst}]",                         // Store final dst limb

            // Tail completion
            "4:",

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            s = out(reg) _,
            d0 = out(reg) _,
            d1 = out(reg) _,
            options(nostack)
        );
    }
    (c0, c1)
}
