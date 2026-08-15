//! ARM (32-bit ARMv7-A / Cortex-A) fused multiply-subtract limb kernel.
//!
//! Uses 32×32→64-bit unsigned multipliers (`umull`), carry-chain addition (`adds`/`adc`),
//! reverse-subtraction borrow synthesis (`rsbs`), and condition-code subtraction (`sbcs`).

use core::arch::asm;

use super::Limb;

/// Multiply `len` 32-bit limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// Computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// 32-bit ARM provides hardware `umull` for 64-bit product generation and post-indexed addressing
/// modes (`[src], #4`) for zero-overhead pointer increments. The multiplication carry is accumulated
/// via `adds`/`adc`, while the subtraction borrow is converted into ARM carry-flag semantics via `rsbs`
/// (Reverse Subtract from Zero) and applied directly via `sbcs`. The main loop is 4-way unrolled.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 32-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 32-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in 32-bit ARM multi-precision hot paths"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry: Limb = 0;
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Pointer post-increments remain within `len * 4` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cmp {chunks}, #0",                          // Compare 4-limb chunk counter with 0
            "beq 2f",                                    // If chunks == 0, skip to remainder handler (2f)
            ".p2align 2",

            // Main 4-way unrolled loop body
            "1:",

            // [Limb 0 Multiply-Subtract]
            "ldr {s}, [{src}], #4",                      // Load src[0] and advance pointer by 4 bytes
            "ldr {d}, [{dst}]",                          // Load dst[0]
            "umull {p_lo}, {p_hi}, {s}, {scalar}",       // p_hi:p_lo = src[0] * scalar (64-bit product)
            "adds {p_lo}, {p_lo}, {carry}",              // p_lo += carry, set C flag
            "adc {carry}, {p_hi}, #0",                   // carry = p_hi + C flag
            "rsbs {borrow}, {borrow}, #0",               // C = 1 - borrow (convert borrow into ARM carry)
            "sbcs {d}, {d}, {p_lo}",                     // dst[0] = dst[0] - p_lo - borrow, update C
            "str {d}, [{dst}], #4",                      // Store updated dst[0] and advance dst pointer
            "mov {borrow}, #0",                          // Default borrow = 0
            "movcc {borrow}, #1",                        // If C==0 (Carry Clear), borrow = 1

            // [Limb 1 Multiply-Subtract]
            "ldr {s}, [{src}], #4",                      // Load src[1]
            "ldr {d}, [{dst}]",                          // Load dst[1]
            "umull {p_lo}, {p_hi}, {s}, {scalar}",       // src[1] * scalar
            "adds {p_lo}, {p_lo}, {carry}",              // p_lo += carry
            "adc {carry}, {p_hi}, #0",                   // carry = p_hi + C
            "rsbs {borrow}, {borrow}, #0",               // Restore borrow to C
            "sbcs {d}, {d}, {p_lo}",                     // dst[1] -= p_lo + borrow
            "str {d}, [{dst}], #4",                      // Store dst[1]
            "mov {borrow}, #0",                          // Reset borrow
            "movcc {borrow}, #1",                        // Capture new borrow

            // [Limb 2 Multiply-Subtract]
            "ldr {s}, [{src}], #4",                      // Load src[2]
            "ldr {d}, [{dst}]",                          // Load dst[2]
            "umull {p_lo}, {p_hi}, {s}, {scalar}",       // src[2] * scalar
            "adds {p_lo}, {p_lo}, {carry}",              // p_lo += carry
            "adc {carry}, {p_hi}, #0",                   // carry = p_hi + C
            "rsbs {borrow}, {borrow}, #0",               // Restore borrow to C
            "sbcs {d}, {d}, {p_lo}",                     // dst[2] -= p_lo + borrow
            "str {d}, [{dst}], #4",                      // Store dst[2]
            "mov {borrow}, #0",                          // Reset borrow
            "movcc {borrow}, #1",                        // Capture new borrow

            // [Limb 3 Multiply-Subtract]
            "ldr {s}, [{src}], #4",                      // Load src[3]
            "ldr {d}, [{dst}]",                          // Load dst[3]
            "umull {p_lo}, {p_hi}, {s}, {scalar}",       // src[3] * scalar
            "adds {p_lo}, {p_lo}, {carry}",              // p_lo += carry
            "adc {carry}, {p_hi}, #0",                   // carry = p_hi + C
            "rsbs {borrow}, {borrow}, #0",               // Restore borrow to C
            "sbcs {d}, {d}, {p_lo}",                     // dst[3] -= p_lo + borrow
            "str {d}, [{dst}], #4",                      // Store dst[3]
            "mov {borrow}, #0",                          // Reset borrow
            "movcc {borrow}, #1",                        // Capture new borrow

            "subs {chunks}, {chunks}, #1",               // Decrement chunk counter
            "bne 1b",                                    // Repeat loop while chunks != 0

            // Remainder processing (0 to 3 limbs)
            "2:",
            "cmp {rem}, #0",                             // Compare remainder with 0
            "beq 4f",                                    // If rem == 0, skip to end (4f)
            ".p2align 2",

            // 1-limb tail loop
            "3:",
            "ldr {s}, [{src}], #4",                      // Load single src limb
            "ldr {d}, [{dst}]",                          // Load single dst limb
            "umull {p_lo}, {p_hi}, {s}, {scalar}",       // 32x32->64 product
            "adds {p_lo}, {p_lo}, {carry}",              // Add carry
            "adc {carry}, {p_hi}, #0",                   // Propagate carry
            "rsbs {borrow}, {borrow}, #0",               // Convert borrow to C
            "sbcs {d}, {d}, {p_lo}",                     // Subtract product + borrow
            "str {d}, [{dst}], #4",                      // Store updated limb
            "mov {borrow}, #0",                          // Reset borrow
            "movcc {borrow}, #1",                        // Capture new borrow
            "subs {rem}, {rem}, #1",                     // Decrement remainder
            "bne 3b",                                    // Repeat while rem != 0

            // Tail completion
            "4:",

            carry = inout(reg) carry,
            borrow = inout(reg) borrow,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            scalar = in(reg) scalar,
            s = out(reg) _,
            d = out(reg) _,
            p_lo = out(reg) _,
            p_hi = out(reg) _,
            options(nostack)
        );
    }
    (carry, borrow)
}
