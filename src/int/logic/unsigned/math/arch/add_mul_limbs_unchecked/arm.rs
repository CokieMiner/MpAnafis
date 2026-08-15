//! 32-bit ARM (`ARMv6` / ARMv7-A / Cortex-A) fused multiply-add limb kernel.
//!
//! Uses 32×32→64-bit unsigned multiplication (`umull`), carry-propagating additions
//! (`adds`/`adc`), and post-indexed memory addressing (`[src], #4`).

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
/// Uses ARM's `umull` instruction which performs a 32×32→64-bit unsigned product in 1–2 cycles
/// across dual destination registers (`{p_lo}`, `{p_hi}`). The incoming running carry is added
/// with `adds`, and the destination accumulator is updated with post-indexed auto-increment
/// addressing `str {d}, [{dst}], #4`. The loop is unrolled 4-way for Cortex-A8/A9/A15 pipeline efficiency.
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
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Pointer auto-increments (`#4`) remain within `len * 4` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cmp {chunks}, #0",                          // Check if chunks == 0
            "beq 2f",                                    // If chunks == 0, skip to remainder (2f)
            ".p2align 2",                                // Align loop header

            // Main 4-way unrolled loop body
            "1:",                                        // Loop head label
            // [Limb 0]
            "ldr {s}, [{src}], #4",                      // Load src[0] and advance src pointer (+4)
            "ldr {d}, [{dst}]",                          // Load dst[0]
            "umull {p_lo}, {p_hi}, {s}, {scalar}",       // {p_hi}:{p_lo} = src[0] * scalar
            "adds {p_lo}, {p_lo}, {carry}",              // Add incoming carry to low product
            "adc {p_hi}, {p_hi}, #0",                    // Propagate carry bit into high product
            "adds {d}, {d}, {p_lo}",                     // Accumulate low product into destination limb
            "str {d}, [{dst}], #4",                      // Store updated limb to dst[0] and advance (+4)
            "adc {carry}, {p_hi}, #0",                   // carry = p_hi + C flag

            // [Limb 1]
            "ldr {s}, [{src}], #4",                      // Load src[1]
            "ldr {d}, [{dst}]",                          // Load dst[1]
            "umull {p_lo}, {p_hi}, {s}, {scalar}",       // 64-bit product
            "adds {p_lo}, {p_lo}, {carry}",              // Add incoming carry
            "adc {p_hi}, {p_hi}, #0",                    // Propagate carry bit
            "adds {d}, {d}, {p_lo}",                     // Accumulate into destination limb
            "str {d}, [{dst}], #4",                      // Store updated limb
            "adc {carry}, {p_hi}, #0",                   // Update running carry

            // [Limb 2]
            "ldr {s}, [{src}], #4",                      // Load src[2]
            "ldr {d}, [{dst}]",                          // Load dst[2]
            "umull {p_lo}, {p_hi}, {s}, {scalar}",       // 64-bit product
            "adds {p_lo}, {p_lo}, {carry}",              // Add incoming carry
            "adc {p_hi}, {p_hi}, #0",                    // Propagate carry bit
            "adds {d}, {d}, {p_lo}",                     // Accumulate into destination limb
            "str {d}, [{dst}], #4",                      // Store updated limb
            "adc {carry}, {p_hi}, #0",                   // Update running carry

            // [Limb 3]
            "ldr {s}, [{src}], #4",                      // Load src[3]
            "ldr {d}, [{dst}]",                          // Load dst[3]
            "umull {p_lo}, {p_hi}, {s}, {scalar}",       // 64-bit product
            "adds {p_lo}, {p_lo}, {carry}",              // Add incoming carry
            "adc {p_hi}, {p_hi}, #0",                    // Propagate carry bit
            "adds {d}, {d}, {p_lo}",                     // Accumulate into destination limb
            "str {d}, [{dst}], #4",                      // Store updated limb
            "adc {carry}, {p_hi}, #0",                   // Update running carry

            "subs {chunks}, {chunks}, #1",               // Decrement chunk counter
            "bne 1b",                                    // Repeat while chunks != 0

            // Remainder processing entry point (0 to 3 limbs)
            "2:",                                        // Remainder entry label
            "cmp {rem}, #0",                             // Check if rem == 0
            "beq 4f",                                    // If rem == 0, exit (4f)
            ".p2align 2",                                // Align remainder loop header

            // 1-limb unrolled tail loop
            "3:",                                        // Tail loop label
            "ldr {s}, [{src}], #4",                      // Load single src limb
            "ldr {d}, [{dst}]",                          // Load single dst limb
            "umull {p_lo}, {p_hi}, {s}, {scalar}",       // 64-bit product
            "adds {p_lo}, {p_lo}, {carry}",              // Add incoming carry
            "adc {p_hi}, {p_hi}, #0",                    // Propagate carry bit
            "adds {d}, {d}, {p_lo}",                     // Accumulate into destination limb
            "str {d}, [{dst}], #4",                      // Store updated limb
            "adc {carry}, {p_hi}, #0",                   // Update running carry
            "subs {rem}, {rem}, #1",                     // Decrement remainder counter
            "bne 3b",                                    // Repeat while rem != 0

            // Tail completion
            "4:",                                        // Completion label

            carry = inout(reg) carry,
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
    carry
}
