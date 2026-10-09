//! 32-bit ARM (`ARMv6` / ARMv7-A / Cortex-A) fused multiply-add limb kernel.
//!
//! Uses 32x32->64-bit unsigned multiplication (`umull`), carry-propagating additions
//! (`adds`/`adc`), and post-indexed memory addressing (`[src], #4`).

use core::arch::asm;

use super::Limb;

/// Accumulates `src * scalar` into `dst` and returns the high carry limb.
///
/// Four limbs are processed per block, followed by a scalar tail.
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// Nonempty spans must cover `len` aligned, initialized limbs in disjoint live
/// allocations of at most `isize::MAX` bytes. The destination must be writable.
#[expect(
    clippy::inline_always,
    reason = "Keep the unrolled hardware row recurrence in the selected multiplication caller"
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

    // SAFETY: 4*chunks + rem == len bounds both initialized spans. Four-byte
    // post-increments preserve alignment within their live allocations. Stores
    // target only the disjoint writable destination. Empty spans skip all loads.
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
