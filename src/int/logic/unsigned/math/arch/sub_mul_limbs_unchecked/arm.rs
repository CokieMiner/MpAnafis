//! ARM (32-bit ARMv7-A / Cortex-A) fused multiply-subtract limb kernel.
//!
//! Uses 32x32->64-bit unsigned multipliers (`umull`), carry-chain addition (`adds`/`adc`),
//! reverse-subtraction borrow synthesis (`rsbs`), and condition-code subtraction (`sbcs`).

use core::arch::asm;

use super::Limb;

/// Multiply `len` 32-bit limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// For B = 2^32, `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
/// Four-limb blocks restore C = 1 - borrow with `rsbs` before subtraction.
///
/// # Safety
///
/// Both pointers must cover `len` aligned, initialized limbs in disjoint spans;
/// `dst` requires exclusive access. Each span's byte length must fit in `isize`.
/// Zero length permits null pointers.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "short and unrolled assembly paths share one call boundary in the arithmetic hot path"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    if len < 4 {
        if len == 0 {
            return (0, 0);
        }
        let carry: Limb;
        let borrow: Limb;
        // SAFETY: 1 <= len <= 3 bounds all accesses to the caller's disjoint
        // aligned initialized spans. Post-increments advance by one 32-bit limb. The
        // first product has no incoming carry; rsbs restores C = 1 - borrow
        // before every subsequent subtraction.
        unsafe {
            asm!(
                "ldr {s}, [{src}], #4",
                "umull {lo}, {carry}, {s}, {scalar}",
                "ldr {d}, [{dst}]",
                "mov {borrow}, #0",
                "subs {d}, {d}, {lo}",
                "movcc {borrow}, #1",
                "str {d}, [{dst}], #4",
                "subs {count}, {count}, #1",
                "beq 3f",
                "2:",
                "ldr {s}, [{src}], #4",
                "umull {lo}, {hi}, {s}, {scalar}",
                "ldr {d}, [{dst}]",
                "adds {lo}, {lo}, {carry}",
                "adc {carry}, {hi}, #0",
                "rsbs {borrow}, {borrow}, #0",
                "sbcs {d}, {d}, {lo}",
                "mov {borrow}, #0",
                "movcc {borrow}, #1",
                "str {d}, [{dst}], #4",
                "subs {count}, {count}, #1",
                "bne 2b",
                "3:",
                src = inout(reg) src => _,
                dst = inout(reg) dst => _,
                count = inout(reg) len => _,
                scalar = in(reg) scalar,
                carry = out(reg) carry,
                borrow = out(reg) borrow,
                s = out(reg) _,
                d = out(reg) _,
                lo = out(reg) _,
                hi = out(reg) _,
                options(nostack),
            );
        }
        return (carry, borrow);
    }
    let mut carry: Limb = 0;
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: len >= 4 gives at least one complete four-limb block, followed
    // by len % 4 limbs, within the aligned initialized disjoint spans. Each
    // post-increment advances one limb. Product plus carry is below B^2;
    // ADC consumes that carry before RSBS restores C = 1 - borrow for SBCS.
    // Early outputs preserve all live inputs and asm declares flag modification.
    unsafe {
        asm!(
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
