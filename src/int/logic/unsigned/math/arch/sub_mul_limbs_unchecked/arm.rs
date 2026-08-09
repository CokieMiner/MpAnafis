//! ARM (32-bit) `submul` kernel — pure Rust for native 32-bit limbs.
//!
//! On 32-bit ARM, `Limb = u32` and `DoubleLimb = u64`. Each limb is a single
//! native word, so the multiply-accumulate loop uses `DoubleLimb` for the
//! product and carry, then subtracts the low word from the destination with
//! borrow propagation.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// This computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] * scalar)
/// ```
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
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

    // SAFETY: Caller guarantees `dst` and `src` are valid for `len` elements
    unsafe {
        asm!(
            "cmp {chunks}, #0",
            "beq 2f",
            ".p2align 2",
            "1:",
            // Limb 0
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "umull {p_lo}, {p_hi}, {s}, {scalar}",
            "adds {p_lo}, {p_lo}, {carry}",
            "adc {carry}, {p_hi}, #0",
            "rsbs {borrow}, {borrow}, #0",
            "sbcs {d}, {d}, {p_lo}",
            "str {d}, [{dst}], #4",
            "mov {borrow}, #0",
            "movcc {borrow}, #1",

            // Limb 1
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "umull {p_lo}, {p_hi}, {s}, {scalar}",
            "adds {p_lo}, {p_lo}, {carry}",
            "adc {carry}, {p_hi}, #0",
            "rsbs {borrow}, {borrow}, #0",
            "sbcs {d}, {d}, {p_lo}",
            "str {d}, [{dst}], #4",
            "mov {borrow}, #0",
            "movcc {borrow}, #1",

            // Limb 2
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "umull {p_lo}, {p_hi}, {s}, {scalar}",
            "adds {p_lo}, {p_lo}, {carry}",
            "adc {carry}, {p_hi}, #0",
            "rsbs {borrow}, {borrow}, #0",
            "sbcs {d}, {d}, {p_lo}",
            "str {d}, [{dst}], #4",
            "mov {borrow}, #0",
            "movcc {borrow}, #1",

            // Limb 3
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "umull {p_lo}, {p_hi}, {s}, {scalar}",
            "adds {p_lo}, {p_lo}, {carry}",
            "adc {carry}, {p_hi}, #0",
            "rsbs {borrow}, {borrow}, #0",
            "sbcs {d}, {d}, {p_lo}",
            "str {d}, [{dst}], #4",
            "mov {borrow}, #0",
            "movcc {borrow}, #1",

            "subs {chunks}, {chunks}, #1",
            "bne 1b",

            "2:",
            "cmp {rem}, #0",
            "beq 4f",
            ".p2align 2",
            "3:",
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "umull {p_lo}, {p_hi}, {s}, {scalar}",
            "adds {p_lo}, {p_lo}, {carry}",
            "adc {carry}, {p_hi}, #0",
            "rsbs {borrow}, {borrow}, #0",
            "sbcs {d}, {d}, {p_lo}",
            "str {d}, [{dst}], #4",
            "mov {borrow}, #0",
            "movcc {borrow}, #1",
            "subs {rem}, {rem}, #1",
            "bne 3b",
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
        (carry, borrow)
    }
}
