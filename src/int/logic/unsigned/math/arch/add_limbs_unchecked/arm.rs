//! ARM implementation of `add_limbs_unchecked`.
//!
//! Evaluates `dst += src` using 4-way unrolled `adcs` chains with post-increment addressing.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Add `len` limbs from `src` into `dst` with carry propagation.
///
/// Returns the final carry-out limb (0 or 1).
///
/// Computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + src[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// Small inputs (2..=4 limbs) execute straight-line `adds`/`adcs` chains without branch overhead.
/// Larger slices run on a 4-way unrolled `adcs` loop with post-indexed `ldr`/`str` instructions.
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
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    // SAFETY: The caller guarantees both pointers cover `len` elements.
    if len == 0 {
        return 0;
    }
    if len == 1 {
        // SAFETY: The caller guarantees both pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*dst).overflowing_add(*src) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if len <= 4 {
        // SAFETY: Caller guarantees `dst` and `src` are valid for `len in 2..=4`.
        return unsafe { add_small_unchecked(dst, src, len) };
    }
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` and `src` are valid for `len` 32-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "cmp {chunks}, #0",                          // Check if chunks == 0
            "beq 2f",                                    // If chunks == 0, skip to remainder (2f)
            "lsrs {carry}, {carry}, #1",                 // Set C flag from carry (C = carry)
            ".p2align 4",

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ldr {s}, [{src}], #4",                      // Load src[0] and advance (+4)
            "ldr {d}, [{dst}]",                          // Load dst[0]
            "adcs {d}, {d}, {s}",                        // d = dst[0] + src[0] + C flag (updates C flag)
            "str {d}, [{dst}], #4",                      // Store updated dst[0] and advance (+4)

            // [Limb 1]
            "ldr {s}, [{src}], #4",                      // Load src[1]
            "ldr {d}, [{dst}]",                          // Load dst[1]
            "adcs {d}, {d}, {s}",                        // Add with carry
            "str {d}, [{dst}], #4",                      // Store dst[1]

            // [Limb 2]
            "ldr {s}, [{src}], #4",                      // Load src[2]
            "ldr {d}, [{dst}]",                          // Load dst[2]
            "adcs {d}, {d}, {s}",                        // Add with carry
            "str {d}, [{dst}], #4",                      // Store dst[2]

            // [Limb 3]
            "ldr {s}, [{src}], #4",                      // Load src[3]
            "ldr {d}, [{dst}]",                          // Load dst[3]
            "adcs {d}, {d}, {s}",                        // Add with carry
            "str {d}, [{dst}], #4",                      // Store dst[3]

            // Loop iteration check preserving C flag across branch
            "mov {carry}, #0",                           // carry = 0
            "adc {carry}, {carry}, #0",                  // carry = C flag (0 or 1)
            "subs {chunks}, {chunks}, #1",               // Decrement chunk counter
            "beq 2f",                                    // If chunks == 0, proceed to remainder
            "lsrs {carry}, {carry}, #1",                 // Restore C flag from carry
            "b 1b",                                      // Repeat loop

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "cmp {rem}, #0",                             // Check if rem == 0
            "beq 4f",                                    // If rem == 0, exit (4f)
            "lsrs {carry}, {carry}, #1",                 // Restore C flag
            ".p2align 4",

            // 1-limb tail loop
            "3:",
            "ldr {s}, [{src}], #4",                      // Load single src limb
            "ldr {d}, [{dst}]",                          // Load single dst limb
            "adcs {d}, {d}, {s}",                        // Add with carry
            "str {d}, [{dst}], #4",                      // Store single dst limb

            "mov {carry}, #0",                           // carry = 0
            "adc {carry}, {carry}, #0",                  // carry = C flag
            "subs {rem}, {rem}, #1",                     // Decrement remainder
            "beq 4f",                                    // If rem == 0, exit
            "lsrs {carry}, {carry}, #1",                 // Restore C flag
            "b 3b",

            // Exit
            "4:",

            carry = inout(reg) carry,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            s = out(reg) _,
            d = out(reg) _,
            options(nostack)
        );
        carry
    }
}

/// Straight-line `dst[i] = dst[i] + src[i] + carry` chain for `len` in `2..=4`.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint.
#[allow(
    clippy::inline_always,
    reason = "The fixed-size carry chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst` and `src` are valid for 2 limbs.
            unsafe {
                asm!(
                    "ldr {s0}, [{src}]",                 // Load src[0]
                    "ldr {s1}, [{src}, #4]",             // Load src[1]
                    "ldr {d0}, [{dst}]",                 // Load dst[0]
                    "ldr {d1}, [{dst}, #4]",             // Load dst[1]
                    "adds {d0}, {d0}, {s0}",             // d0 = dst[0] + src[0], set C flag
                    "adcs {d1}, {d1}, {s1}",             // d1 = dst[1] + src[1] + C flag
                    "str {d0}, [{dst}]",                 // Store updated dst[0]
                    "str {d1}, [{dst}, #4]",             // Store updated dst[1]
                    "mov {carry}, #0",                   // Clear carry
                    "adc {carry}, {carry}, #0",          // Capture final C flag into carry (0 or 1)
                    src = in(reg) src,
                    dst = in(reg) dst,
                    s0 = out(reg) _, s1 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        3 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst` and `src` are valid for 3 limbs.
            unsafe {
                asm!(
                    "ldr {s0}, [{src}]",                 // Load src[0]
                    "ldr {s1}, [{src}, #4]",             // Load src[1]
                    "ldr {s2}, [{src}, #8]",             // Load src[2]
                    "ldr {d0}, [{dst}]",                 // Load dst[0]
                    "ldr {d1}, [{dst}, #4]",             // Load dst[1]
                    "ldr {d2}, [{dst}, #8]",             // Load dst[2]
                    "adds {d0}, {d0}, {s0}",             // d0 = dst[0] + src[0], set C flag
                    "adcs {d1}, {d1}, {s1}",             // d1 = dst[1] + src[1] + C flag
                    "adcs {d2}, {d2}, {s2}",             // d2 = dst[2] + src[2] + C flag
                    "str {d0}, [{dst}]",                 // Store updated dst[0]
                    "str {d1}, [{dst}, #4]",             // Store updated dst[1]
                    "str {d2}, [{dst}, #8]",             // Store updated dst[2]
                    "mov {carry}, #0",                   // Clear carry
                    "adc {carry}, {carry}, #0",          // Capture final C flag
                    src = in(reg) src,
                    dst = in(reg) dst,
                    s0 = out(reg) _, s1 = out(reg) _, s2 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _, d2 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        4 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst` and `src` are valid for 4 limbs.
            unsafe {
                asm!(
                    "ldr {s0}, [{src}]",                 // Load src[0]
                    "ldr {s1}, [{src}, #4]",             // Load src[1]
                    "ldr {s2}, [{src}, #8]",             // Load src[2]
                    "ldr {s3}, [{src}, #12]",            // Load src[3]
                    "ldr {d0}, [{dst}]",                 // Load dst[0]
                    "ldr {d1}, [{dst}, #4]",             // Load dst[1]
                    "ldr {d2}, [{dst}, #8]",             // Load dst[2]
                    "ldr {d3}, [{dst}, #12]",            // Load dst[3]
                    "adds {d0}, {d0}, {s0}",             // d0 = dst[0] + src[0], set C flag
                    "adcs {d1}, {d1}, {s1}",             // d1 = dst[1] + src[1] + C flag
                    "adcs {d2}, {d2}, {s2}",             // d2 = dst[2] + src[2] + C flag
                    "adcs {d3}, {d3}, {s3}",             // d3 = dst[3] + src[3] + C flag
                    "str {d0}, [{dst}]",                 // Store updated dst[0]
                    "str {d1}, [{dst}, #4]",             // Store updated dst[1]
                    "str {d2}, [{dst}, #8]",             // Store updated dst[2]
                    "str {d3}, [{dst}, #12]",            // Store updated dst[3]
                    "mov {carry}, #0",                   // Clear carry
                    "adc {carry}, {carry}, #0",          // Capture final C flag
                    src = in(reg) src,
                    dst = in(reg) dst,
                    s0 = out(reg) _, s1 = out(reg) _, s2 = out(reg) _, s3 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _, d2 = out(reg) _, d3 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        // SAFETY: Caller guarantees `len in 2..=4`.
        _ => unsafe { unreachable_unchecked() },
    }
}
