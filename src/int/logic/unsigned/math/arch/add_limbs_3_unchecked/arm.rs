//! ARM 32-bit (`ARMv6` / ARMv7-A) implementation of `add_limbs_3_unchecked`.
//!
//! Evaluates `dst = src1 + src2` using 4-way unrolled `adcs` chains with post-increment addressing.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Compute `dst[i] = src1[i] + src2[i] + carry` for `len` limbs,
/// returning the final carry.
///
/// # Microarchitectural Strategy
///
/// Small inputs (2..=4 limbs) execute specialized straight-line `adds`/`adcs` chains with zero loop overhead.
/// Larger slices run on a 4-way unrolled `adcs` loop with post-indexed `ldr`/`str` instructions.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for reads and writes of `len` 32-bit limbs.
/// - `dst` must not alias `src1` or `src2`.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    // SAFETY: The caller guarantees both pointers cover `len` elements.
    if len == 0 {
        return 0;
    }
    if len == 1 {
        // SAFETY: The caller guarantees all pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if len <= 4 {
        // SAFETY: Caller guarantees `dst`, `src1`, `src2` valid for `len in 2..=4`.
        return unsafe { add_small_3_unchecked(dst, src1, src2, len) };
    }
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst`, `src1`, `src2` are valid for `len` 32-bit `Limb` elements.
    // 2. Post-increment addresses remain within allocated bounds.
    // 3. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cmp {chunks}, #0",                          // Check if chunks == 0
            "beq 2f",                                    // If chunks == 0, skip to remainder (2f)
            "lsrs {carry}, {carry}, #1",                 // Set C flag from carry (C = carry)
            ".p2align 4",
            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "ldr {s1}, [{src1}], #4",                    // Load src1[0] and advance (+4)
            "ldr {s2}, [{src2}], #4",                    // Load src2[0] and advance (+4)
            "adcs {s1}, {s1}, {s2}",                     // s1 = s1 + s2 + C flag (updates C flag)
            "str {s1}, [{dst}], #4",                     // Store dst[0] and advance (+4)

            // [Limb 1]
            "ldr {s1}, [{src1}], #4",                    // Load src1[1]
            "ldr {s2}, [{src2}], #4",                    // Load src2[1]
            "adcs {s1}, {s1}, {s2}",                     // Add with carry
            "str {s1}, [{dst}], #4",                     // Store dst[1]

            // [Limb 2]
            "ldr {s1}, [{src1}], #4",                    // Load src1[2]
            "ldr {s2}, [{src2}], #4",                    // Load src2[2]
            "adcs {s1}, {s1}, {s2}",                     // Add with carry
            "str {s1}, [{dst}], #4",                     // Store dst[2]

            // [Limb 3]
            "ldr {s1}, [{src1}], #4",                    // Load src1[3]
            "ldr {s2}, [{src2}], #4",                    // Load src2[3]
            "adcs {s1}, {s1}, {s2}",                     // Add with carry
            "str {s1}, [{dst}], #4",                     // Store dst[3]

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
            "ldr {s1}, [{src1}], #4",                    // Load single src1 limb
            "ldr {s2}, [{src2}], #4",                    // Load single src2 limb
            "adcs {s1}, {s1}, {s2}",                     // Add with carry
            "str {s1}, [{dst}], #4",                     // Store single dst limb

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
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            dst = inout(reg) dst => _,
            s1 = out(reg) _,
            s2 = out(reg) _,
            options(nostack)
        );
    }
    carry
}

/// Straight-line `dst[i] = src1[i] + src2[i] + carry` chain for `len` in `2..=4`.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span.
#[allow(
    clippy::inline_always,
    reason = "The fixed-size carry chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn add_small_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 2 limbs.
            unsafe {
                asm!(
                    "ldr {a0}, [{src1}]",                // Load src1[0]
                    "ldr {a1}, [{src1}, #4]",            // Load src1[1]
                    "ldr {b0}, [{src2}]",                // Load src2[0]
                    "ldr {b1}, [{src2}, #4]",            // Load src2[1]
                    "adds {a0}, {a0}, {b0}",             // a0 = src1[0] + src2[0], set C flag
                    "adcs {a1}, {a1}, {b1}",             // a1 = src1[1] + src2[1] + C flag
                    "str {a0}, [{dst}]",                 // Store dst[0]
                    "str {a1}, [{dst}, #4]",             // Store dst[1]
                    "mov {carry}, #0",                   // Clear carry
                    "adc {carry}, {carry}, #0",          // Capture final C flag into carry (0 or 1)
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    a0 = out(reg) _, a1 = out(reg) _,
                    b0 = out(reg) _, b1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        3 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 3 limbs.
            unsafe {
                asm!(
                    "ldr {a0}, [{src1}]",                // Load src1[0]
                    "ldr {a1}, [{src1}, #4]",            // Load src1[1]
                    "ldr {a2}, [{src1}, #8]",            // Load src1[2]
                    "ldr {b0}, [{src2}]",                // Load src2[0]
                    "ldr {b1}, [{src2}, #4]",            // Load src2[1]
                    "ldr {b2}, [{src2}, #8]",            // Load src2[2]
                    "adds {a0}, {a0}, {b0}",             // a0 = src1[0] + src2[0], set C flag
                    "adcs {a1}, {a1}, {b1}",             // a1 = src1[1] + src2[1] + C flag
                    "adcs {a2}, {a2}, {b2}",             // a2 = src1[2] + src2[2] + C flag
                    "str {a0}, [{dst}]",                 // Store dst[0]
                    "str {a1}, [{dst}, #4]",             // Store dst[1]
                    "str {a2}, [{dst}, #8]",             // Store dst[2]
                    "mov {carry}, #0",                   // Clear carry
                    "adc {carry}, {carry}, #0",          // Capture final C flag
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    a0 = out(reg) _, a1 = out(reg) _, a2 = out(reg) _,
                    b0 = out(reg) _, b1 = out(reg) _, b2 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        4 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 4 limbs.
            unsafe {
                asm!(
                    "ldr {a0}, [{src1}]",                // Load src1[0]
                    "ldr {a1}, [{src1}, #4]",            // Load src1[1]
                    "ldr {a2}, [{src1}, #8]",            // Load src1[2]
                    "ldr {a3}, [{src1}, #12]",           // Load src1[3]
                    "ldr {b0}, [{src2}]",                // Load src2[0]
                    "ldr {b1}, [{src2}, #4]",            // Load src2[1]
                    "ldr {b2}, [{src2}, #8]",            // Load src2[2]
                    "ldr {b3}, [{src2}, #12]",           // Load src2[3]
                    "adds {a0}, {a0}, {b0}",             // a0 = src1[0] + src2[0], set C flag
                    "adcs {a1}, {a1}, {b1}",             // a1 = src1[1] + src2[1] + C flag
                    "adcs {a2}, {a2}, {b2}",             // a2 = src1[2] + src2[2] + C flag
                    "adcs {a3}, {a3}, {b3}",             // a3 = src1[3] + src2[3] + C flag
                    "str {a0}, [{dst}]",                 // Store dst[0]
                    "str {a1}, [{dst}, #4]",             // Store dst[1]
                    "str {a2}, [{dst}, #8]",             // Store dst[2]
                    "str {a3}, [{dst}, #12]",            // Store dst[3]
                    "mov {carry}, #0",                   // Clear carry
                    "adc {carry}, {carry}, #0",          // Capture final C flag
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    a0 = out(reg) _, a1 = out(reg) _, a2 = out(reg) _, a3 = out(reg) _,
                    b0 = out(reg) _, b1 = out(reg) _, b2 = out(reg) _, b3 = out(reg) _,
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
