//! ARM implementation of `add_limbs_unchecked`.
//!
//! Evaluates `dst += src` using 4-way unrolled `adcs` chains with post-increment addressing.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Adds `src` to `dst` and returns the binary carry.
///
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// Nonempty spans must be aligned, initialized, and cover `len` limbs in
/// live allocations of at most `isize::MAX` bytes. The destination must be
/// writable. The two spans must be disjoint or exactly identical.
#[expect(
    clippy::inline_always,
    reason = "Keep the hardware carry chain in the selected arithmetic caller"
)]
#[inline(always)]
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
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

    // SAFETY: len > 4 proves chunks > 0; 4 * chunks + rem == len bounds
    // every aligned, initialized access. Each limb's inputs are read before
    // its result is stored, including when the two spans are identical.
    unsafe {
        asm!(
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
/// The pointer contract of [`add_limbs_unchecked`] applies and `2 <= len <= 4`.
#[expect(
    clippy::inline_always,
    reason = "The fixed-size carry chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: len == 2 bounds both aligned, initialized spans.
            // All inputs are loaded before stores, permitting exact aliasing.
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
            // SAFETY: len == 3 bounds both aligned, initialized spans.
            // All inputs are loaded before stores, permitting exact aliasing.
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
            // SAFETY: len == 4 bounds both aligned, initialized spans.
            // All inputs are loaded before stores, permitting exact aliasing.
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
        // SAFETY: the caller establishes 2 <= len <= 4, all matched above.
        _ => unsafe { unreachable_unchecked() },
    }
}
