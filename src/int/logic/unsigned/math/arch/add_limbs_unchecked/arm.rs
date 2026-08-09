//! ARM implementation of `add_limbs_unchecked`.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Add `len` limbs from `src` into `dst` with carry propagation.
///
/// Returns the final carry-out limb (0 or 1).
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

    // SAFETY: Assembly block uses pointers guaranteed to be valid by caller bounds
    unsafe {
        asm!(
            "cmp {chunks}, #0",
            "beq 2f",
            "lsrs {carry}, {carry}, #1",
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            // Limb 0
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "adcs {d}, {d}, {s}",
            "str {d}, [{dst}], #4",
            // Limb 1
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "adcs {d}, {d}, {s}",
            "str {d}, [{dst}], #4",
            // Limb 2
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "adcs {d}, {d}, {s}",
            "str {d}, [{dst}], #4",
            // Limb 3
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "adcs {d}, {d}, {s}",
            "str {d}, [{dst}], #4",

            "mov {carry}, #0",
            "adc {carry}, {carry}, #0",
            "subs {chunks}, {chunks}, #1",
            "beq 2f",
            "lsrs {carry}, {carry}, #1",
            "b 1b",

            "2:",
            "cmp {rem}, #0",
            "beq 4f",
            "lsrs {carry}, {carry}, #1",
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "adcs {d}, {d}, {s}",
            "str {d}, [{dst}], #4",

            "mov {carry}, #0",
            "adc {carry}, {carry}, #0",
            "subs {rem}, {rem}, #1",
            "beq 4f",
            "lsrs {carry}, {carry}, #1",
            "b 3b",
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

/// Straight-line `dst[i] = dst[i] + src[i] + carry` chain for `len` in
/// `2..=4`.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint: the kernel
///   reads `src[i]` and `dst[i]` and then writes `dst[i]`, so a partial
///   overlap is a data race.
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
                    "ldr {s0}, [{src}]",
                    "ldr {s1}, [{src}, #4]",
                    "ldr {d0}, [{dst}]",
                    "ldr {d1}, [{dst}, #4]",
                    "adds {d0}, {d0}, {s0}",
                    "adcs {d1}, {d1}, {s1}",
                    "str {d0}, [{dst}]",
                    "str {d1}, [{dst}, #4]",
                    "mov {carry}, #0",
                    "adc {carry}, {carry}, #0",
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
                    "ldr {s0}, [{src}]",
                    "ldr {s1}, [{src}, #4]",
                    "ldr {s2}, [{src}, #8]",
                    "ldr {d0}, [{dst}]",
                    "ldr {d1}, [{dst}, #4]",
                    "ldr {d2}, [{dst}, #8]",
                    "adds {d0}, {d0}, {s0}",
                    "adcs {d1}, {d1}, {s1}",
                    "adcs {d2}, {d2}, {s2}",
                    "str {d0}, [{dst}]",
                    "str {d1}, [{dst}, #4]",
                    "str {d2}, [{dst}, #8]",
                    "mov {carry}, #0",
                    "adc {carry}, {carry}, #0",
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
                    "ldr {s0}, [{src}]",
                    "ldr {s1}, [{src}, #4]",
                    "ldr {s2}, [{src}, #8]",
                    "ldr {s3}, [{src}, #12]",
                    "ldr {d0}, [{dst}]",
                    "ldr {d1}, [{dst}, #4]",
                    "ldr {d2}, [{dst}, #8]",
                    "ldr {d3}, [{dst}, #12]",
                    "adds {d0}, {d0}, {s0}",
                    "adcs {d1}, {d1}, {s1}",
                    "adcs {d2}, {d2}, {s2}",
                    "adcs {d3}, {d3}, {s3}",
                    "str {d0}, [{dst}]",
                    "str {d1}, [{dst}, #4]",
                    "str {d2}, [{dst}, #8]",
                    "str {d3}, [{dst}, #12]",
                    "mov {carry}, #0",
                    "adc {carry}, {carry}, #0",
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
