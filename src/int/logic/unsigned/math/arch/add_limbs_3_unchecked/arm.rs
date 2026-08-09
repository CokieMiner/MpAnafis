//! ARM implementation of `add_limbs_3_unchecked`.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Compute `dst[i] = src1[i] + src2[i] + carry` for `len` limbs,
/// returning the final carry.
///
/// # Safety
///
/// `dst`, `src1`, and `src2` must each be valid for `len` elements.
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

    // SAFETY: Assembly block uses pointers guaranteed to be valid by caller bounds
    unsafe {
        asm!(
            "cmp {chunks}, #0",
            "beq 2f",
            "lsrs {carry}, {carry}, #1",
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            // Limb 0
            "ldr {s1}, [{src1}], #4",
            "ldr {s2}, [{src2}], #4",
            "adcs {s1}, {s1}, {s2}",
            "str {s1}, [{dst}], #4",
            // Limb 1
            "ldr {s1}, [{src1}], #4",
            "ldr {s2}, [{src2}], #4",
            "adcs {s1}, {s1}, {s2}",
            "str {s1}, [{dst}], #4",
            // Limb 2
            "ldr {s1}, [{src1}], #4",
            "ldr {s2}, [{src2}], #4",
            "adcs {s1}, {s1}, {s2}",
            "str {s1}, [{dst}], #4",
            // Limb 3
            "ldr {s1}, [{src1}], #4",
            "ldr {s2}, [{src2}], #4",
            "adcs {s1}, {s1}, {s2}",
            "str {s1}, [{dst}], #4",

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
            "ldr {s1}, [{src1}], #4",
            "ldr {s2}, [{src2}], #4",
            "adcs {s1}, {s1}, {s2}",
            "str {s1}, [{dst}], #4",

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
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            dst = inout(reg) dst => _,
            s1 = out(reg) _,
            s2 = out(reg) _,
            options(nostack)
        );
        carry
    }
}

/// Straight-line `dst[i] = src1[i] + src2[i] + carry` chain for `len` in
/// `2..=4`.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span: it is written while `src1`
///   and `src2` are read.
/// - `src1` and `src2` are read-only and may alias each other.
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
                    "ldr {a0}, [{src1}]",
                    "ldr {a1}, [{src1}, #4]",
                    "ldr {b0}, [{src2}]",
                    "ldr {b1}, [{src2}, #4]",
                    "adds {a0}, {a0}, {b0}",
                    "adcs {a1}, {a1}, {b1}",
                    "str {a0}, [{dst}]",
                    "str {a1}, [{dst}, #4]",
                    "mov {carry}, #0",
                    "adc {carry}, {carry}, #0",
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
                    "ldr {a0}, [{src1}]",
                    "ldr {a1}, [{src1}, #4]",
                    "ldr {a2}, [{src1}, #8]",
                    "ldr {b0}, [{src2}]",
                    "ldr {b1}, [{src2}, #4]",
                    "ldr {b2}, [{src2}, #8]",
                    "adds {a0}, {a0}, {b0}",
                    "adcs {a1}, {a1}, {b1}",
                    "adcs {a2}, {a2}, {b2}",
                    "str {a0}, [{dst}]",
                    "str {a1}, [{dst}, #4]",
                    "str {a2}, [{dst}, #8]",
                    "mov {carry}, #0",
                    "adc {carry}, {carry}, #0",
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
                    "ldr {a0}, [{src1}]",
                    "ldr {a1}, [{src1}, #4]",
                    "ldr {a2}, [{src1}, #8]",
                    "ldr {a3}, [{src1}, #12]",
                    "ldr {b0}, [{src2}]",
                    "ldr {b1}, [{src2}, #4]",
                    "ldr {b2}, [{src2}, #8]",
                    "ldr {b3}, [{src2}, #12]",
                    "adds {a0}, {a0}, {b0}",
                    "adcs {a1}, {a1}, {b1}",
                    "adcs {a2}, {a2}, {b2}",
                    "adcs {a3}, {a3}, {b3}",
                    "str {a0}, [{dst}]",
                    "str {a1}, [{dst}, #4]",
                    "str {a2}, [{dst}, #8]",
                    "str {a3}, [{dst}, #12]",
                    "mov {carry}, #0",
                    "adc {carry}, {carry}, #0",
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
