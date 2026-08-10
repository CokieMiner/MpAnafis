//! ARM implementation of `sub_limbs_3_unchecked`.

use core::arch::asm;

use super::Limb;

/// Compute `dst[i] = src1[i] − src2[i] − borrow` for `len` limbs,
/// returning the final borrow.
///
/// # Safety
///
/// `dst`, `src1`, and `src2` must each be valid for `len` elements.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn sub_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: Assembly block uses pointers guaranteed to be valid by caller bounds
    unsafe {
        asm!(
            "cmp {chunks}, #0",
            "beq 2f",
            "rsbs {borrow}, {borrow}, #0",
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            // Limb 0
            "ldr {s1}, [{src1}], #4",
            "ldr {s2}, [{src2}], #4",
            "sbcs {s1}, {s1}, {s2}",
            "str {s1}, [{dst}], #4",
            // Limb 1
            "ldr {s1}, [{src1}], #4",
            "ldr {s2}, [{src2}], #4",
            "sbcs {s1}, {s1}, {s2}",
            "str {s1}, [{dst}], #4",
            // Limb 2
            "ldr {s1}, [{src1}], #4",
            "ldr {s2}, [{src2}], #4",
            "sbcs {s1}, {s1}, {s2}",
            "str {s1}, [{dst}], #4",
            // Limb 3
            "ldr {s1}, [{src1}], #4",
            "ldr {s2}, [{src2}], #4",
            "sbcs {s1}, {s1}, {s2}",
            "str {s1}, [{dst}], #4",

            "mov {borrow}, #0",
            "movcc {borrow}, #1",
            "subs {chunks}, {chunks}, #1",
            "beq 2f",
            "rsbs {borrow}, {borrow}, #0",
            "b 1b",

            "2:",
            "cmp {rem}, #0",
            "beq 4f",
            "rsbs {borrow}, {borrow}, #0",
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "ldr {s1}, [{src1}], #4",
            "ldr {s2}, [{src2}], #4",
            "sbcs {s1}, {s1}, {s2}",
            "str {s1}, [{dst}], #4",

            "mov {borrow}, #0",
            "movcc {borrow}, #1",
            "subs {rem}, {rem}, #1",
            "beq 4f",
            "rsbs {borrow}, {borrow}, #0",
            "b 3b",
            "4:",

            borrow = inout(reg) borrow,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            dst = inout(reg) dst => _,
            s1 = out(reg) _,
            s2 = out(reg) _,
            options(nostack)
        );
        borrow
    }
}
