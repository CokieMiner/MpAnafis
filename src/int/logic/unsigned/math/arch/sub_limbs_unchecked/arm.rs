//! ARM implementation of `sub_limbs_unchecked`.

use core::arch::asm;

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` with borrow propagation.
///
/// Returns the final borrow-out limb (0 or 1).
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
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
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
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "sbcs {d}, {d}, {s}",
            "str {d}, [{dst}], #4",
            // Limb 1
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "sbcs {d}, {d}, {s}",
            "str {d}, [{dst}], #4",
            // Limb 2
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "sbcs {d}, {d}, {s}",
            "str {d}, [{dst}], #4",
            // Limb 3
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "sbcs {d}, {d}, {s}",
            "str {d}, [{dst}], #4",

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
            "ldr {s}, [{src}], #4",
            "ldr {d}, [{dst}]",
            "sbcs {d}, {d}, {s}",
            "str {d}, [{dst}], #4",

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
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            s = out(reg) _,
            d = out(reg) _,
            options(nostack)
        );
        borrow
    }
}
