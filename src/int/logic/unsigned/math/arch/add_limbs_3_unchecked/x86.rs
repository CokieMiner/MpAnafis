//! x86 addition into a disjoint destination.
//!
//! `adcl` propagates CF through four-limb blocks and a scalar tail.
//! `movl`, `leal`, `decl`, and branches preserve CF between additions.
//! EAX holds each limb sum and then the final carry, leaving five other
//! registers for pointers and counts when EBP is reserved as a frame pointer.

use core::arch::asm;

use super::Limb;

/// Writes the low `len` limbs of `src1 + src2` and returns the binary carry.
///
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// For nonempty spans, all pointers must be aligned and cover `len` limbs.
/// Each span must lie within one live allocation and have at most `isize::MAX` bytes.
/// The sources must be initialized and readable; the destination must be
/// writable and disjoint from both sources. Its previous contents are unused.
/// The two source spans may overlap each other.
#[expect(
    clippy::inline_always,
    reason = "Keep the hardware carry chain in the selected arithmetic caller"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 1 {
        // SAFETY: len == 1 bounds both aligned, initialized source spans.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: len == 1 bounds the aligned, writable destination span.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }

    let carry: Limb;
    let chunks = len >> 2;
    let remainder = len & 3;

    // SAFETY: 4 * chunks + remainder == len bounds every aligned read and
    // write. Signed counters fit because each span is at most isize::MAX bytes.
    // Empty spans skip memory access. Both sources are initialized and may
    // overlap; dst is disjoint and only written. All modified GPRs are declared.
    unsafe {
        asm!(
            "clc",                                      // Initialize CF to zero
            "decl {chunks}",                             // Pre-decrement chunk counter for sign test
            "js 2f",                                     // If chunks < 0 (len < 4), skip to remainder (2f)

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "movl 0({src1}), %eax",                      // Load src1[0]
            "adcl 0({src2}), %eax",                      // %eax += src2[0] + CF (updates CF)
            "movl %eax, 0({dst})",                       // Store dst[0]

            // [Limb 1]
            "movl 4({src1}), %eax",                      // Load src1[1]
            "adcl 4({src2}), %eax",                      // %eax += src2[1] + CF
            "movl %eax, 4({dst})",                       // Store dst[1]

            // [Limb 2]
            "movl 8({src1}), %eax",                      // Load src1[2]
            "adcl 8({src2}), %eax",                      // %eax += src2[2] + CF
            "movl %eax, 8({dst})",                       // Store dst[2]

            // [Limb 3]
            "movl 12({src1}), %eax",                     // Load src1[3]
            "adcl 12({src2}), %eax",                     // %eax += src2[3] + CF
            "movl %eax, 12({dst})",                      // Store dst[3]

            // Advance pointers and decrement the count without changing CF.
            "leal 16({src1}), {src1}",                   // Advance src1 pointer (preserves CF)
            "leal 16({src2}), {src2}",                   // Advance src2 pointer (preserves CF)
            "leal 16({dst}), {dst}",                     // Advance dst pointer (preserves CF)
            "decl {chunks}",                             // Decrement chunk counter (preserves CF)
            "jns 1b",                                    // Repeat while chunks >= 0

            // Remainder entry point (0 to 3 limbs)
            "2:",
            "decl {remainder}",                          // Pre-decrement remainder counter (preserves CF)
            "js 4f",                                     // If remainder < 0, skip to exit (4f)

            // 1-limb tail loop
            "3:",
            "movl 0({src1}), %eax",                      // Load single src1 limb
            "adcl 0({src2}), %eax",                      // Add with CF
            "movl %eax, 0({dst})",                       // Store single dst limb
            "leal 4({src1}), {src1}",                    // Advance pointers (preserves CF)
            "leal 4({src2}), {src2}",
            "leal 4({dst}), {dst}",
            "decl {remainder}",                          // Decrement remainder (preserves CF)
            "jns 3b",                                    // Repeat while remainder >= 0

            // Capture final carry bit
            "4:",
            "setc %al",                                 // Capture the final CF bit
            "movzbl %al, %eax",                          // Return the carry as a complete Limb

            dst = inout(reg) dst => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            chunks = inout(reg) chunks => _,
            remainder = inout(reg) remainder => _,
            out("eax") carry,
            options(nostack, att_syntax)
        );
    }
    carry
}
