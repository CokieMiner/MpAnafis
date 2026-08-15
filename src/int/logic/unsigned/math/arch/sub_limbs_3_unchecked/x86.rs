//! 32-bit x86 three-span subtraction kernel.
//!
//! Evaluates `dst = src1 - src2` using 4-way unrolled `sbbl` borrow chains and CF-preserving addressing.

use core::arch::asm;

use super::Limb;

/// Write `src1 - src2` into `dst` and return the final borrow.
///
/// Computes:
///
/// ```text
///   (borrow, dst[0..len]) = src1[0..len] - src2[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// `sbbl` borrow flags flow continuously across iterations because `leal`, `decl`, `movl`,
/// and jump instructions preserve CF on x86. The loop is 4-way unrolled to maximize ILP.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span.
#[allow(
    clippy::inline_always,
    reason = "Three-span subtraction is a hot interpolation loop and preserving CF removes per-limb borrow conversion"
)]
#[inline(always)]
pub unsafe fn sub_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 1 {
        // SAFETY: the caller guarantees both source pointers cover one limb.
        let (difference, underflow) = unsafe { (*src1).overflowing_sub(*src2) };
        // SAFETY: the caller guarantees one writable destination limb.
        unsafe {
            *dst = difference;
        }
        return Limb::from(underflow);
    }

    let borrow: Limb;
    let chunks = len >> 2;
    let remainder = len & 3;

    // SAFETY:
    // 1. `dst`, `src1`, and `src2` are valid for `len` 32-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "xorl {borrow}, {borrow}",                   // Clear CF and zero borrow output register
            "decl {chunks}",                             // Pre-decrement chunk counter for sign test
            "js 2f",                                     // If chunks < 0 (len < 4), skip to remainder (2f)

            // Main 4-way unrolled loop
            "1:",
            // [Limb 0]
            "movl 0({src1}), %eax",                      // Load src1[0]
            "sbbl 0({src2}), %eax",                      // %eax -= src2[0] + CF (updates CF)
            "movl %eax, 0({dst})",                       // Store dst[0]

            // [Limb 1]
            "movl 4({src1}), %eax",                      // Load src1[1]
            "sbbl 4({src2}), %eax",                      // %eax -= src2[1] + CF
            "movl %eax, 4({dst})",                       // Store dst[1]

            // [Limb 2]
            "movl 8({src1}), %eax",                      // Load src1[2]
            "sbbl 8({src2}), %eax",                      // %eax -= src2[2] + CF
            "movl %eax, 8({dst})",                       // Store dst[2]

            // [Limb 3]
            "movl 12({src1}), %eax",                     // Load src1[3]
            "sbbl 12({src2}), %eax",                     // %eax -= src2[3] + CF
            "movl %eax, 12({dst})",                      // Store dst[3]

            // Advance pointers by 16 bytes and loop (leal and decl preserve CF!)
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
            "sbbl 0({src2}), %eax",                      // Subtract with CF
            "movl %eax, 0({dst})",                       // Store single dst limb
            "leal 4({src1}), {src1}",                    // Advance pointers (preserves CF)
            "leal 4({src2}), {src2}",
            "leal 4({dst}), {dst}",
            "decl {remainder}",                          // Decrement remainder (preserves CF)
            "jns 3b",                                    // Repeat while remainder >= 0

            // Capture final borrow bit
            "4:",
            "adcl {borrow}, {borrow}",                   // borrow = borrow + borrow + CF (0 or 1)

            dst = inout(reg) dst => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            chunks = inout(reg) chunks => _,
            remainder = inout(reg) remainder => _,
            borrow = out(reg) borrow,
            out("eax") _,
            options(nostack, att_syntax)
        );
    }
    borrow
}
