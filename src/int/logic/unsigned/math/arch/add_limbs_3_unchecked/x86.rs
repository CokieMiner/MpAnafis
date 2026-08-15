//! 32-bit x86 three-span addition kernel.
//!
//! Evaluates `dst = src1 + src2` using 4-way unrolled `adcl` carry chains and CF-preserving addressing.

use core::arch::asm;

use super::Limb;

/// Write `src1 + src2` into `dst` and return the final carry.
///
/// Computes:
///
/// ```text
///   (carry, dst[0..len]) = src1[0..len] + src2[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// `adcl` carry flags flow continuously across iterations because `leal`, `decl`, `movl`,
/// and jump instructions preserve CF on x86. The loop is 4-way unrolled to maximize ILP.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for reads and writes of `len` 32-bit limbs.
/// - `dst` must not overlap either input span.
#[allow(
    clippy::inline_always,
    reason = "Three-span addition is a hot interpolation loop and preserving CF removes per-limb carry conversion"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 1 {
        // SAFETY: the caller guarantees both source pointers cover one limb.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: the caller guarantees one writable destination limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }

    let carry: Limb;
    let chunks = len >> 2;
    let remainder = len & 3;

    // SAFETY:
    // 1. `dst`, `src1`, and `src2` cover `len` 32-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "xorl {carry}, {carry}",                     // Clear CF and zero carry output register
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
            "adcl 0({src2}), %eax",                      // Add with CF
            "movl %eax, 0({dst})",                       // Store single dst limb
            "leal 4({src1}), {src1}",                    // Advance pointers (preserves CF)
            "leal 4({src2}), {src2}",
            "leal 4({dst}), {dst}",
            "decl {remainder}",                          // Decrement remainder (preserves CF)
            "jns 3b",                                    // Repeat while remainder >= 0

            // Capture final carry bit
            "4:",
            "adcl {carry}, {carry}",                     // carry = carry + carry + CF (0 or 1)

            dst = inout(reg) dst => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            chunks = inout(reg) chunks => _,
            remainder = inout(reg) remainder => _,
            carry = out(reg) carry,
            out("eax") _,
            options(nostack, att_syntax)
        );
    }
    carry
}
