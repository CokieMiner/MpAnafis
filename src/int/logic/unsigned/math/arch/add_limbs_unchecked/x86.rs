//! 32-bit x86 in-place addition kernel.
//!
//! Evaluates `dst += src` using prefix and 8-way unrolled `adcl` carry chains with CF-preserving addressing.

use core::arch::asm;

use super::Limb;

/// Add `src[0..len]` into `dst[0..len]` and return the final carry.
///
/// Computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + src[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// `adcl` carry flags flow continuously across iterations because `leal`, `decl`, `movl`,
/// and jump instructions preserve CF on x86. The loop is 8-way unrolled to maximize ILP.
///
/// # Safety
///
/// - Both pointers must cover `len` limbs.
/// - Memory spans must not overlap.
#[allow(
    clippy::inline_always,
    reason = "Addition is a foundational limb loop and preserving CF avoids per-limb carry materialization"
)]
#[inline(always)]
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    if len == 1 {
        // SAFETY: the caller guarantees both pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*dst).overflowing_add(*src) };
        // SAFETY: the caller guarantees the destination limb is writable.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }

    let carry: Limb;
    let prefix = (len >> 2) & 1;
    let chunks = len >> 3;
    let remainder = len & 3;

    // SAFETY:
    // 1. `dst` and `src` cover `len` 32-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "xorl {carry}, {carry}",                     // Clear CF and zero carry register
            // Optional 4-limb prefix block
            "decl {prefix}",                             // Decrement prefix flag (preserves CF)
            "js 1f",                                     // If prefix == 0, jump to 8-way loop (1f)
            "movl 0({src}), %eax",                       // Load src[0]
            "adcl %eax, 0({dst})",                       // dst[0] += src[0] + CF
            "movl 4({src}), %eax",                       // Load src[1]
            "adcl %eax, 4({dst})",                       // dst[1] += src[1] + CF
            "movl 8({src}), %eax",                       // Load src[2]
            "adcl %eax, 8({dst})",                       // dst[2] += src[2] + CF
            "movl 12({src}), %eax",                      // Load src[3]
            "adcl %eax, 12({dst})",                      // dst[3] += src[3] + CF
            "leal 16({src}), {src}",                     // Advance src pointer by 16 (preserves CF)
            "leal 16({dst}), {dst}",                     // Advance dst pointer by 16 (preserves CF)

            // Main 8-way unrolled loop
            "1:",
            "decl {chunks}",                             // Pre-decrement chunk counter (preserves CF)
            "js 3f",                                     // If chunks < 0, jump to remainder (3f)
            "2:",
            "movl 0({src}), %eax",                       // Load src[0]
            "adcl %eax, 0({dst})",                       // dst[0] += src[0] + CF
            "movl 4({src}), %eax",                       // Load src[1]
            "adcl %eax, 4({dst})",                       // dst[1] += src[1] + CF
            "movl 8({src}), %eax",                       // Load src[2]
            "adcl %eax, 8({dst})",                       // dst[2] += src[2] + CF
            "movl 12({src}), %eax",                      // Load src[3]
            "adcl %eax, 12({dst})",                      // dst[3] += src[3] + CF
            "movl 16({src}), %eax",                      // Load src[4]
            "adcl %eax, 16({dst})",                      // dst[4] += src[4] + CF
            "movl 20({src}), %eax",                      // Load src[5]
            "adcl %eax, 20({dst})",                      // dst[5] += src[5] + CF
            "movl 24({src}), %eax",                      // Load src[6]
            "adcl %eax, 24({dst})",                      // dst[6] += src[6] + CF
            "movl 28({src}), %eax",                      // Load src[7]
            "adcl %eax, 28({dst})",                      // dst[7] += src[7] + CF
            "leal 32({src}), {src}",                     // Advance src by 32 (preserves CF)
            "leal 32({dst}), {dst}",                     // Advance dst by 32 (preserves CF)
            "decl {chunks}",                             // Decrement chunks (preserves CF)
            "jns 2b",                                    // Repeat while chunks >= 0

            // Remainder entry point (0 to 3 limbs)
            "3:",
            "decl {remainder}",                          // Pre-decrement remainder counter (preserves CF)
            "js 5f",                                     // If remainder < 0, skip to exit (5f)
            "4:",
            "movl 0({src}), %eax",                       // Load single src limb
            "adcl %eax, 0({dst})",                       // dst += src + CF
            "leal 4({src}), {src}",                      // Advance src (preserves CF)
            "leal 4({dst}), {dst}",                      // Advance dst (preserves CF)
            "decl {remainder}",                          // Decrement remainder (preserves CF)
            "jns 4b",                                    // Repeat while remainder >= 0

            // Capture final carry
            "5:",
            "adcl {carry}, {carry}",                     // carry = 0 + CF (0 or 1)

            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            prefix = inout(reg) prefix => _,
            chunks = inout(reg) chunks => _,
            remainder = inout(reg) remainder => _,
            carry = out(reg) carry,
            out("eax") _,
            options(nostack, att_syntax)
        );
    }
    carry
}
