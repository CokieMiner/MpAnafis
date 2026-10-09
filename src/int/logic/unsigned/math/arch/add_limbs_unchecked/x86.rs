//! x86 in-place addition with a hardware carry chain.
//!
//! A four-limb prefix precedes eight-limb blocks and a scalar tail.
//! `movl`, `leal`, `decl`, and branches preserve CF between additions.
//! EAX holds source limbs and then the final carry, leaving five registers
//! for pointers and counts when EBP is reserved as a frame pointer.

use core::arch::asm;

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

    // SAFETY: 4 * prefix + 8 * chunks + remainder == len bounds every
    // aligned access. Counts fit signed registers because spans have at most
    // isize::MAX bytes. Each input is read before its result is written, so
    // exact aliasing is valid. Empty spans skip all memory access.
    unsafe {
        asm!(
            "clc",                                      // Initialize CF to zero
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
            "setc %al",                                 // Capture the final CF bit
            "movzbl %al, %eax",                          // Initialize the complete carry limb

            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            prefix = inout(reg) prefix => _,
            chunks = inout(reg) chunks => _,
            remainder = inout(reg) remainder => _,
            out("eax") carry,
            options(nostack, att_syntax)
        );
    }
    carry
}
