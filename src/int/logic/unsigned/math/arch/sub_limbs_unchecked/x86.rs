//! 32-bit x86 in-place subtraction kernel.
//!
//! Evaluates `dst -= src` using prefix and 8-way unrolled `sbbl` borrow chains with CF-preserving addressing.

use core::arch::asm;

use super::Limb;

/// Subtract `src[0..len]` from `dst[0..len]` and return the final borrow.
///
/// `dst_after - borrow*B^len = dst_before - src`, with `borrow` in `{0,1}`.
///
/// # Safety
///
/// `dst` must cover `len` aligned initialized writable limbs. `src` must cover
/// `len` aligned initialized readable limbs. The spans must be identical or
/// disjoint, and each byte span must fit in `isize::MAX`. `len == 0` performs
/// no pointer access.
#[expect(
    clippy::inline_always,
    reason = "Subtraction is a foundational limb loop and preserving CF avoids per-limb borrow materialization"
)]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    if len == 1 {
        // SAFETY: the caller guarantees both pointers cover the sole limb.
        let (difference, underflow) = unsafe { (*dst).overflowing_sub(*src) };
        // SAFETY: the caller guarantees the destination limb is writable.
        unsafe {
            *dst = difference;
        }
        return Limb::from(underflow);
    }

    let borrow: Limb;
    let prefix = (len >> 2) & 1;
    let chunks = len >> 3;
    let remainder = len & 3;

    // SAFETY: the four-limb prefix, eight-limb blocks, and len&3 tail partition
    // exactly len initialized aligned limbs. Each source is loaded before its
    // matching write, permitting exact alias. The byte span bounds the signed
    // count. LEA, DEC, MOV, and branches preserve CF. EAX is an early output
    // shared by digit temporaries and final borrow, leaving six registers total.
    unsafe {
        asm!(
            "xorl %eax, %eax",                          // CF = 0
            // Optional 4-limb prefix block
            "decl {prefix}",                             // Decrement prefix flag (preserves CF)
            "js 1f",                                     // If prefix == 0, jump to 8-way loop (1f)
            "movl 0({src}), %eax",                       // Load src[0]
            "sbbl %eax, 0({dst})",                       // dst[0] -= src[0] + CF
            "movl 4({src}), %eax",                       // Load src[1]
            "sbbl %eax, 4({dst})",                       // dst[1] -= src[1] + CF
            "movl 8({src}), %eax",                       // Load src[2]
            "sbbl %eax, 8({dst})",                       // dst[2] -= src[2] + CF
            "movl 12({src}), %eax",                      // Load src[3]
            "sbbl %eax, 12({dst})",                      // dst[3] -= src[3] + CF
            "leal 16({src}), {src}",                     // Advance src pointer by 16 (preserves CF)
            "leal 16({dst}), {dst}",                     // Advance dst pointer by 16 (preserves CF)

            // Main 8-way unrolled loop
            "1:",
            "decl {chunks}",                             // Pre-decrement chunk counter (preserves CF)
            "js 3f",                                     // If chunks < 0, jump to remainder (3f)
            "2:",
            "movl 0({src}), %eax",                       // Load src[0]
            "sbbl %eax, 0({dst})",                       // dst[0] -= src[0] + CF
            "movl 4({src}), %eax",                       // Load src[1]
            "sbbl %eax, 4({dst})",                       // dst[1] -= src[1] + CF
            "movl 8({src}), %eax",                       // Load src[2]
            "sbbl %eax, 8({dst})",                       // dst[2] -= src[2] + CF
            "movl 12({src}), %eax",                      // Load src[3]
            "sbbl %eax, 12({dst})",                      // dst[3] -= src[3] + CF
            "movl 16({src}), %eax",                      // Load src[4]
            "sbbl %eax, 16({dst})",                      // dst[4] -= src[4] + CF
            "movl 20({src}), %eax",                      // Load src[5]
            "sbbl %eax, 20({dst})",                      // dst[5] -= src[5] + CF
            "movl 24({src}), %eax",                      // Load src[6]
            "sbbl %eax, 24({dst})",                      // dst[6] -= src[6] + CF
            "movl 28({src}), %eax",                      // Load src[7]
            "sbbl %eax, 28({dst})",                      // dst[7] -= src[7] + CF
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
            "sbbl %eax, 0({dst})",                       // dst -= src + CF
            "leal 4({src}), {src}",                      // Advance src (preserves CF)
            "leal 4({dst}), {dst}",                      // Advance dst (preserves CF)
            "decl {remainder}",                          // Decrement remainder (preserves CF)
            "jns 4b",                                    // Repeat while remainder >= 0

            // Capture final borrow bit
            "5:",
            "sbbl %eax, %eax",                          // EAX = -CF
            "negl %eax",                               // borrow = 0 or 1

            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            prefix = inout(reg) prefix => _,
            chunks = inout(reg) chunks => _,
            remainder = inout(reg) remainder => _,
            out("eax") borrow,
            options(nostack, att_syntax)
        );
    }
    borrow
}
