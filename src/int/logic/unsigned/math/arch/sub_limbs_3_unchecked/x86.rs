//! 32-bit x86 three-span subtraction kernel.
//!
//! Evaluates `dst = src1 - src2` using 4-way unrolled `sbbl` borrow chains and CF-preserving addressing.

use core::arch::asm;

use super::Limb;

/// Write `src1 - src2` into `dst` and return the final borrow.
///
/// `dst - borrow*B^len = src1 - src2`, with `borrow` in `{0,1}`.
///
/// # Safety
///
/// Both sources must cover `len` aligned initialized readable limbs and may
/// alias each other. `dst` must cover `len` aligned writable limbs, disjoint
/// from both sources; its contents may be uninitialized. Each byte span must
/// fit in `isize::MAX`. `len == 0` performs no pointer access.
#[expect(
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

    // SAFETY: four-limb blocks and the len&3 tail access exactly len aligned
    // initialized source limbs and disjoint writable output limbs. Sources may
    // coincide. The byte-span bound keeps the signed block count nonnegative.
    // LEA, DEC, MOV, and branches preserve CF. EAX is an early output used for
    // both the digit temporary and final borrow, leaving six registers in total.
    unsafe {
        asm!(
            "xorl %eax, %eax",                          // CF = 0
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
            "sbbl %eax, %eax",                          // EAX = -CF
            "negl %eax",                               // borrow = 0 or 1

            dst = inout(reg) dst => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            chunks = inout(reg) chunks => _,
            remainder = inout(reg) remainder => _,
            out("eax") borrow,
            options(nostack, att_syntax)
        );
    }
    borrow
}
