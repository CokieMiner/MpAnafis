//! `x86_64` baseline (non-ADX) subtraction kernels (inline assembly).
//!
//! Fixed two- through four-limb chains precede the general eight-limb loop
//! and its four-, two-, and one-limb tail blocks. Loop control preserves CF.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Compute `dst[i] = src1[i] - src2[i] - borrow` for `len` limbs,
/// returning the final borrow.
///
/// `dst - borrow*B^len = src1 - src2`, with `borrow` in `{0,1}`.
///
/// # Safety
///
/// Both sources must cover `len` aligned initialized readable limbs and may
/// alias each other. `dst` must cover `len` aligned writable limbs, disjoint
/// from both sources; its contents may be uninitialized. Each byte span must
/// fit in `isize::MAX`. `len == 0` performs no pointer access.
#[expect(clippy::inline_always, reason = "Inlining exposes the fixed assembly borrow chains to callers")]
#[expect(
    clippy::too_many_lines,
    reason = "Explicit assembly sequence unrolled across 8 limbs and 4/2/1 tail blocks"
)]
#[inline(always)]
pub unsafe fn sub_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if (2..=4).contains(&len) {
        // SAFETY: caller guarantees pointers cover `len` limbs (`2..=4`).
        return unsafe { sub_small_3_unchecked(dst, src1, src2, len) };
    }
    let mut borrow: Limb;
    let chunks = len >> 3;
    let any_tail = len & 7;
    let tail_4 = (len >> 2) & 1;
    let tail_2 = (len >> 1) & 1;
    let tail_1 = len & 1;

    // SAFETY: eight-limb blocks and the 4/2/1 tail partition exactly len
    // initialized aligned source limbs and disjoint writable output limbs.
    // Sources may coincide. The byte-span bound keeps the signed block count
    // nonnegative. LEA, DEC, MOV, and branches preserve CF. Changed pointers,
    // counters, and digit temporaries are early outputs; dst is never read.
    unsafe {
        asm!(
            "xorl {borrow:e}, {borrow:e}",                // CF = 0 (no initial borrow), zero borrow register
            // Main 8-way unrolled loop
            "decq {chunks}",                             // Pre-decrement chunk counter (preserves CF)
            "js 3f",                                     // If chunks < 0, jump to remainder (3f)
            "2:",
            "movq ({src1}), %rax",                       // Load src1[0]
            "movq 8({src1}), %rcx",                      // Load src1[1]
            "sbbq ({src2}), %rax",                       // %rax = src1[0] - src2[0] - CF
            "sbbq 8({src2}), %rcx",                      // %rcx = src1[1] - src2[1] - CF
            "movq 16({src1}), %rdx",                     // Load src1[2]
            "movq 24({src1}), %r8",                      // Load src1[3]
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "sbbq 16({src2}), %rdx",                     // %rdx = src1[2] - src2[2] - CF
            "sbbq 24({src2}), %r8",                      // %r8 = src1[3] - src2[3] - CF
            "movq 32({src1}), %rax",                     // Load src1[4]
            "movq 40({src1}), %rcx",                     // Load src1[5]
            "movq %rdx, 16({dst})",                      // Store dst[2]
            "movq %r8, 24({dst})",                       // Store dst[3]
            "sbbq 32({src2}), %rax",                     // %rax = src1[4] - src2[4] - CF
            "sbbq 40({src2}), %rcx",                     // %rcx = src1[5] - src2[5] - CF
            "movq 48({src1}), %rdx",                     // Load src1[6]
            "movq 56({src1}), %r8",                      // Load src1[7]
            "movq %rax, 32({dst})",                      // Store dst[4]
            "movq %rcx, 40({dst})",                      // Store dst[5]
            "sbbq 48({src2}), %rdx",                     // %rdx = src1[6] - src2[6] - CF
            "sbbq 56({src2}), %r8",                      // %r8 = src1[7] - src2[7] - CF
            "movq %rdx, 48({dst})",                      // Store dst[6]
            "movq %r8, 56({dst})",                       // Store dst[7]
            "leaq 64({dst}), {dst}",                     // Advance dst by 64 bytes (preserves CF)
            "leaq 64({src1}), {src1}",                   // Advance src1 by 64 bytes (preserves CF)
            "leaq 64({src2}), {src2}",                   // Advance src2 by 64 bytes (preserves CF)
            "decq {chunks}",                             // Decrement chunks (preserves CF)
            "jns 2b",                                    // Repeat while chunks >= 0

            // Tail: descending 4/2/1 blocks
            "3:",
            "decq {any_tail}",                           // Test if any remainder limbs remain (preserves CF)
            "js 6f",                                     // If none, jump directly to exit (6f)
            "decq {tail_4}",                             // Test 4-limb tail block
            "js 4f",                                     // Skip to 2-limb test if absent (4f)
            "movq ({src1}), %rax",                       // Load src1[0]
            "movq 8({src1}), %rcx",                      // Load src1[1]
            "sbbq ({src2}), %rax",                       // src1[0] - src2[0] - CF
            "sbbq 8({src2}), %rcx",                      // src1[1] - src2[1] - CF
            "movq 16({src1}), %rdx",                     // Load src1[2]
            "movq 24({src1}), %r8",                      // Load src1[3]
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "sbbq 16({src2}), %rdx",                     // src1[2] - src2[2] - CF
            "sbbq 24({src2}), %r8",                      // src1[3] - src2[3] - CF
            "movq %rdx, 16({dst})",                      // Store dst[2]
            "movq %r8, 24({dst})",                       // Store dst[3]
            "leaq 32({dst}), {dst}",                     // Advance dst by 32 bytes (preserves CF)
            "leaq 32({src1}), {src1}",                   // Advance src1 by 32 bytes (preserves CF)
            "leaq 32({src2}), {src2}",                   // Advance src2 by 32 bytes (preserves CF)

            "4:",
            "decq {tail_2}",                             // Test 2-limb tail block
            "js 5f",                                     // Skip to 1-limb test if absent (5f)
            "movq ({src1}), %rax",                       // Load src1[0]
            "movq 8({src1}), %rcx",                      // Load src1[1]
            "sbbq ({src2}), %rax",                       // src1[0] - src2[0] - CF
            "sbbq 8({src2}), %rcx",                      // src1[1] - src2[1] - CF
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "leaq 16({dst}), {dst}",                     // Advance dst by 16 bytes (preserves CF)
            "leaq 16({src1}), {src1}",                   // Advance src1 by 16 bytes (preserves CF)
            "leaq 16({src2}), {src2}",                   // Advance src2 by 16 bytes (preserves CF)

            "5:",
            "decq {tail_1}",                             // Test 1-limb tail block
            "js 6f",                                     // Skip if absent (6f)
            "movq ({src1}), %rax",                       // Load single src1 limb
            "sbbq ({src2}), %rax",                       // %rax = src1[0] - src2[0] - CF
            "movq %rax, ({dst})",                        // Store single dst limb

            // Exit: capture final borrow
            "6:",
            "adcq {borrow}, {borrow}",                   // borrow = 0 + 0 + CF (0 or 1)
            borrow = out(reg) borrow,
            dst = inout(reg) dst => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            chunks = inout(reg) chunks => _,
            any_tail = inout(reg) any_tail => _,
            tail_4 = inout(reg) tail_4 => _,
            tail_2 = inout(reg) tail_2 => _,
            tail_1 = inout(reg) tail_1 => _,
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            out("r8") _,
            options(nostack, att_syntax)
        );
    }
    borrow
}

/// Straight-line `dst[i] = src1[i] - src2[i] - borrow` chain for `len` in `2..=4`.
///
/// # Safety
///
/// `2 <= len <= 4`. Both sources must cover `len` aligned initialized readable
/// limbs and may alias each other. `dst` must cover `len` aligned writable
/// limbs, disjoint from both sources; its contents may be uninitialized.
#[expect(
    clippy::inline_always,
    reason = "The fixed-size borrow chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn sub_small_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    match len {
        2 => {
            let mut borrow: Limb;
            // SAFETY: len=2 selects two initialized source limbs and two
            // disjoint writable output limbs. SUB/SBB threads CF; all digit
            // temporaries are early outputs and dst is never read.
            unsafe {
                asm!(
                    "xorl {borrow:e}, {borrow:e}",       // Clear CF and zero borrow register
                    "movq ({src1}), %rax",               // Load src1[0]
                    "movq 8({src1}), %rcx",              // Load src1[1]
                    "subq ({src2}), %rax",               // %rax = src1[0] - src2[0], set CF
                    "sbbq 8({src2}), %rcx",              // %rcx = src1[1] - src2[1] - CF
                    "movq %rax, ({dst})",                // Store dst[0]
                    "movq %rcx, 8({dst})",               // Store dst[1]
                    "adcq {borrow}, {borrow}",           // borrow = 0 + CF (0 or 1)
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    borrow = out(reg) borrow,
                    out("rax") _,
                    out("rcx") _,
                    options(nostack, att_syntax)
                );
            }
            borrow
        }
        3 => {
            let mut borrow: Limb;
            // SAFETY: len=3 selects three initialized source limbs and three
            // disjoint writable output limbs. SUB/SBB threads CF; all digit
            // temporaries are early outputs and dst is never read.
            unsafe {
                asm!(
                    "xorl {borrow:e}, {borrow:e}",       // Clear CF and zero borrow register
                    "movq ({src1}), %rax",               // Load src1[0]
                    "movq 8({src1}), %rcx",              // Load src1[1]
                    "movq 16({src1}), %rdx",             // Load src1[2]
                    "subq ({src2}), %rax",               // %rax = src1[0] - src2[0], set CF
                    "sbbq 8({src2}), %rcx",              // %rcx = src1[1] - src2[1] - CF
                    "sbbq 16({src2}), %rdx",             // %rdx = src1[2] - src2[2] - CF
                    "movq %rax, ({dst})",                // Store dst[0]
                    "movq %rcx, 8({dst})",               // Store dst[1]
                    "movq %rdx, 16({dst})",              // Store dst[2]
                    "adcq {borrow}, {borrow}",           // borrow = 0 + CF
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    borrow = out(reg) borrow,
                    out("rax") _,
                    out("rcx") _,
                    out("rdx") _,
                    options(nostack, att_syntax)
                );
            }
            borrow
        }
        4 => {
            let mut borrow: Limb;
            // SAFETY: len=4 selects four initialized source limbs and four
            // disjoint writable output limbs. SUB/SBB threads CF; all digit
            // temporaries are early outputs and dst is never read.
            unsafe {
                asm!(
                    "xorl {borrow:e}, {borrow:e}",       // Clear CF and zero borrow register
                    "movq ({src1}), %rax",               // Load src1[0]
                    "movq 8({src1}), %rcx",              // Load src1[1]
                    "movq 16({src1}), %rdx",             // Load src1[2]
                    "movq 24({src1}), %r8",              // Load src1[3]
                    "subq ({src2}), %rax",               // %rax = src1[0] - src2[0], set CF
                    "sbbq 8({src2}), %rcx",              // %rcx = src1[1] - src2[1] - CF
                    "sbbq 16({src2}), %rdx",             // %rdx = src1[2] - src2[2] - CF
                    "sbbq 24({src2}), %r8",              // %r8 = src1[3] - src2[3] - CF
                    "movq %rax, ({dst})",                // Store dst[0]
                    "movq %rcx, 8({dst})",               // Store dst[1]
                    "movq %rdx, 16({dst})",              // Store dst[2]
                    "movq %r8, 24({dst})",               // Store dst[3]
                    "adcq {borrow}, {borrow}",           // borrow = 0 + CF
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    borrow = out(reg) borrow,
                    out("rax") _,
                    out("rcx") _,
                    out("rdx") _,
                    out("r8") _,
                    options(nostack, att_syntax)
                );
            }
            borrow
        }
        // SAFETY: The caller guarantees `2 <= len <= 4`.
        _ => unsafe { unreachable_unchecked() },
    }
}
