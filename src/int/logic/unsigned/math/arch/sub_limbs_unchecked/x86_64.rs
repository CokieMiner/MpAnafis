//! `x86_64` baseline (non-ADX) subtraction kernels (inline assembly).
//!
//! Fixed two- through four-limb chains precede the general eight-limb loop
//! and its four-, two-, and one-limb tail blocks. Loop control preserves CF.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` and return the final borrow.
///
/// `dst_after - borrow*B^len = dst_before - src`, with `borrow` in `{0,1}`.
///
/// # Safety
///
/// Both pointers must cover `len` aligned initialized limbs; `dst` must be
/// writable. Spans must be identical or disjoint and their byte widths must
/// fit in `isize::MAX`. `len == 0` performs no pointer access.
#[expect(clippy::inline_always, reason = "Inlining exposes the fixed assembly borrow chains to callers")]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    if (2..=4).contains(&len) {
        // SAFETY: the caller guarantees both pointers cover `len` limbs, and
        // this branch proves the fixed kernel's `2..=4` length precondition.
        return unsafe { sub_small_unchecked(dst, src, len) };
    }
    let mut borrow: Limb;
    let chunks = len >> 3;
    let any_tail = len & 7;
    let tail_4 = (len >> 2) & 1;
    let tail_2 = (len >> 1) & 1;
    let tail_1 = len & 1;

    // SAFETY: eight-limb blocks and the 4/2/1 tail partition exactly len
    // initialized aligned limbs. Each source value is read before its matching
    // write, permitting exact alias. The byte span bounds the signed block
    // count. LEA, DEC, MOV, and branches preserve CF. Changed pointers, counters,
    // and digit temporaries are early outputs; no stack memory is accessed.
    unsafe {
        asm!(
            "xorl {borrow:e}, {borrow:e}",                // CF = 0 (no initial borrow), zero borrow register
            // Main 8-way unrolled loop
            "decq {chunks}",                             // Pre-decrement chunk counter (preserves CF)
            "js 3f",                                     // If chunks < 0, jump to remainder (3f)
            "2:",
            "movq ({dst}), %rax",                        // Load dst[0]
            "movq 8({dst}), %rcx",                       // Load dst[1]
            "sbbq ({src}), %rax",                        // %rax = dst[0] - src[0] - CF
            "sbbq 8({src}), %rcx",                       // %rcx = dst[1] - src[1] - CF
            "movq 16({dst}), %rdx",                      // Load dst[2]
            "movq 24({dst}), %r8",                       // Load dst[3]
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "sbbq 16({src}), %rdx",                      // %rdx = dst[2] - src[2] - CF
            "sbbq 24({src}), %r8",                       // %r8 = dst[3] - src[3] - CF
            "movq 32({dst}), %rax",                      // Load dst[4]
            "movq 40({dst}), %rcx",                      // Load dst[5]
            "movq %rdx, 16({dst})",                      // Store dst[2]
            "movq %r8, 24({dst})",                       // Store dst[3]
            "sbbq 32({src}), %rax",                      // %rax = dst[4] - src[4] - CF
            "sbbq 40({src}), %rcx",                      // %rcx = dst[5] - src[5] - CF
            "movq 48({dst}), %rdx",                      // Load dst[6]
            "movq 56({dst}), %r8",                       // Load dst[7]
            "movq %rax, 32({dst})",                      // Store dst[4]
            "movq %rcx, 40({dst})",                      // Store dst[5]
            "sbbq 48({src}), %rdx",                      // %rdx = dst[6] - src[6] - CF
            "sbbq 56({src}), %r8",                       // %r8 = dst[7] - src[7] - CF
            "movq %rdx, 48({dst})",                      // Store dst[6]
            "movq %r8, 56({dst})",                       // Store dst[7]
            "leaq 64({dst}), {dst}",                     // Advance dst by 64 bytes (preserves CF)
            "leaq 64({src}), {src}",                     // Advance src by 64 bytes (preserves CF)
            "decq {chunks}",                             // Decrement chunk counter (preserves CF)
            "jns 2b",                                    // Repeat while chunks >= 0

            // Tail: descending 4/2/1 blocks
            "3:",
            "decq {any_tail}",                           // Test if any remainder limbs remain (preserves CF)
            "js 6f",                                     // If none, jump directly to exit (6f)
            "decq {tail_4}",                             // Test 4-limb tail block
            "js 4f",                                     // Skip to 2-limb test if absent (4f)
            "movq ({dst}), %rax",                        // Load dst[0]
            "movq 8({dst}), %rcx",                       // Load dst[1]
            "sbbq ({src}), %rax",                        // %rax = dst[0] - src[0] - CF
            "sbbq 8({src}), %rcx",                       // %rcx = dst[1] - src[1] - CF
            "movq 16({dst}), %rdx",                      // Load dst[2]
            "movq 24({dst}), %r8",                       // Load dst[3]
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "sbbq 16({src}), %rdx",                      // %rdx = dst[2] - src[2] - CF
            "sbbq 24({src}), %r8",                       // %r8 = dst[3] - src[3] - CF
            "movq %rdx, 16({dst})",                      // Store dst[2]
            "movq %r8, 24({dst})",                       // Store dst[3]
            "leaq 32({dst}), {dst}",                     // Advance dst by 32 bytes (preserves CF)
            "leaq 32({src}), {src}",                     // Advance src by 32 bytes (preserves CF)

            "4:",
            "decq {tail_2}",                             // Test 2-limb tail block
            "js 5f",                                     // Skip to 1-limb test if absent (5f)
            "movq ({dst}), %rax",                        // Load dst[0]
            "movq 8({dst}), %rcx",                       // Load dst[1]
            "sbbq ({src}), %rax",                        // %rax = dst[0] - src[0] - CF
            "sbbq 8({src}), %rcx",                       // %rcx = dst[1] - src[1] - CF
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "leaq 16({dst}), {dst}",                     // Advance dst by 16 bytes (preserves CF)
            "leaq 16({src}), {src}",                     // Advance src by 16 bytes (preserves CF)

            "5:",
            "decq {tail_1}",                             // Test 1-limb tail block
            "js 6f",                                     // Skip if absent (6f)
            "movq ({dst}), %rax",                        // Load single dst limb
            "sbbq ({src}), %rax",                        // %rax = dst[0] - src[0] - CF
            "movq %rax, ({dst})",                        // Store single dst limb

            // Exit: capture final borrow
            "6:",
            "adcq {borrow}, {borrow}",                   // borrow = 0 + 0 + CF (0 or 1)
            borrow = out(reg) borrow,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
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

/// Straight-line `dst[i] = dst[i] - src[i] - borrow` chain for `len` in
/// `2..=4`.
///
/// # Safety
///
/// `2 <= len <= 4`. Both pointers must cover `len` aligned initialized limbs;
/// `dst` must be writable. Spans must be identical or disjoint.
#[expect(
    clippy::inline_always,
    reason = "The fixed-size borrow chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn sub_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut borrow: Limb;
            // SAFETY: len=2 selects two initialized aligned limbs in each span.
            // Source digits are loaded before writes, including exact alias.
            // SUB/SBB threads CF, and digit temporaries are early outputs.
            unsafe {
                asm!(
                    "xorl {borrow:e}, {borrow:e}",       // borrow = 0, clears CF
                    "movq ({src}), %rax",                // Load src[0]
                    "movq 8({src}), %rcx",               // Load src[1]
                    "subq %rax, ({dst})",                // dst[0] -= src[0], sets CF
                    "sbbq %rcx, 8({dst})",               // dst[1] -= src[1] + CF, sets CF
                    "adcq {borrow}, {borrow}",           // borrow = 0 + 0 + CF (extract borrow bit)
                    src = in(reg) src,
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
            // SAFETY: len=3 selects three initialized aligned limbs in each span.
            // Source digits are loaded before writes, including exact alias.
            // SUB/SBB threads CF, and digit temporaries are early outputs.
            unsafe {
                asm!(
                    "movq ({src}), %rax",                // Load src[0]
                    "movq 8({src}), %rcx",               // Load src[1]
                    "movq 16({src}), %rdx",              // Load src[2]
                    "xorl {borrow:e}, {borrow:e}",       // borrow = 0, clears CF
                    "subq %rax, ({dst})",                // dst[0] -= src[0], sets CF
                    "sbbq %rcx, 8({dst})",               // dst[1] -= src[1] + CF, sets CF
                    "sbbq %rdx, 16({dst})",              // dst[2] -= src[2] + CF, sets CF
                    "adcq {borrow}, {borrow}",           // borrow = 0 + 0 + CF (extract borrow bit)
                    src = in(reg) src,
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
            // SAFETY: len=4 selects four initialized aligned limbs in each span.
            // Source digits are loaded before writes, including exact alias.
            // SUB/SBB threads CF, and digit temporaries are early outputs.
            unsafe {
                asm!(
                    "xorl {borrow:e}, {borrow:e}",       // borrow = 0, clears CF
                    "movq ({src}), %rax",                // Load src[0]
                    "movq 8({src}), %rcx",               // Load src[1]
                    "movq 16({src}), %rdx",              // Load src[2]
                    "movq 24({src}), %r8",               // Load src[3]
                    "subq %rax, ({dst})",                // dst[0] -= src[0], sets CF
                    "sbbq %rcx, 8({dst})",               // dst[1] -= src[1] + CF, sets CF
                    "sbbq %rdx, 16({dst})",              // dst[2] -= src[2] + CF, sets CF
                    "sbbq %r8, 24({dst})",               // dst[3] -= src[3] + CF, sets CF
                    "adcq {borrow}, {borrow}",           // borrow = 0 + 0 + CF (extract borrow bit)
                    src = in(reg) src,
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
