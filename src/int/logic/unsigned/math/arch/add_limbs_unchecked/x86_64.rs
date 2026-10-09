//! x86-64 in-place addition with a hardware carry chain.
//!
//! Eight-limb blocks preserve each pointer's initial alignment modulo 64.
//! Remaining limbs use descending four-, two-, and one-limb blocks.
//! `movq`, `leaq`, `decq`, and branches preserve CF between additions.

use core::{arch::asm, hint::unreachable_unchecked};

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
#[expect(
    clippy::too_many_lines,
    reason = "Explicit assembly sequence unrolled across 8 limbs and 4/2/1 tail blocks"
)]
#[inline(always)]
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    if len == 0 {
        return 0;
    }
    if len == 1 {
        // SAFETY: the caller guarantees both pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*dst).overflowing_add(*src) };
        // SAFETY: the caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if (2..=4).contains(&len) {
        // SAFETY: the caller guarantees both pointers cover `len` limbs, and
        // this branch proves the fixed kernel's `2..=4` length precondition.
        return unsafe { add_small_unchecked(dst, src, len) };
    }
    let mut carry: Limb;
    let chunks = len >> 3;
    let any_tail = len & 7;
    let tail_4 = (len >> 2) & 1;
    let tail_2 = (len >> 1) & 1;
    let tail_1 = len & 1;

    // SAFETY: 8 * chunks + 4 * tail_4 + 2 * tail_2 + tail_1 == len bounds
    // every aligned, initialized access. Counts fit signed registers because
    // spans have at most isize::MAX bytes. Each input pair is read before its
    // result is written, permitting exact aliasing. Clobbers are declared.
    unsafe {
        asm!(
            "xorl {carry:e}, {carry:e}",                  // carry = 0, also clears CF
            // Main 8-way unrolled loop
            "decq {chunks}",                             // Pre-decrement chunk counter (preserves CF)
            "js 3f",                                     // If chunks < 0, skip main loop (3f)
            "2:",
            "movq ({dst}), %rax",                        // Load dst[0]
            "movq 8({dst}), %rcx",                       // Load dst[1]
            "adcq ({src}), %rax",                        // %rax = dst[0] + src[0] + CF
            "adcq 8({src}), %rcx",                       // %rcx = dst[1] + src[1] + CF
            "movq 16({dst}), %rdx",                      // Load dst[2]
            "movq 24({dst}), %r8",                       // Load dst[3]
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "adcq 16({src}), %rdx",                      // %rdx = dst[2] + src[2] + CF
            "adcq 24({src}), %r8",                       // %r8 = dst[3] + src[3] + CF
            "movq 32({dst}), %rax",                      // Load dst[4]
            "movq 40({dst}), %rcx",                      // Load dst[5]
            "movq %rdx, 16({dst})",                      // Store dst[2]
            "movq %r8, 24({dst})",                       // Store dst[3]
            "adcq 32({src}), %rax",                      // %rax = dst[4] + src[4] + CF
            "adcq 40({src}), %rcx",                      // %rcx = dst[5] + src[5] + CF
            "movq 48({dst}), %rdx",                      // Load dst[6]
            "movq 56({dst}), %r8",                       // Load dst[7]
            "movq %rax, 32({dst})",                      // Store dst[4]
            "movq %rcx, 40({dst})",                      // Store dst[5]
            "adcq 48({src}), %rdx",                      // %rdx = dst[6] + src[6] + CF
            "adcq 56({src}), %r8",                       // %r8 = dst[7] + src[7] + CF
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
            "adcq ({src}), %rax",                        // %rax = dst[0] + src[0] + CF
            "adcq 8({src}), %rcx",                       // %rcx = dst[1] + src[1] + CF
            "movq 16({dst}), %rdx",                      // Load dst[2]
            "movq 24({dst}), %r8",                       // Load dst[3]
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "adcq 16({src}), %rdx",                      // %rdx = dst[2] + src[2] + CF
            "adcq 24({src}), %r8",                       // %r8 = dst[3] + src[3] + CF
            "movq %rdx, 16({dst})",                      // Store dst[2]
            "movq %r8, 24({dst})",                       // Store dst[3]
            "leaq 32({dst}), {dst}",                     // Advance dst by 32 bytes (preserves CF)
            "leaq 32({src}), {src}",                     // Advance src by 32 bytes (preserves CF)

            "4:",
            "decq {tail_2}",                             // Test 2-limb tail block
            "js 5f",                                     // Skip to 1-limb test if absent (5f)
            "movq ({dst}), %rax",                        // Load dst[0]
            "movq 8({dst}), %rcx",                       // Load dst[1]
            "adcq ({src}), %rax",                        // %rax = dst[0] + src[0] + CF
            "adcq 8({src}), %rcx",                       // %rcx = dst[1] + src[1] + CF
            "movq %rax, ({dst})",                        // Store dst[0]
            "movq %rcx, 8({dst})",                       // Store dst[1]
            "leaq 16({dst}), {dst}",                     // Advance dst by 16 bytes (preserves CF)
            "leaq 16({src}), {src}",                     // Advance src by 16 bytes (preserves CF)

            "5:",
            "decq {tail_1}",                             // Test 1-limb tail block
            "js 6f",                                     // Skip if absent (6f)
            "movq ({dst}), %rax",                        // Load single dst limb
            "adcq ({src}), %rax",                        // %rax = dst[0] + src[0] + CF
            "movq %rax, ({dst})",                        // Store single dst limb

            // Exit: extract final carry
            "6:",
            "adcq {carry}, {carry}",                     // carry = 0 + 0 + CF (extract final carry)
            carry = out(reg) carry,
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
    carry
}

/// Straight-line `dst[i] = dst[i] + src[i] + carry` chain for `len` in `2..=4`.
///
/// # Safety
///
/// The pointer contract of [`add_limbs_unchecked`] applies and `2 <= len <= 4`.
#[expect(
    clippy::inline_always,
    reason = "The fixed-size carry chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: len == 2 bounds both initialized spans. Source loads
            // precede destination writes, permitting exact aliasing.
            unsafe {
                asm!(
                    "xorl {carry:e}, {carry:e}",         // carry = 0, clears CF
                    "movq ({src}), %rax",                // Load src[0]
                    "movq 8({src}), %rcx",               // Load src[1]
                    "addq %rax, ({dst})",                // dst[0] += src[0], sets CF
                    "adcq %rcx, 8({dst})",               // dst[1] += src[1] + CF, sets CF
                    "adcq {carry}, {carry}",             // carry = 0 + 0 + CF (extract carry out)
                    src = in(reg) src,
                    dst = in(reg) dst,
                    carry = out(reg) carry,
                    out("rax") _,
                    out("rcx") _,
                    options(nostack, att_syntax)
                );
            }
            carry
        }
        3 => {
            let mut carry: Limb;
            // SAFETY: len == 3 bounds both initialized spans. Source loads
            // precede destination writes, permitting exact aliasing.
            unsafe {
                asm!(
                    "xorl {carry:e}, {carry:e}",         // carry = 0, clears CF
                    "movq ({src}), %rax",                // Load src[0]
                    "movq 8({src}), %rcx",               // Load src[1]
                    "movq 16({src}), %rdx",              // Load src[2]
                    "addq %rax, ({dst})",                // dst[0] += src[0], sets CF
                    "adcq %rcx, 8({dst})",               // dst[1] += src[1] + CF, sets CF
                    "adcq %rdx, 16({dst})",              // dst[2] += src[2] + CF, sets CF
                    "adcq {carry}, {carry}",             // carry = 0 + 0 + CF (extract carry out)
                    src = in(reg) src,
                    dst = in(reg) dst,
                    carry = out(reg) carry,
                    out("rax") _,
                    out("rcx") _,
                    out("rdx") _,
                    options(nostack, att_syntax)
                );
            }
            carry
        }
        4 => {
            let mut carry: Limb;
            // SAFETY: len == 4 bounds both initialized spans. Source loads
            // precede destination writes, permitting exact aliasing.
            unsafe {
                asm!(
                    "xorl {carry:e}, {carry:e}",         // carry = 0, clears CF
                    "movq ({src}), %rax",                // Load src[0]
                    "movq 8({src}), %rcx",               // Load src[1]
                    "movq 16({src}), %rdx",              // Load src[2]
                    "movq 24({src}), %r8",               // Load src[3]
                    "addq %rax, ({dst})",                // dst[0] += src[0], sets CF
                    "adcq %rcx, 8({dst})",               // dst[1] += src[1] + CF, sets CF
                    "adcq %rdx, 16({dst})",              // dst[2] += src[2] + CF, sets CF
                    "adcq %r8, 24({dst})",               // dst[3] += src[3] + CF, sets CF
                    "adcq {carry}, {carry}",             // carry = 0 + 0 + CF (extract carry out)
                    src = in(reg) src,
                    dst = in(reg) dst,
                    carry = out(reg) carry,
                    out("rax") _,
                    out("rcx") _,
                    out("rdx") _,
                    out("r8") _,
                    options(nostack, att_syntax)
                );
            }
            carry
        }
        // SAFETY: The caller guarantees `2 <= len <= 4`.
        _ => unsafe { unreachable_unchecked() },
    }
}
