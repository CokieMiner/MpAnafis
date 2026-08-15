//! x86‑64 (AMD64) `add_n` kernel.
//!
//! Evaluates `dst += src` using 8-way unrolled `adcq` loops from index zero and a 4/2/1 descending tail.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Add `len` limbs from `src` into `dst` and return the final carry.
///
/// Computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + src[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// 8-way unrolled loop executes from index zero to preserve 64-byte destination alignment and avoid cache straddling.
/// Remaining limbs (len & 7) are resolved via binary-descending 4/2/1 blocks with zero-overhead `decq` tests.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    // SAFETY: The caller guarantees both pointers cover `len` elements.
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
    let idx = 0_usize;

    // SAFETY:
    // 1. `dst` and `src` are valid for `len` 64-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "xorl {carry:e}, {carry:e}",                  // carry = 0, also clears CF
            // Main 8-way unrolled loop
            "decq {chunks}",                             // Pre-decrement chunk counter (preserves CF)
            "js 3f",                                     // If chunks < 0, skip main loop (3f)
            "2:",
            "movq ({src}, {idx}, 8), %rax",              // Load src[0]
            "movq 8({src}, {idx}, 8), %rcx",             // Load src[1]
            "movq 16({src}, {idx}, 8), %rdx",            // Load src[2]
            "movq 24({src}, {idx}, 8), %r8",             // Load src[3]
            "adcq %rax, ({dst}, {idx}, 8)",              // dst[0] += src[0] + CF
            "adcq %rcx, 8({dst}, {idx}, 8)",             // dst[1] += src[1] + CF
            "adcq %rdx, 16({dst}, {idx}, 8)",            // dst[2] += src[2] + CF
            "adcq %r8, 24({dst}, {idx}, 8)",             // dst[3] += src[3] + CF
            "movq 32({src}, {idx}, 8), %rax",            // Load src[4]
            "movq 40({src}, {idx}, 8), %rcx",            // Load src[5]
            "movq 48({src}, {idx}, 8), %rdx",            // Load src[6]
            "movq 56({src}, {idx}, 8), %r8",             // Load src[7]
            "adcq %rax, 32({dst}, {idx}, 8)",            // dst[4] += src[4] + CF
            "adcq %rcx, 40({dst}, {idx}, 8)",            // dst[5] += src[5] + CF
            "adcq %rdx, 48({dst}, {idx}, 8)",            // dst[6] += src[6] + CF
            "adcq %r8, 56({dst}, {idx}, 8)",             // dst[7] += src[7] + CF
            "leaq 8({idx}), {idx}",                      // Advance idx by 8 (preserves CF)
            "decq {chunks}",                             // Decrement chunk counter (preserves CF)
            "jns 2b",                                    // Repeat while chunks >= 0

            // Tail: descending 4/2/1 blocks
            "3:",
            "decq {any_tail}",                           // Test if any remainder limbs remain (preserves CF)
            "js 6f",                                     // If none, jump directly to exit (6f)
            "decq {tail_4}",                             // Test 4-limb tail block
            "js 4f",                                     // Skip to 2-limb test if absent (4f)
            "movq ({src}, {idx}, 8), %rax",              // Load src[0]
            "movq 8({src}, {idx}, 8), %rcx",             // Load src[1]
            "movq 16({src}, {idx}, 8), %rdx",            // Load src[2]
            "movq 24({src}, {idx}, 8), %r8",             // Load src[3]
            "adcq %rax, ({dst}, {idx}, 8)",              // dst[0] += src[0] + CF
            "adcq %rcx, 8({dst}, {idx}, 8)",             // dst[1] += src[1] + CF
            "adcq %rdx, 16({dst}, {idx}, 8)",            // dst[2] += src[2] + CF
            "adcq %r8, 24({dst}, {idx}, 8)",             // dst[3] += src[3] + CF
            "leaq 4({idx}), {idx}",                      // Advance idx by 4 (preserves CF)

            "4:",
            "decq {tail_2}",                             // Test 2-limb tail block
            "js 5f",                                     // Skip to 1-limb test if absent (5f)
            "movq ({src}, {idx}, 8), %rax",              // Load src[0]
            "movq 8({src}, {idx}, 8), %rcx",             // Load src[1]
            "adcq %rax, ({dst}, {idx}, 8)",              // dst[0] += src[0] + CF
            "adcq %rcx, 8({dst}, {idx}, 8)",             // dst[1] += src[1] + CF
            "leaq 2({idx}), {idx}",                      // Advance idx by 2 (preserves CF)

            "5:",
            "decq {tail_1}",                             // Test 1-limb tail block
            "js 6f",                                     // Skip if absent (6f)
            "movq ({src}, {idx}, 8), %rax",              // Load single src limb
            "adcq %rax, ({dst}, {idx}, 8)",              // dst += src + CF

            // Exit: extract final carry
            "6:",
            "adcq {carry}, {carry}",                     // carry = 0 + 0 + CF (extract final carry)
            carry = out(reg) carry,
            idx = inout(reg) idx => _,
            dst = in(reg) dst,
            src = in(reg) src,
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
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint.
#[allow(
    clippy::inline_always,
    reason = "The fixed-size carry chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: The caller guarantees `dst` and `src` are valid for 2 limbs.
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
            // SAFETY: The caller guarantees `dst` and `src` are valid for 3 limbs.
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
            // SAFETY: The caller guarantees `dst` and `src` are valid for 4 limbs.
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
