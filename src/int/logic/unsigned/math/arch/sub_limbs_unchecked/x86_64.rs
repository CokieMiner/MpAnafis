//! `x86_64` baseline (non-ADX) subtraction kernels (inline assembly).
//!
//! Evaluates `dst -= src` using optional 4-limb prefix, 8-way unrolled `sbbq` loops, and CF-preserving indexed addressing.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` and return the final borrow.
///
/// Computes:
///
/// ```text
///   (borrow, dst[0..len]) = dst[0..len] − src[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// Small inputs (2..=4 limbs) execute straight-line `subq`/`sbbq` chains without branching.
/// Larger slices process an optional 4-limb prefix followed by an 8-way unrolled `sbbq` loop with scaled indexing.
///
/// # Safety
///
/// `dst` and `src` must each be valid for `len` elements of type `Limb`.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    if (2..=4).contains(&len) {
        // SAFETY: the caller guarantees both pointers cover `len` limbs, and
        // this branch proves the fixed kernel's `2..=4` length precondition.
        return unsafe { sub_small_unchecked(dst, src, len) };
    }
    let mut borrow: Limb;
    let prefix = (len >> 2) & 1;
    let chunks = len >> 3;
    let rem = len & 3;
    let idx = 0_usize;

    // SAFETY:
    // 1. `dst` and `src` are valid for `len` 64-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "xorl {borrow:e}, {borrow:e}",                // CF = 0 (no initial borrow), zero borrow register
            // Optional 4-limb prefix block
            "decq {prefix}",                             // Decrement prefix flag (preserves CF)
            "js 1f",                                     // If prefix == 0, jump to 8-way loop (1f)
            "movq ({src}, {idx}, 8), %rax",              // Load src[0]
            "movq 8({src}, {idx}, 8), %rcx",             // Load src[1]
            "movq 16({src}, {idx}, 8), %rdx",            // Load src[2]
            "movq 24({src}, {idx}, 8), %r8",             // Load src[3]
            "sbbq %rax, ({dst}, {idx}, 8)",              // dst[0] -= src[0] + CF
            "sbbq %rcx, 8({dst}, {idx}, 8)",             // dst[1] -= src[1] + CF
            "sbbq %rdx, 16({dst}, {idx}, 8)",            // dst[2] -= src[2] + CF
            "sbbq %r8, 24({dst}, {idx}, 8)",             // dst[3] -= src[3] + CF
            "leaq 4({idx}), {idx}",                      // Advance idx by 4 (preserves CF)

            // Main 8-way unrolled loop
            "1:",
            "decq {chunks}",                             // Pre-decrement chunk counter (preserves CF)
            "js 3f",                                     // If chunks < 0, jump to remainder (3f)
            "2:",
            "movq ({src}, {idx}, 8), %rax",              // Load src[0]
            "movq 8({src}, {idx}, 8), %rcx",             // Load src[1]
            "movq 16({src}, {idx}, 8), %rdx",            // Load src[2]
            "movq 24({src}, {idx}, 8), %r8",             // Load src[3]
            "sbbq %rax, ({dst}, {idx}, 8)",              // dst[0] -= src[0] + CF
            "sbbq %rcx, 8({dst}, {idx}, 8)",             // dst[1] -= src[1] + CF
            "sbbq %rdx, 16({dst}, {idx}, 8)",            // dst[2] -= src[2] + CF
            "sbbq %r8, 24({dst}, {idx}, 8)",             // dst[3] -= src[3] + CF
            "movq 32({src}, {idx}, 8), %rax",            // Load src[4]
            "movq 40({src}, {idx}, 8), %rcx",            // Load src[5]
            "movq 48({src}, {idx}, 8), %rdx",            // Load src[6]
            "movq 56({src}, {idx}, 8), %r8",             // Load src[7]
            "sbbq %rax, 32({dst}, {idx}, 8)",            // dst[4] -= src[4] + CF
            "sbbq %rcx, 40({dst}, {idx}, 8)",            // dst[5] -= src[5] + CF
            "sbbq %rdx, 48({dst}, {idx}, 8)",            // dst[6] -= src[6] + CF
            "sbbq %r8, 56({dst}, {idx}, 8)",             // dst[7] -= src[7] + CF
            "leaq 8({idx}), {idx}",                      // Advance idx by 8 (preserves CF)
            "decq {chunks}",                             // Decrement chunks (preserves CF)
            "jns 2b",                                    // Repeat while chunks >= 0

            // Remainder entry point (0 to 3 limbs)
            "3:",
            "decq {rem}",                                // Pre-decrement remainder counter (preserves CF)
            "js 5f",                                     // If rem < 0, skip to exit (5f)
            "4:",
            "movq ({src}, {idx}, 8), %rax",              // Load single src limb
            "sbbq %rax, ({dst}, {idx}, 8)",              // dst -= src + CF
            "leaq 1({idx}), {idx}",                      // Advance idx by 1 (preserves CF)
            "decq {rem}",                                // Decrement remainder (preserves CF)
            "jns 4b",                                    // Repeat while rem >= 0

            // Exit: capture final borrow
            "5:",
            "adcq {borrow}, {borrow}",                   // borrow = 0 + 0 + CF (0 or 1)
            borrow = out(reg) borrow,
            idx = inout(reg) idx => _,
            dst = in(reg) dst,
            src = in(reg) src,
            prefix = inout(reg) prefix => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
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
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint.
#[allow(
    clippy::inline_always,
    reason = "The fixed-size borrow chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn sub_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut borrow: Limb;
            // SAFETY: The caller guarantees `dst` and `src` are valid for 2 limbs.
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
            // SAFETY: The caller guarantees `dst` and `src` are valid for 3 limbs.
            unsafe {
                asm!(
                    "xorl {borrow:e}, {borrow:e}",       // borrow = 0, clears CF
                    "movq ({src}), %rax",                // Load src[0]
                    "movq 8({src}), %rcx",               // Load src[1]
                    "movq 16({src}), %rdx",              // Load src[2]
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
            // SAFETY: The caller guarantees `dst` and `src` are valid for 4 limbs.
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
