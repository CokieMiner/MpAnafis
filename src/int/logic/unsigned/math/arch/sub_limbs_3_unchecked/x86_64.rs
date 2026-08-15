//! `x86_64` baseline (non-ADX) subtraction kernels (inline assembly).
//!
//! Evaluates `dst = src1 - src2` using 4-limb prefix, 8-way unrolled `sbbq` loops, and CF-preserving indexed addressing.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Compute `dst[i] = src1[i] − src2[i] − borrow` for `len` limbs,
/// returning the final borrow.
///
/// Computes:
///
/// ```text
///   (borrow, dst[0..len]) = src1[0..len] - src2[0..len]
/// ```
///
/// # Microarchitectural Strategy
///
/// Small inputs (2..=4 limbs) execute straight-line `subq`/`sbbq` chains without branching.
/// Larger slices process an optional 4-limb prefix followed by an 8-way unrolled `sbbq` loop with scaled indexing.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
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
    let prefix = (len >> 2) & 1;
    let chunks = len >> 3;
    let rem = len & 3;
    let idx = 0_usize;

    // SAFETY:
    // 1. `dst`, `src1`, and `src2` are valid for `len` 64-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "xorl {borrow:e}, {borrow:e}",                // CF = 0 (no initial borrow), zero borrow register
            // Optional 4-limb prefix block
            "decq {prefix}",                             // Decrement prefix flag (preserves CF)
            "js 1f",                                     // If prefix == 0, jump to 8-way loop (1f)
            "movq ({src1}, {idx}, 8), %rax",             // Load src1[0]
            "movq 8({src1}, {idx}, 8), %rcx",            // Load src1[1]
            "movq 16({src1}, {idx}, 8), %rdx",           // Load src1[2]
            "movq 24({src1}, {idx}, 8), %r8",            // Load src1[3]
            "sbbq ({src2}, {idx}, 8), %rax",             // %rax -= src2[0] + CF
            "sbbq 8({src2}, {idx}, 8), %rcx",            // %rcx -= src2[1] + CF
            "sbbq 16({src2}, {idx}, 8), %rdx",           // %rdx -= src2[2] + CF
            "sbbq 24({src2}, {idx}, 8), %r8",            // %r8 -= src2[3] + CF
            "movq %rax, ({dst}, {idx}, 8)",              // Store dst[0]
            "movq %rcx, 8({dst}, {idx}, 8)",             // Store dst[1]
            "movq %rdx, 16({dst}, {idx}, 8)",            // Store dst[2]
            "movq %r8, 24({dst}, {idx}, 8)",             // Store dst[3]
            "leaq 4({idx}), {idx}",                      // Advance idx by 4 (preserves CF)

            // Main 8-way unrolled loop
            "1:",
            "decq {chunks}",                             // Pre-decrement chunk counter (preserves CF)
            "js 3f",                                     // If chunks < 0, jump to remainder (3f)
            "2:",
            "movq ({src1}, {idx}, 8), %rax",             // Load first 4 limbs of src1
            "movq 8({src1}, {idx}, 8), %rcx",            // Load src1[1]
            "movq 16({src1}, {idx}, 8), %rdx",           // Load src1[2]
            "movq 24({src1}, {idx}, 8), %r8",            // Load src1[3]
            "sbbq ({src2}, {idx}, 8), %rax",             // Subtract src2[0] + CF
            "sbbq 8({src2}, {idx}, 8), %rcx",            // Subtract src2[1] + CF
            "sbbq 16({src2}, {idx}, 8), %rdx",           // Subtract src2[2] + CF
            "sbbq 24({src2}, {idx}, 8), %r8",            // Subtract src2[3] + CF
            "movq %rax, ({dst}, {idx}, 8)",              // Store dst[0]
            "movq %rcx, 8({dst}, {idx}, 8)",             // Store dst[1]
            "movq %rdx, 16({dst}, {idx}, 8)",            // Store dst[2]
            "movq %r8, 24({dst}, {idx}, 8)",             // Store dst[3]
            "movq 32({src1}, {idx}, 8), %rax",           // Load next 4 limbs of src1
            "movq 40({src1}, {idx}, 8), %rcx",           // Load src1[5]
            "movq 48({src1}, {idx}, 8), %rdx",           // Load src1[6]
            "movq 56({src1}, {idx}, 8), %r8",            // Load src1[7]
            "sbbq 32({src2}, {idx}, 8), %rax",           // Subtract src2[4] + CF
            "sbbq 40({src2}, {idx}, 8), %rcx",           // Subtract src2[5] + CF
            "sbbq 48({src2}, {idx}, 8), %rdx",           // Subtract src2[6] + CF
            "sbbq 56({src2}, {idx}, 8), %r8",            // Subtract src2[7] + CF
            "movq %rax, 32({dst}, {idx}, 8)",            // Store dst[4]
            "movq %rcx, 40({dst}, {idx}, 8)",            // Store dst[5]
            "movq %rdx, 48({dst}, {idx}, 8)",            // Store dst[6]
            "movq %r8, 56({dst}, {idx}, 8)",             // Store dst[7]
            "leaq 8({idx}), {idx}",                      // Advance idx by 8 (preserves CF)
            "decq {chunks}",                             // Decrement chunks (preserves CF)
            "jns 2b",                                    // Repeat while chunks >= 0

            // Remainder entry point (0 to 3 limbs)
            "3:",
            "decq {rem}",                                // Pre-decrement remainder counter (preserves CF)
            "js 5f",                                     // If rem < 0, skip to exit (5f)
            "4:",
            "movq ({src1}, {idx}, 8), %rax",             // Load single src1 limb
            "sbbq ({src2}, {idx}, 8), %rax",             // %rax -= src2 limb + CF
            "movq %rax, ({dst}, {idx}, 8)",              // Store single dst limb
            "leaq 1({idx}), {idx}",                      // Advance idx by 1 (preserves CF)
            "decq {rem}",                                // Decrement remainder (preserves CF)
            "jns 4b",                                    // Repeat while rem >= 0

            // Exit: capture final borrow
            "5:",
            "adcq {borrow}, {borrow}",                   // borrow = 0 + 0 + CF (0 or 1)
            borrow = out(reg) borrow,
            idx = inout(reg) idx => _,
            dst = in(reg) dst,
            src1 = in(reg) src1,
            src2 = in(reg) src2,
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

/// Straight-line `dst[i] = src1[i] - src2[i] - borrow` chain for `len` in `2..=4`.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span.
#[allow(
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
            // SAFETY: The caller guarantees `dst`, `src1`, and `src2` are valid for 2 limbs.
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
            // SAFETY: The caller guarantees `dst`, `src1`, and `src2` are valid for 3 limbs.
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
            // SAFETY: The caller guarantees `dst`, `src1`, and `src2` are valid for 4 limbs.
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
