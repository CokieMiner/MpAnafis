//! `x86_64` baseline (non-ADX) subtraction kernels (inline assembly).
//!
//! Provides its own 3-operand path using `sbb` for borrow tracking and `adcq`
//! for borrow extraction.
//!
//! ## Loop structure
//!
//! 4-way unrolled (`len >> 2`) with a single-limb tail.  `decq`/`jns`
//! iteration counters correctly handle `len == 0`.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

// ── sub_n_3 (3-operand) ───────────────────────────────────────────────────

/// Compute `dst[i] = src1[i] − src2[i] − borrow` for `len` limbs,
/// returning the final borrow.
///
/// # Safety
///
/// `dst`, `src1`, and `src2` must each be valid for `len` elements.
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
    // SAFETY: Caller guarantees dst, src1, src2 each hold at least `len` valid Limb values.
    unsafe {
        asm!(
            "xorl {borrow:e}, {borrow:e}",        // CF = 0 (no initial borrow)
            // ── Optional 4-limb prefix ────────────────────────────────
            "decq {prefix}",
            "js 1f",
            "movq ({src1}, {idx}, 8), %rax",
            "movq 8({src1}, {idx}, 8), %rcx",
            "movq 16({src1}, {idx}, 8), %rdx",
            "movq 24({src1}, {idx}, 8), %r8",
            "sbbq ({src2}, {idx}, 8), %rax",
            "sbbq 8({src2}, {idx}, 8), %rcx",
            "sbbq 16({src2}, {idx}, 8), %rdx",
            "sbbq 24({src2}, {idx}, 8), %r8",
            "movq %rax, ({dst}, {idx}, 8)",
            "movq %rcx, 8({dst}, {idx}, 8)",
            "movq %rdx, 16({dst}, {idx}, 8)",
            "movq %r8, 24({dst}, {idx}, 8)",
            "leaq 4({idx}), {idx}",
            // ── 8-way unrolled loop (do-while with decq/jns) ───────────
            "1:",
            "decq {chunks}",                      // decrement: jump if -1 (chunks == 0)
            "js 3f",                              // skip main loop if chunks == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "movq ({src1}, {idx}, 8), %rax",       // load first 4 limbs of src1
            "movq 8({src1}, {idx}, 8), %rcx",
            "movq 16({src1}, {idx}, 8), %rdx",
            "movq 24({src1}, {idx}, 8), %r8",
            "sbbq ({src2}, {idx}, 8), %rax",       // subtract first 4 limbs of src2 + CF
            "sbbq 8({src2}, {idx}, 8), %rcx",
            "sbbq 16({src2}, {idx}, 8), %rdx",
            "sbbq 24({src2}, {idx}, 8), %r8",
            "movq %rax, ({dst}, {idx}, 8)",        // store first 4 limbs to dst
            "movq %rcx, 8({dst}, {idx}, 8)",
            "movq %rdx, 16({dst}, {idx}, 8)",
            "movq %r8, 24({dst}, {idx}, 8)",

            "movq 32({src1}, {idx}, 8), %rax",     // load next 4 limbs of src1
            "movq 40({src1}, {idx}, 8), %rcx",
            "movq 48({src1}, {idx}, 8), %rdx",
            "movq 56({src1}, {idx}, 8), %r8",
            "sbbq 32({src2}, {idx}, 8), %rax",     // subtract next 4 limbs of src2 + CF
            "sbbq 40({src2}, {idx}, 8), %rcx",
            "sbbq 48({src2}, {idx}, 8), %rdx",
            "sbbq 56({src2}, {idx}, 8), %r8",
            "movq %rax, 32({dst}, {idx}, 8)",      // store next 4 limbs to dst
            "movq %rcx, 40({dst}, {idx}, 8)",
            "movq %rdx, 48({dst}, {idx}, 8)",
            "movq %r8, 56({dst}, {idx}, 8)",

            "leaq 8({idx}), {idx}",                // advance idx by 8
            "decq {chunks}",                       // decrement chunk counter
            "jns 2b",                              // loop back if chunks >= 0
            // ── Tail: single‑limb remainder loop ───────────────────────
            "3:",
            "decq {rem}",                          // decrement: jump if -1 (rem == 0)
            "js 5f",                               // skip tail if rem == 0
            ".p2align 4",                           // align tail loop header
            "4:",
            "movq ({src1}, {idx}, 8), %rax",       // load src1 limb
            "sbbq ({src2}, {idx}, 8), %rax",       // subtract src2 limb + CF
            "movq %rax, ({dst}, {idx}, 8)",        // store to dst

            "leaq 1({idx}), {idx}",                // advance idx by 1
            "decq {rem}",                          // decrement remainder counter
            "jns 4b",                              // loop back if rem >= 0
            "5:",
            "adcq {borrow}, {borrow}",             // borrow = CF
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

/// Straight-line `dst[i] = src1[i] - src2[i] - borrow` chain for `len` in
/// `2..=4`.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span: the kernel writes `dst`
///   while it reads `src1` and `src2`.
/// - `src1` and `src2` are read-only and may alias each other.
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
                    "xorl {borrow:e}, {borrow:e}",
                    "movq ({src1}), %rax",
                    "movq 8({src1}), %rcx",
                    "subq ({src2}), %rax",
                    "sbbq 8({src2}), %rcx",
                    "movq %rax, ({dst})",
                    "movq %rcx, 8({dst})",
                    "adcq {borrow}, {borrow}",
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
                    "xorl {borrow:e}, {borrow:e}",
                    "movq ({src1}), %rax",
                    "movq 8({src1}), %rcx",
                    "movq 16({src1}), %rdx",
                    "subq ({src2}), %rax",
                    "sbbq 8({src2}), %rcx",
                    "sbbq 16({src2}), %rdx",
                    "movq %rax, ({dst})",
                    "movq %rcx, 8({dst})",
                    "movq %rdx, 16({dst})",
                    "adcq {borrow}, {borrow}",
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
                    "xorl {borrow:e}, {borrow:e}",
                    "movq ({src1}), %rax",
                    "movq 8({src1}), %rcx",
                    "movq 16({src1}), %rdx",
                    "movq 24({src1}), %r8",
                    "subq ({src2}), %rax",
                    "sbbq 8({src2}), %rcx",
                    "sbbq 16({src2}), %rdx",
                    "sbbq 24({src2}), %r8",
                    "movq %rax, ({dst})",
                    "movq %rcx, 8({dst})",
                    "movq %rdx, 16({dst})",
                    "movq %r8, 24({dst})",
                    "adcq {borrow}, {borrow}",
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
