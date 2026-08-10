//! `x86_64` baseline (non-ADX) subtraction kernels (inline assembly).
//!
//! Re-uses `sub_limbs_unchecked` from [`super::x86_64_shared`] and provides
//! its own 3-operand path using `sbb` for borrow tracking and `adcq` for
//! borrow extraction.
//!
//! ## Loop structure
//!
//! 4-way unrolled (`len >> 2`) with a single-limb tail.  `decq`/`jns`
//! iteration counters correctly handle `len == 0`.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` and return the final borrow.
///
/// ```text
///   (borrow, dst[0..len]) = dst[0..len] − src[0..len]
/// ```
///
/// # Safety
///
/// `dst` and `src` must each be valid for `len` elements of type `u64`.
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
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    unsafe {
        asm!(
            "xorl {borrow:e}, {borrow:e}",        // CF = 0 (no initial borrow)
            // ── Optional 4-limb prefix ────────────────────────────────
            "decq {prefix}",
            "js 1f",
            "movq ({src}, {idx}, 8), %rax",
            "movq 8({src}, {idx}, 8), %rcx",
            "movq 16({src}, {idx}, 8), %rdx",
            "movq 24({src}, {idx}, 8), %r8",
            "sbbq %rax, ({dst}, {idx}, 8)",
            "sbbq %rcx, 8({dst}, {idx}, 8)",
            "sbbq %rdx, 16({dst}, {idx}, 8)",
            "sbbq %r8, 24({dst}, {idx}, 8)",
            "leaq 4({idx}), {idx}",
            // ── 8-way unrolled loop (do-while with decq/jns) ───────────
            "1:",
            "decq {chunks}",                      // decrement: jump if -1 (chunks == 0)
            "js 3f",                              // skip main loop if chunks == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "movq ({src}, {idx}, 8), %rax",      // load first 4 limbs of src
            "movq 8({src}, {idx}, 8), %rcx",
            "movq 16({src}, {idx}, 8), %rdx",
            "movq 24({src}, {idx}, 8), %r8",
            "sbbq %rax, ({dst}, {idx}, 8)",      // subtract first 4 limbs from dst + CF
            "sbbq %rcx, 8({dst}, {idx}, 8)",
            "sbbq %rdx, 16({dst}, {idx}, 8)",
            "sbbq %r8, 24({dst}, {idx}, 8)",

            "movq 32({src}, {idx}, 8), %rax",    // load next 4 limbs of src
            "movq 40({src}, {idx}, 8), %rcx",
            "movq 48({src}, {idx}, 8), %rdx",
            "movq 56({src}, {idx}, 8), %r8",
            "sbbq %rax, 32({dst}, {idx}, 8)",    // subtract next 4 limbs from dst + CF
            "sbbq %rcx, 40({dst}, {idx}, 8)",
            "sbbq %rdx, 48({dst}, {idx}, 8)",
            "sbbq %r8, 56({dst}, {idx}, 8)",

            "leaq 8({idx}), {idx}",              // advance idx by 8
            "decq {chunks}",                      // decrement chunk counter
            "jns 2b",                             // loop back if chunks >= 0
            // ── Tail: single‑limb remainder loop ───────────────────────
            "3:",
            "decq {rem}",                         // decrement: jump if -1 (rem == 0)
            "js 5f",                              // skip tail if rem == 0
            ".p2align 4",                          // align tail loop header
            "4:",
            "movq ({src}, {idx}, 8), %rax",      // load src limb
            "sbbq %rax, ({dst}, {idx}, 8)",      // dst -= src + CF

            "leaq 1({idx}), {idx}",              // advance idx by 1
            "decq {rem}",                         // decrement remainder counter
            "jns 4b",                             // loop back if rem >= 0
            "5:",
            "adcq {borrow}, {borrow}",            // borrow = CF
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
/// - The `dst` and `src` spans must be identical or disjoint: the kernel
///   reads `src[i]` and `dst[i]` and then writes `dst[i]`, so a partial
///   overlap is a data race.
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
                        "xorl {borrow:e}, {borrow:e}",
                        "movq ({src}), %rax",
                        "movq 8({src}), %rcx",
                        "subq %rax, ({dst})",
                        "sbbq %rcx, 8({dst})",
                        "adcq {borrow}, {borrow}",
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
                        "xorl {borrow:e}, {borrow:e}",
                        "movq ({src}), %rax",
                        "movq 8({src}), %rcx",
                        "movq 16({src}), %rdx",
                        "subq %rax, ({dst})",
                        "sbbq %rcx, 8({dst})",
                        "sbbq %rdx, 16({dst})",
                        "adcq {borrow}, {borrow}",
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
                        "xorl {borrow:e}, {borrow:e}",
                        "movq ({src}), %rax",
                        "movq 8({src}), %rcx",
                        "movq 16({src}), %rdx",
                        "movq 24({src}), %r8",
                        "subq %rax, ({dst})",
                        "sbbq %rcx, 8({dst})",
                        "sbbq %rdx, 16({dst})",
                        "sbbq %r8, 24({dst})",
                        "adcq {borrow}, {borrow}",
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
