//! x86‑64 (AMD64) 3‑operand addition kernel.
//!
//! Uses `adcq` for all limbs. Lengths above four use an optional 4-limb
//! prefix followed by an **8-way unrolled** loop.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Compute `dst[i] = src1[i] + src2[i] + carry` for `len` limbs,
/// returning the final carry.
///
/// # Safety
///
/// `dst`, `src1`, and `src2` must each be valid for `len` elements.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 1 {
        // SAFETY: the caller guarantees both source pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: the caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if (2..=4).contains(&len) {
        // SAFETY: the caller guarantees all pointers cover `len` limbs, and
        // this branch proves the fixed kernel's `2..=4` length precondition.
        return unsafe { add_small_3_unchecked(dst, src1, src2, len) };
    }
    let mut carry: Limb;
    let prefix = (len >> 2) & 1;
    let chunks = len >> 3;
    let rem = len & 3;
    let idx = 0_usize;
    // Binary decomposition gives `len = 8*chunks + 4*prefix + rem` with
    // `prefix <= 1` and `rem < 4`. The blocks run in increasing index order;
    // every intervening `decq`, `js`/`jns`, and `leaq` preserves CF, so the
    // carry established by limb i is consumed unchanged by limb i+1.
    // SAFETY: Assembly block accesses `len` elements from `dst`, `src1`, and `src2`, which caller guarantees are valid.
    unsafe {
        asm!(
            "xorl {carry:e}, {carry:e}",        // carry = 0, clears CF
            // ── Optional 4-limb prefix, then 8-way loop ──────────────────
            "decq {prefix}",
            "js 1f",
            "movq ({src1}, {idx}, 8), %rax",
            "movq 8({src1}, {idx}, 8), %rcx",
            "movq 16({src1}, {idx}, 8), %rdx",
            "movq 24({src1}, {idx}, 8), %r8",
            "adcq ({src2}, {idx}, 8), %rax",
            "adcq 8({src2}, {idx}, 8), %rcx",
            "adcq 16({src2}, {idx}, 8), %rdx",
            "adcq 24({src2}, {idx}, 8), %r8",
            "movq %rax, ({dst}, {idx}, 8)",
            "movq %rcx, 8({dst}, {idx}, 8)",
            "movq %rdx, 16({dst}, {idx}, 8)",
            "movq %r8, 24({dst}, {idx}, 8)",
            "leaq 4({idx}), {idx}",
            "1:",
            "decq {chunks}",                      // jump if -1 (chunks == 0)
            "js 3f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "movq ({src1}, {idx}, 8), %rax",     // load first 4 limbs of src1
            "movq 8({src1}, {idx}, 8), %rcx",
            "movq 16({src1}, {idx}, 8), %rdx",
            "movq 24({src1}, {idx}, 8), %r8",
            "adcq ({src2}, {idx}, 8), %rax",     // add first 4 limbs of src2 + CF
            "adcq 8({src2}, {idx}, 8), %rcx",
            "adcq 16({src2}, {idx}, 8), %rdx",
            "adcq 24({src2}, {idx}, 8), %r8",
            "movq %rax, ({dst}, {idx}, 8)",      // store first 4 limbs to dst
            "movq %rcx, 8({dst}, {idx}, 8)",
            "movq %rdx, 16({dst}, {idx}, 8)",
            "movq %r8, 24({dst}, {idx}, 8)",
            "movq 32({src1}, {idx}, 8), %rax",   // load next 4 limbs of src1
            "movq 40({src1}, {idx}, 8), %rcx",
            "movq 48({src1}, {idx}, 8), %rdx",
            "movq 56({src1}, {idx}, 8), %r8",
            "adcq 32({src2}, {idx}, 8), %rax",   // add next 4 limbs of src2 + CF
            "adcq 40({src2}, {idx}, 8), %rcx",
            "adcq 48({src2}, {idx}, 8), %rdx",
            "adcq 56({src2}, {idx}, 8), %r8",
            "movq %rax, 32({dst}, {idx}, 8)",    // store next 4 limbs to dst
            "movq %rcx, 40({dst}, {idx}, 8)",
            "movq %rdx, 48({dst}, {idx}, 8)",
            "movq %r8, 56({dst}, {idx}, 8)",
            "leaq 8({idx}), {idx}",              // advance idx by 8
            "decq {chunks}",                      // decrement chunks
            "jns 2b",                             // loop back if chunks >= 0
            // ── Tail: single‑limb remainder ────────────────────────────
            "3:",
            "decq {rem}",                         // jump if -1 (rem == 0)
            "js 5f",
            ".p2align 4",                          // align tail loop header
            "4:",
            "movq ({src1}, {idx}, 8), %rax",     // load src1[idx]
            "adcq ({src2}, {idx}, 8), %rax",     // rax += src2[idx] + CF
            "movq %rax, ({dst}, {idx}, 8)",      // store dst[idx]
            "leaq 1({idx}), {idx}",              // advance idx by 1
            "decq {rem}",                         // decrement remainder
            "jns 4b",                             // loop back if rem >= 0
            "5:",
            "adcq {carry}, {carry}",             // carry = 0 + 0 + CF
            carry = out(reg) carry,
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
    carry
}

/// Straight-line `dst[i] = src1[i] + src2[i] + carry` chain for `len` in
/// `2..=4`.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span: it is written while `src1`
///   and `src2` are read.
/// - `src1` and `src2` are read-only and may alias each other.
#[allow(
    clippy::inline_always,
    reason = "The fixed-size carry chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn add_small_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: The caller guarantees all pointers are valid for 2 limbs.
            unsafe {
                asm!(
                    "xorl {carry:e}, {carry:e}",
                    "movq ({src1}), %rax",
                    "movq 8({src1}), %rcx",
                    "addq ({src2}), %rax",
                    "adcq 8({src2}), %rcx",
                    "movq %rax, ({dst})",
                    "movq %rcx, 8({dst})",
                    "adcq {carry}, {carry}",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
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
            // SAFETY: The caller guarantees all pointers are valid for 3 limbs.
            unsafe {
                asm!(
                    "xorl {carry:e}, {carry:e}",
                    "movq ({src1}), %rax",
                    "movq 8({src1}), %rcx",
                    "movq 16({src1}), %rdx",
                    "addq ({src2}), %rax",
                    "adcq 8({src2}), %rcx",
                    "adcq 16({src2}), %rdx",
                    "movq %rax, ({dst})",
                    "movq %rcx, 8({dst})",
                    "movq %rdx, 16({dst})",
                    "adcq {carry}, {carry}",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
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
            // SAFETY: The caller guarantees all pointers are valid for 4 limbs.
            unsafe {
                asm!(
                    "xorl {carry:e}, {carry:e}",
                    "movq ({src1}), %rax",
                    "movq 8({src1}), %rcx",
                    "movq 16({src1}), %rdx",
                    "movq 24({src1}), %r8",
                    "addq ({src2}), %rax",
                    "adcq 8({src2}), %rcx",
                    "adcq 16({src2}), %rdx",
                    "adcq 24({src2}), %r8",
                    "movq %rax, ({dst})",
                    "movq %rcx, 8({dst})",
                    "movq %rdx, 16({dst})",
                    "movq %r8, 24({dst})",
                    "adcq {carry}, {carry}",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
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
