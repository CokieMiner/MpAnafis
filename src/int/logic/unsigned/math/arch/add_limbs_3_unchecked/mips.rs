//! MIPS 32-bit addition kernels (inline assembly).
//!
//! MIPS has no hardware carry flag, so carry is tracked manually with
//! `sltu` (set-less-than unsigned) and `or`.
//!
//! The loop is **4-way unrolled** (`len >> 2`) for maximum throughput.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Compute `dst[i] = src1[i] + src2[i] + carry` for `len` limbs, returning
/// the final carry.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span: the kernel writes `dst`
///   while it reads `src1` and `src2`.
/// - `src1` and `src2` are read-only and may alias each other.
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
    // SAFETY: The caller guarantees both pointers cover `len` elements.
    if len == 0 {
        return 0;
    }
    if len == 1 {
        // SAFETY: The caller guarantees all pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if len <= 4 {
        // SAFETY: Caller guarantees `dst`, `src1`, `src2` valid for `len in 2..=4`.
        return unsafe { add_small_3_unchecked(dst, src1, src2, len) };
    }
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: Caller guarantees pointers are valid for `len` elements.
    unsafe {
        asm!(
            ".set noat",
            "beqz {chunks}, 2f",           // skip main loop if chunks == 0

            // ── 4‑way unrolled loop ────────────────────────────────────
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            "lw {t0}, 0({src1})",
            "lw {t1}, 0({src2})",
            "addu {t2}, {t1}, {t0}",
            "sltu {c0}, {t2}, {t0}",
            "addu {t2}, {t2}, {carry}",
            "sltu {c1}, {t2}, {carry}",
            "or {carry}, {c0}, {c1}",
            "sw {t2}, 0({dst})",

            "lw {t0}, 4({src1})",
            "lw {t1}, 4({src2})",
            "addu {t2}, {t1}, {t0}",
            "sltu {c0}, {t2}, {t0}",
            "addu {t2}, {t2}, {carry}",
            "sltu {c1}, {t2}, {carry}",
            "or {carry}, {c0}, {c1}",
            "sw {t2}, 4({dst})",

            "lw {t0}, 8({src1})",
            "lw {t1}, 8({src2})",
            "addu {t2}, {t1}, {t0}",
            "sltu {c0}, {t2}, {t0}",
            "addu {t2}, {t2}, {carry}",
            "sltu {c1}, {t2}, {carry}",
            "or {carry}, {c0}, {c1}",
            "sw {t2}, 8({dst})",

            "lw {t0}, 12({src1})",
            "lw {t1}, 12({src2})",
            "addu {t2}, {t1}, {t0}",
            "sltu {c0}, {t2}, {t0}",
            "addu {t2}, {t2}, {carry}",
            "sltu {c1}, {t2}, {carry}",
            "or {carry}, {c0}, {c1}",
            "sw {t2}, 12({dst})",

            "addiu {src1}, {src1}, 16",
            "addiu {src2}, {src2}, 16",
            "addiu {dst}, {dst}, 16",
            "addiu {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            // ── Tail: single‑limb remainder loop ───────────────────────
            "2:",
            "beqz {rem}, 4f",              // skip tail if rem == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "lw {t0}, 0({src1})",
            "lw {t1}, 0({src2})",
            "addu {t2}, {t1}, {t0}",
            "sltu {c0}, {t2}, {t0}",
            "addu {t2}, {t2}, {carry}",
            "sltu {c1}, {t2}, {carry}",
            "or {carry}, {c0}, {c1}",
            "sw {t2}, 0({dst})",
            "addiu {src1}, {src1}, 4",
            "addiu {src2}, {src2}, 4",
            "addiu {dst}, {dst}, 4",
            "addiu {rem}, {rem}, -1",
            "bnez {rem}, 3b",
            "4:",

            carry = inout(reg) carry,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            dst = inout(reg) dst => _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            c0 = out(reg) _,
            c1 = out(reg) _,
            options(nostack)
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
    clippy::too_many_lines,
    reason = "The fixed-size carry chains must remain visibly unrolled and inline into the public hot kernel"
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
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 2 limbs.
            unsafe {
                asm!(
                    ".set noat",
                    // Limb 0 (carry-in = 0)
                    "lw {t0}, 0({src1})",
                    "lw {t1}, 0({src2})",
                    "addu {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "sw {t1}, 0({dst})",
                    // Limb 1
                    "lw {t0}, 4({src1})",
                    "lw {t1}, 4({src2})",
                    "addu {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "addu {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 4({dst})",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    t0 = out(reg) _, t1 = out(reg) _,
                    c0 = out(reg) _, c1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        3 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 3 limbs.
            unsafe {
                asm!(
                    ".set noat",
                    // Limb 0 (carry-in = 0)
                    "lw {t0}, 0({src1})",
                    "lw {t1}, 0({src2})",
                    "addu {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "sw {t1}, 0({dst})",
                    // Limb 1
                    "lw {t0}, 4({src1})",
                    "lw {t1}, 4({src2})",
                    "addu {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "addu {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 4({dst})",
                    // Limb 2
                    "lw {t0}, 8({src1})",
                    "lw {t1}, 8({src2})",
                    "addu {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "addu {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 8({dst})",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    t0 = out(reg) _, t1 = out(reg) _,
                    c0 = out(reg) _, c1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        4 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 4 limbs.
            unsafe {
                asm!(
                    ".set noat",
                    // Limb 0 (carry-in = 0)
                    "lw {t0}, 0({src1})",
                    "lw {t1}, 0({src2})",
                    "addu {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "sw {t1}, 0({dst})",
                    // Limb 1
                    "lw {t0}, 4({src1})",
                    "lw {t1}, 4({src2})",
                    "addu {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "addu {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 4({dst})",
                    // Limb 2
                    "lw {t0}, 8({src1})",
                    "lw {t1}, 8({src2})",
                    "addu {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "addu {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 8({dst})",
                    // Limb 3
                    "lw {t0}, 12({src1})",
                    "lw {t1}, 12({src2})",
                    "addu {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "addu {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 12({dst})",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    t0 = out(reg) _, t1 = out(reg) _,
                    c0 = out(reg) _, c1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        // SAFETY: Caller guarantees `len in 2..=4`.
        _ => unsafe { unreachable_unchecked() },
    }
}
