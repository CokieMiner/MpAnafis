//! IBM z/Architecture (`s390x`) addition kernels (inline assembly).
//!
//! Uses the condition code (CC) via `alcgr` (add logical with carry).
//! Both `add_limbs_unchecked` and `add_limbs_3_unchecked` are **2‑way unrolled**
//! (`len >> 1`), processing two limbs per iteration with a single‑limb tail.
//!
//! ## Unrolling Rationale (Why 2-way instead of 4-way)
//!
//! Unlike `x86_64` or `aarch64` which use 4-way unrolling, `s390x` kernels use 2-way unrolling.
//! This is because remainder handling (`len & 1`) with 2-way unrolling can be executed
//! using a single `brctg` (branch on count) instruction, which branches without clobbering
//! CC. Unrolling 4-way would leave up to 3 remainder limbs (`len & 3`), requiring additional
//! compare and jump instructions (`cgij`/`clgij`) or loops that would clobber CC or require
//! CC-saving overhead, negating any benefits of higher unrolling factors.
//!
//! ## Carry tracking
//!
//! `lghi {carry}, 0` / `algr {carry}, {carry}` clears CC = 0.
//! Each `alcgr` adds its two operands plus CC, writing the new CC (carry out).
//! The final carry is extracted with `lghi {carry}, 0` followed by
//! `alcgr {carry}, {carry}` which sets `carry = 0 + 0 + CC`.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// 3-operand add: `dst[i] = src1[i] + src2[i] + carry` for `len` limbs.
///
/// Iterates two limbs per loop using `alcgr` with `brctg` for counting and CC preservation.
///
/// # Safety
///
/// `dst`, `src1`, and `src2` must each be valid for `len` elements of type `Limb`.
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
    let mut carry: Limb;
    let chunks = len >> 1;
    let rem = len & 1;
    // SAFETY: Caller guarantees `dst`, `src1`, `src2` have at least `len` elements.
    // `cgij` clobbers CC; we reset CC=0 after the guard before entering the loop.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",         // skip main loop if chunks == 0

            // chunks > 0: reset CC=0 before entering main loop
            "lghi {carry}, 0",
            "algr {carry}, {carry}",           // CC = 0 (reset carry flag)
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "lg {src1_val0}, 0({src1})",    // load src1[0]
            "lg {src2_val0}, 0({src2})",    // load src2[0]
            "alcgr {src1_val0}, {src2_val0}", // dst[0] = src1[0] + src2[0] + CC
            "stg {src1_val0}, 0({dst})",    // store result
            "lg {src1_val1}, 8({src1})",    // load src1[1]
            "lg {src2_val1}, 8({src2})",    // load src2[1]
            "alcgr {src1_val1}, {src2_val1}", // dst[1] = src1[1] + src2[1] + CC
            "stg {src1_val1}, 8({dst})",    // store result
            "la {src1}, 16({src1})",        // advance src1 by 16 bytes
            "la {src2}, 16({src2})",        // advance src2 by 16 bytes
            "la {dst}, 16({dst})",          // advance dst by 16 bytes
            "brctg {chunks}, 2b",           // chunks--, loop if != 0 (preserves CC)
            "j 4f",                          // skip CC re-init (main loop done, CC already set)

            "1:",                             // chunks == 0: set CC=0 for tail
            "lghi {carry}, 0",
            "algr {carry}, {carry}",         // CC = 0 (reset carry flag)

            "4:",                             // common: CC is ready
            "brctg {rem}, 3f",              // if rem was 0 → wrap to MAX, branch (skip tail); preserves CC
            "lg {src1_val0}, 0({src1})",    // load last limb
            "lg {src2_val0}, 0({src2})",    // load last src2
            "alcgr {src1_val0}, {src2_val0}", // dst += src1 + src2 + CC
            "stg {src1_val0}, 0({dst})",    // store result
            "3:",
            "lghi {carry}, 0",              // carry = 0
            "alcgr {carry}, {carry}",       // carry = 0 + 0 + CC (final carry bit)
            carry = out(reg) carry,
            dst = inout(reg_addr) dst => _,
            src1 = inout(reg_addr) src1 => _,
            src2 = inout(reg_addr) src2 => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1_val0 = out(reg) _, src2_val0 = out(reg) _,
            src1_val1 = out(reg) _, src2_val1 = out(reg) _,
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
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 2 limbs.
            unsafe {
                asm!(
                    "lg {a0}, 0({src1})",
                    "lg {b0}, 0({src2})",
                    "lg {a1}, 8({src1})",
                    "lg {b1}, 8({src2})",
                    "algr {a0}, {b0}",
                    "alcgr {a1}, {b1}",
                    "stg {a0}, 0({dst})",
                    "stg {a1}, 8({dst})",
                    "lghi {carry}, 0",
                    "alcgr {carry}, {carry}",
                    src1 = inout(reg_addr) src1 => _,
                    src2 = inout(reg_addr) src2 => _,
                    dst = inout(reg_addr) dst => _,
                    a0 = out(reg) _, a1 = out(reg) _,
                    b0 = out(reg) _, b1 = out(reg) _,
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
                    "lg {a0}, 0({src1})",
                    "lg {b0}, 0({src2})",
                    "lg {a1}, 8({src1})",
                    "lg {b1}, 8({src2})",
                    "lg {a2}, 16({src1})",
                    "lg {b2}, 16({src2})",
                    "algr {a0}, {b0}",
                    "alcgr {a1}, {b1}",
                    "alcgr {a2}, {b2}",
                    "stg {a0}, 0({dst})",
                    "stg {a1}, 8({dst})",
                    "stg {a2}, 16({dst})",
                    "lghi {carry}, 0",
                    "alcgr {carry}, {carry}",
                    src1 = inout(reg_addr) src1 => _,
                    src2 = inout(reg_addr) src2 => _,
                    dst = inout(reg_addr) dst => _,
                    a0 = out(reg) _, a1 = out(reg) _, a2 = out(reg) _,
                    b0 = out(reg) _, b1 = out(reg) _, b2 = out(reg) _,
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
                    "lg {a0}, 0({src1})",
                    "lg {b0}, 0({src2})",
                    "lg {a1}, 8({src1})",
                    "lg {b1}, 8({src2})",
                    "lg {a2}, 16({src1})",
                    "lg {b2}, 16({src2})",
                    "lg {a3}, 24({src1})",
                    "lg {b3}, 24({src2})",
                    "algr {a0}, {b0}",
                    "alcgr {a1}, {b1}",
                    "alcgr {a2}, {b2}",
                    "alcgr {a3}, {b3}",
                    "stg {a0}, 0({dst})",
                    "stg {a1}, 8({dst})",
                    "stg {a2}, 16({dst})",
                    "stg {a3}, 24({dst})",
                    "lghi {carry}, 0",
                    "alcgr {carry}, {carry}",
                    src1 = inout(reg_addr) src1 => _,
                    src2 = inout(reg_addr) src2 => _,
                    dst = inout(reg_addr) dst => _,
                    a0 = out(reg) _, a1 = out(reg) _, a2 = out(reg) _, a3 = out(reg) _,
                    b0 = out(reg) _, b1 = out(reg) _, b2 = out(reg) _, b3 = out(reg) _,
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
