//! IBM z/Architecture (`s390x`) addition kernels (inline assembly).
//!
//! Uses the condition code (CC) via `alcgr` (add logical with carry).
//! `add_limbs_unchecked` is **2‑way unrolled** (`len >> 1`), processing
//! two limbs per iteration.  `add_limbs_3_unchecked` operates limb‑by‑limb.
//! Both use `brctg` (branch on count) for zero‑overhead looping that
//! preserves CC across iterations.
//!
//! ## Carry tracking
//!
//! `lghi {carry}, 0` / `algr {carry}, {carry}` clears CC = 0.
//! Each `alcgr` adds its two operands plus CC, writing the new CC (carry out).
//! The final carry is extracted with `lghi {carry}, 0` followed by
//! `alcgr {carry}, {carry}` which sets `carry = 0 + 0 + CC`.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Add `len` limbs from `src` into `dst` and return the final carry.
///
/// # Safety
///
/// `dst` and `src` must each be valid for `len` elements of type `Limb`.
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
        // SAFETY: The caller guarantees both pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*dst).overflowing_add(*src) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if len <= 4 {
        // SAFETY: Caller guarantees `dst` and `src` are valid for `len in 2..=4`.
        return unsafe { add_small_unchecked(dst, src, len) };
    }
    let mut carry: Limb;
    let chunks = len >> 1;
    let rem = len & 1;
    // SAFETY: Caller guarantees `dst`, `src` have at least `len` elements.
    //
    // CC-carrier pitfalls handled:
    //   - `cgij` clobbers CC, so we reset CC=0 *after* the chunks guard and
    //     before the main loop (the guard's CC result is not needed there).
    //   - The remainder skip uses `brctg` (which does NOT touch CC) instead of
    //     `cgij`: when `rem == 0`, `brctg` wraps to `u64::MAX` and branches
    //     (skipping the tail); when `rem == 1`, it decrements to 0 and falls
    //     through into the tail, preserving the main-loop carry in CC.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",      // skip main loop if chunks == 0

            // chunks > 0: reset CC=0 before entering main loop
            "lghi {carry}, 0",
            "algr {carry}, {carry}",        // CC = 0 (reset carry flag)
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "lg {src_val0}, 0({src})",      // load src[0]
            "lg {dst_val0}, 0({dst})",      // load dst[0]
            "alcgr {dst_val0}, {src_val0}", // dst[0] += src[0] + CC (carry-add)
            "stg {dst_val0}, 0({dst})",     // store result
            "lg {src_val1}, 8({src})",      // load src[1]
            "lg {dst_val1}, 8({dst})",      // load dst[1]
            "alcgr {dst_val1}, {src_val1}", // dst[1] += src[1] + CC
            "stg {dst_val1}, 8({dst})",     // store result
            "la {src}, 16({src})",          // advance src by 16 bytes
            "la {dst}, 16({dst})",          // advance dst by 16 bytes
            "brctg {chunks}, 2b",           // chunks--, loop if != 0 (preserves CC)
            "j 4f",                          // skip CC re-init (main loop done, CC already set)

            "1:",                             // chunks == 0: set CC=0 for tail
            "lghi {carry}, 0",
            "algr {carry}, {carry}",         // CC = 0 (reset carry flag)

            "4:",                             // common: CC is ready
            "brctg {rem}, 3f",              // if rem was 0 → wrap to MAX, branch (skip tail); preserves CC
            "lg {src_val0}, 0({src})",      // load last limb
            "lg {dst_val0}, 0({dst})",      // load last dst
            "alcgr {dst_val0}, {src_val0}", // dst += src + CC
            "stg {dst_val0}, 0({dst})",     // store result
            "3:",
            "lghi {carry}, 0",              // carry = 0
            "alcgr {carry}, {carry}",       // carry = 0 + 0 + CC (final carry bit)
            carry = out(reg) carry,
            dst = inout(reg_addr) dst => _,
            src = inout(reg_addr) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src_val0 = out(reg) _,
            src_val1 = out(reg) _,
            dst_val0 = out(reg) _,
            dst_val1 = out(reg) _,
            options(nostack)
        );
    }
    carry
}

/// Straight-line `dst[i] = dst[i] + src[i] + carry` chain for `len` in
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
    reason = "The fixed-size carry chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst` and `src` are valid for 2 limbs.
            unsafe {
                asm!(
                    "lg {s0}, 0({src})",
                    "lg {d0}, 0({dst})",
                    "lg {s1}, 8({src})",
                    "lg {d1}, 8({dst})",
                    "algr {d0}, {s0}",
                    "alcgr {d1}, {s1}",
                    "stg {d0}, 0({dst})",
                    "stg {d1}, 8({dst})",
                    "lghi {carry}, 0",
                    "alcgr {carry}, {carry}",
                    src = inout(reg_addr) src => _,
                    dst = inout(reg_addr) dst => _,
                    s0 = out(reg) _, s1 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        3 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst` and `src` are valid for 3 limbs.
            unsafe {
                asm!(
                    "lg {s0}, 0({src})",
                    "lg {d0}, 0({dst})",
                    "lg {s1}, 8({src})",
                    "lg {d1}, 8({dst})",
                    "lg {s2}, 16({src})",
                    "lg {d2}, 16({dst})",
                    "algr {d0}, {s0}",
                    "alcgr {d1}, {s1}",
                    "alcgr {d2}, {s2}",
                    "stg {d0}, 0({dst})",
                    "stg {d1}, 8({dst})",
                    "stg {d2}, 16({dst})",
                    "lghi {carry}, 0",
                    "alcgr {carry}, {carry}",
                    src = inout(reg_addr) src => _,
                    dst = inout(reg_addr) dst => _,
                    s0 = out(reg) _, s1 = out(reg) _, s2 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _, d2 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        4 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst` and `src` are valid for 4 limbs.
            unsafe {
                asm!(
                    "lg {s0}, 0({src})",
                    "lg {d0}, 0({dst})",
                    "lg {s1}, 8({src})",
                    "lg {d1}, 8({dst})",
                    "lg {s2}, 16({src})",
                    "lg {d2}, 16({dst})",
                    "lg {s3}, 24({src})",
                    "lg {d3}, 24({dst})",
                    "algr {d0}, {s0}",
                    "alcgr {d1}, {s1}",
                    "alcgr {d2}, {s2}",
                    "alcgr {d3}, {s3}",
                    "stg {d0}, 0({dst})",
                    "stg {d1}, 8({dst})",
                    "stg {d2}, 16({dst})",
                    "stg {d3}, 24({dst})",
                    "lghi {carry}, 0",
                    "alcgr {carry}, {carry}",
                    src = inout(reg_addr) src => _,
                    dst = inout(reg_addr) dst => _,
                    s0 = out(reg) _, s1 = out(reg) _, s2 = out(reg) _, s3 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _, d2 = out(reg) _, d3 = out(reg) _,
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
