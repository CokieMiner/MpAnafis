//! s390x in-place addition with two-limb carry chains.
//!
//! `alcgr` propagates the carry encoded by CC. `brctg` decrements block
//! and tail counts without modifying CC. An `algr` of zero initializes
//! CC; an `alcgr` of zero extracts the final carry.

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
#[inline(always)]
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
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
    // SAFETY: len > 4 proves chunks > 0; 2 * chunks + rem == len bounds
    // every aligned, initialized access. Each input pair is read before its
    // result is written, permitting exact aliasing.
    unsafe {
        asm!(
            // len > 4 guarantees a nonempty block loop.
            "lghi {carry}, 0",
            "algr {carry}, {carry}",        // CC = 0 (reset carry flag)
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "lg {src_val0}, 0({src})",      // load src[0]
            "lg {dst_val0}, 0({dst})",      // load dst[0]
            "alcgr {dst_val0}, {src_val0}", // dst[0] += src[0] + incoming carry
            "stg {dst_val0}, 0({dst})",     // store result
            "lg {src_val1}, 8({src})",      // load src[1]
            "lg {dst_val1}, 8({dst})",      // load dst[1]
            "alcgr {dst_val1}, {src_val1}", // dst[1] += src[1] + incoming carry
            "stg {dst_val1}, 8({dst})",     // store result
            "la {src}, 16({src})",          // advance src by 16 bytes
            "la {dst}, 16({dst})",          // advance dst by 16 bytes
            "brctg {chunks}, 2b",           // chunks--, loop if != 0 (preserves CC)
            // rem is 0 or 1; decrementing zero skips the single-limb tail.
            "brctg {rem}, 3f",              // if rem was 0 -> wrap to MAX, branch (skip tail); preserves CC
            "lg {src_val0}, 0({src})",      // load last limb
            "lg {dst_val0}, 0({dst})",      // load last dst
            "alcgr {dst_val0}, {src_val0}", // dst += src + incoming carry
            "stg {dst_val0}, 0({dst})",     // store result
            "3:",
            "lghi {carry}, 0",              // carry = 0
            "alcgr {carry}, {carry}",       // Extract the carry encoded by CC
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
            // SAFETY: len == 2 bounds both aligned, initialized spans.
            // All inputs are loaded before stores, permitting exact aliasing.
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
            // SAFETY: len == 3 bounds both aligned, initialized spans.
            // All inputs are loaded before stores, permitting exact aliasing.
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
            // SAFETY: len == 4 bounds both aligned, initialized spans.
            // All inputs are loaded before stores, permitting exact aliasing.
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
        // SAFETY: the caller establishes 2 <= len <= 4, all matched above.
        _ => unsafe { unreachable_unchecked() },
    }
}
