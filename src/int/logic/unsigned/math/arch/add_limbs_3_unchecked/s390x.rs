//! s390x addition into a disjoint destination.
//!
//! `alcgr` propagates the carry encoded by CC through two-limb blocks.
//! `brctg` decrements block and tail counts without modifying CC.
//! An `algr` of zero initializes CC; an `alcgr` of zero extracts the carry.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Writes the low `len` limbs of `src1 + src2` and returns the binary carry.
///
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// For nonempty spans, all pointers must be aligned and cover `len` limbs.
/// Each span must lie within one live allocation and have at most `isize::MAX` bytes.
/// The sources must be initialized and readable; the destination must be
/// writable and disjoint from both sources. Its previous contents are unused.
/// The two source spans may overlap each other.
#[expect(
    clippy::inline_always,
    reason = "Keep the hardware carry chain in the selected arithmetic caller"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 0 {
        return 0;
    }
    if len == 1 {
        // SAFETY: len == 1 bounds both aligned, initialized source spans.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: len == 1 bounds the aligned, writable destination span.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if len <= 4 {
        // SAFETY: prior returns and len <= 4 prove 2 <= len <= 4. The caller
        // supplies aligned sources and a disjoint writable destination.
        return unsafe { add_small_3_unchecked(dst, src1, src2, len) };
    }
    let mut carry: Limb;
    let chunks = len >> 1;
    let rem = len & 1;
    // SAFETY: len > 4 gives chunks > 0; 2 * chunks + rem == len bounds every
    // aligned read and write. Both sources are initialized and may overlap;
    // dst is disjoint and only written. All modified GPRs are declared.
    unsafe {
        asm!(
            // len > 4 guarantees a nonempty block loop.
            "lghi {carry}, 0",
            "algr {carry}, {carry}",           // CC = 0 (reset carry flag)
            ".p2align 4",                          // align the loop to 16 bytes
            "2:",
            "lg {src1_val0}, 0({src1})",    // load src1[0]
            "lg {src2_val0}, 0({src2})",    // load src2[0]
            "alcgr {src1_val0}, {src2_val0}", // dst[0] = src1[0] + src2[0] + incoming carry
            "stg {src1_val0}, 0({dst})",    // store result
            "lg {src1_val1}, 8({src1})",    // load src1[1]
            "lg {src2_val1}, 8({src2})",    // load src2[1]
            "alcgr {src1_val1}, {src2_val1}", // dst[1] = src1[1] + src2[1] + incoming carry
            "stg {src1_val1}, 8({dst})",    // store result
            "la {src1}, 16({src1})",        // advance src1 by 16 bytes
            "la {src2}, 16({src2})",        // advance src2 by 16 bytes
            "la {dst}, 16({dst})",          // advance dst by 16 bytes
            "brctg {chunks}, 2b",           // chunks--, loop if != 0 (preserves CC)
            // rem is 0 or 1; decrementing zero skips the single-limb tail.
            "brctg {rem}, 3f",              // if rem was 0 -> wrap to MAX, branch (skip tail); preserves CC
            "lg {src1_val0}, 0({src1})",    // load last limb
            "lg {src2_val0}, 0({src2})",    // load last src2
            "alcgr {src1_val0}, {src2_val0}", // final source sum plus incoming carry
            "stg {src1_val0}, 0({dst})",    // store result
            "3:",
            "lghi {carry}, 0",              // carry = 0
            "alcgr {carry}, {carry}",       // Extract the carry encoded by CC
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

/// Writes a sum with a straight carry chain for `2 <= len <= 4`.
///
/// # Safety
///
/// `len` must be in `2..=4`. Both aligned sources must contain `len`
/// initialized limbs. The aligned destination must cover `len` writable limbs
/// and be disjoint from both sources; the sources may overlap each other.
#[expect(
    clippy::inline_always,
    reason = "Keep fixed-size carry chains inside the selected kernel"
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
            // SAFETY: len == 2 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
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
            // SAFETY: len == 3 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
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
            // SAFETY: len == 4 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
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
        // SAFETY: the caller establishes 2 <= len <= 4, all matched above.
        _ => unsafe { unreachable_unchecked() },
    }
}
