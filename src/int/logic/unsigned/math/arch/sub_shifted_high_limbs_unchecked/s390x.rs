//! `s390x` (IBM Z) cross-limb shifted-high subtraction.
//!
//! `slbgr` (subtract logical with borrow) reads the borrow out of the
//! condition code, so the whole span can run on one CC chain only if every
//! instruction between two `slbgr` operations leaves the CC alone. That rules
//! out `ogr` and `algr`, which both set the CC. The two shifted fragments
//! occupy disjoint bit ranges, so their OR equals their sum, and `la` computes
//! that sum as an address without touching the CC. `srlg`/`sllg`, `lg`/`stg`,
//! `lgr`, `lghi`, and `brctg` are likewise CC-neutral.
//!
//! `risbgn` would fuse the shift and insert without setting the CC, but its
//! bit range is an immediate and the shift here is a runtime ring offset.
//!
//! The CC seed is duplicated on the two entry paths because the guard that
//! skips an empty loop is itself a compare, and it therefore has to run before
//! the borrow reaches the condition code.

use core::arch::asm;

use super::Limb;

/// Subtract a cross-limb shifted source span from `dst`, including `borrow`.
///
/// For every `i < len`, the subtrahend limb is
/// `(src[i] >> (Limb::BITS - shift)) | (src[i + 1] << shift)`, with the
/// out-of-range `src[len]` term defined as zero.
///
/// # Safety
///
/// - `dst` and `src` must each cover `len` limbs.
/// - Their active spans must not overlap.
/// - `0 < shift < Limb::BITS`.
/// - `borrow <= 1`.
#[allow(
    clippy::inline_always,
    clippy::as_conversions,
    reason = "Inlining is critical for peak performance, and Limb::BITS is 64 here while shift is strictly below it, so both shift counts fit a Limb"
)]
#[inline(always)]
pub unsafe fn sub_shifted_high_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
    borrow: Limb,
) -> Limb {
    debug_assert!(
        shift > 0 && shift < Limb::BITS,
        "the cross-limb shift must be strictly inside one limb"
    );
    debug_assert!(borrow <= 1, "a subtraction borrow is one bit");
    if len == 0 {
        return borrow;
    }

    let dst_ptr = dst;
    let src_ptr = src;
    // The final limb is handled outside the loop: the mathematical limb above
    // it is zero, not an addressable padding limb.
    let paired = len.wrapping_sub(1);
    let left_shift = shift as Limb;
    let right_shift = (Limb::BITS as Limb).wrapping_sub(left_shift);
    let borrow_out: Limb;

    // SAFETY: the caller proves both `len`-limb spans. The loop runs
    // `len - 1` times and reads `src[i + 1]`, so the last read is `src[len-1]`;
    // the closing block consumes that limb with a zero high neighbour.
    unsafe {
        asm!(
            "lg {prev}, 0({src_ptr})",
            // `cgij` sets the CC, so the empty-loop guard has to precede the
            // borrow seed. Both entry paths seed it just before they need it.
            "cgij {paired}, 0, 8, 1f",
            "lghi {scratch}, 0",
            // 0 - borrow: CC 2 (carry, no borrow) when borrow is 0,
            // CC 1 (no carry, borrow) when borrow is 1. `slbgr` reads exactly
            // that carry.
            "slgr {scratch}, {borrow}",
            ".p2align 4",
            "2:",
            "lg {next}, 8({src_ptr})",
            "srlg {shifted}, {prev}, 0({right_shift})",
            "sllg {high}, {next}, 0({left_shift})",
            "la {shifted}, 0({shifted},{high})",
            "lg {minuend}, 0({dst_ptr})",
            "slbgr {minuend}, {shifted}",
            "stg {minuend}, 0({dst_ptr})",
            "lgr {prev}, {next}",
            "la {src_ptr}, 8({src_ptr})",
            "la {dst_ptr}, 8({dst_ptr})",
            "brctg {paired}, 2b",
            "j 4f",
            "1:",
            "lghi {scratch}, 0",
            "slgr {scratch}, {borrow}",
            "4:",
            "srlg {shifted}, {prev}, 0({right_shift})",
            "lg {minuend}, 0({dst_ptr})",
            "slbgr {minuend}, {shifted}",
            "stg {minuend}, 0({dst_ptr})",
            // 0 - 0 - borrow_in leaves 0 or -1; negate to recover the bit.
            "lghi {scratch}, 0",
            "slbgr {scratch}, {scratch}",
            "lcgr {scratch}, {scratch}",
            src_ptr = inout(reg_addr) src_ptr => _,
            dst_ptr = inout(reg_addr) dst_ptr => _,
            shifted = out(reg_addr) _,
            high = out(reg_addr) _,
            right_shift = in(reg_addr) right_shift,
            left_shift = in(reg_addr) left_shift,
            paired = inout(reg) paired => _,
            borrow = in(reg) borrow,
            scratch = out(reg) borrow_out,
            prev = out(reg) _,
            next = out(reg) _,
            minuend = out(reg) _,
            options(nostack)
        );
    }
    borrow_out
}
