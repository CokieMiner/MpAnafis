//! `AArch64` cross-limb shifted-high subtraction.
//!
//! `lsr`/`lsl`/`orr`, `ldr`/`ldp`/`str`, the non-flag-setting `add`/`sub`
//! pointer and counter updates, and `cbz`/`cbnz` all leave NZCV untouched, so
//! one `sbcs` chain carries the borrow across the complete span. That is the
//! property the portable loop cannot express: `overflowing_sub` forces LLVM to
//! materialize each borrow into a general register and re-add it, because the
//! shift/`orr` sequence sits between consecutive subtractions.
//!
//! A64's `extr` funnel shift takes an *immediate* `#lsb`, but the shift here is
//! a runtime ring offset, so the register-form `lsr`+`lsl`+`orr` pair is used.
//!
//! The span is walked with the same three-part split as the x86-64 BMI2 kernel:
//! `(len - 1) / 2` paired iterations, one optional extra limb for even lengths,
//! and one final limb whose mathematical high neighbour is zero rather than an
//! addressable padding limb.

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
    reason = "Inlining is critical for peak assembly performance, and Limb::BITS is 64 here while shift is strictly below it, so both shift counts fit a Limb"
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
    // Leave one final limb outside the paired loop. An even length leaves one
    // additional ordinary limb before that final zero-extended limb.
    let pair_count = len.wrapping_sub(1).wrapping_shr(1);
    let even_extra = usize::from(len.is_multiple_of(2));
    let left_shift = shift as Limb;
    let right_shift = (Limb::BITS as Limb).wrapping_sub(left_shift);
    let borrow_out: Limb;

    // SAFETY: the caller proves both `len`-limb spans. Each paired iteration
    // reads three source limbs to produce two results; `pair_count =
    // (len - 1) / 2` keeps that third read inside `src[..len]`. The optional
    // even-length limb and the final zero-extended limb complete the span.
    // Every instruction between `sbcs` operations leaves NZCV unchanged.
    unsafe {
        asm!(
            "ldr {prev}, [{src_ptr}]",
            // C = 1 means "no borrow" on AArch64, so seed it from 0 - borrow.
            "cmp xzr, {borrow}",
            "cbz {pair_count}, 2f",
            ".p2align 4",
            "1:",
            "ldp {next0}, {next1}, [{src_ptr}, #8]",
            "lsr {shifted}, {prev}, {right_shift}",
            "lsl {high}, {next0}, {left_shift}",
            "orr {shifted}, {shifted}, {high}",
            "ldr {minuend}, [{dst_ptr}]",
            "sbcs {minuend}, {minuend}, {shifted}",
            "str {minuend}, [{dst_ptr}]",
            "lsr {shifted}, {next0}, {right_shift}",
            "lsl {high}, {next1}, {left_shift}",
            "orr {shifted}, {shifted}, {high}",
            "ldr {minuend}, [{dst_ptr}, #8]",
            "sbcs {minuend}, {minuend}, {shifted}",
            "str {minuend}, [{dst_ptr}, #8]",
            "mov {prev}, {next1}",
            "add {src_ptr}, {src_ptr}, #16",
            "add {dst_ptr}, {dst_ptr}, #16",
            "sub {pair_count}, {pair_count}, #1",
            "cbnz {pair_count}, 1b",
            "2:",
            "cbz {even_extra}, 3f",
            "ldr {next0}, [{src_ptr}, #8]",
            "lsr {shifted}, {prev}, {right_shift}",
            "lsl {high}, {next0}, {left_shift}",
            "orr {shifted}, {shifted}, {high}",
            "ldr {minuend}, [{dst_ptr}]",
            "sbcs {minuend}, {minuend}, {shifted}",
            "str {minuend}, [{dst_ptr}]",
            "mov {prev}, {next0}",
            "add {src_ptr}, {src_ptr}, #8",
            "add {dst_ptr}, {dst_ptr}, #8",
            // The limb above the last source limb is mathematically zero, so
            // only the right-shifted fragment survives.
            "3:",
            "lsr {shifted}, {prev}, {right_shift}",
            "ldr {minuend}, [{dst_ptr}]",
            "sbcs {minuend}, {minuend}, {shifted}",
            "str {minuend}, [{dst_ptr}]",
            "cset {borrow_out}, cc",
            src_ptr = inout(reg) src_ptr => _,
            dst_ptr = inout(reg) dst_ptr => _,
            pair_count = inout(reg) pair_count => _,
            even_extra = in(reg) even_extra,
            left_shift = in(reg) left_shift,
            right_shift = in(reg) right_shift,
            borrow = in(reg) borrow,
            borrow_out = lateout(reg) borrow_out,
            prev = out(reg) _,
            next0 = out(reg) _,
            next1 = out(reg) _,
            shifted = out(reg) _,
            high = out(reg) _,
            minuend = out(reg) _,
            options(nostack)
        );
    }
    borrow_out
}
