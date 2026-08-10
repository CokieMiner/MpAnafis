//! `PowerPC32` cross-limb shifted-high subtraction.
//!
//! Identical reasoning to the 64-bit kernel: the borrow lives in `XER[CA]`
//! while the loop guard writes a `CR` field and `bdnz` consumes `CTR`, so a
//! single `subfe` chain spans the whole subtraction. `srw`/`slw` and the plain
//! `or` leave both `CR` and `CA` alone, which is what lets the shift and merge
//! sit between two consecutive subtractions.
//!
//! Only the word forms and the four-byte strides differ from `powerpc64.rs`.

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
    reason = "Inlining is critical for peak performance, and Limb::BITS is 32 here while shift is strictly below it, so both shift counts fit a Limb"
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
            "lwz {prev}, 0({src_ptr})",
            "subfic {borrow_out}, {borrow}, 0",
            "addi {dst_ptr}, {dst_ptr}, -4",
            "cmplwi {paired}, 0",
            "beq 1f",
            "mtctr {paired}",
            ".p2align 4",
            "2:",
            "lwzu {next}, 4({src_ptr})",
            "srw {shifted}, {prev}, {right_shift}",
            "slw {high}, {next}, {left_shift}",
            "or {shifted}, {shifted}, {high}",
            "lwzu {minuend}, 4({dst_ptr})",
            "subfe {minuend}, {shifted}, {minuend}",
            "stw {minuend}, 0({dst_ptr})",
            "mr {prev}, {next}",
            "bdnz 2b",
            "1:",
            "srw {shifted}, {prev}, {right_shift}",
            "lwz {minuend}, 4({dst_ptr})",
            "subfe {minuend}, {shifted}, {minuend}",
            "stw {minuend}, 4({dst_ptr})",
            "li {borrow_out}, 0",
            "subfe {borrow_out}, {borrow_out}, {borrow_out}",
            "neg {borrow_out}, {borrow_out}",
            // `reg_nonzero` is required wherever a register is a base or an
            // `addi` source, because PowerPC reads r0 as the literal zero.
            src_ptr = inout(reg_nonzero) src_ptr => _,
            dst_ptr = inout(reg_nonzero) dst_ptr => _,
            paired = inout(reg) paired => _,
            borrow = in(reg) borrow,
            right_shift = in(reg) right_shift,
            left_shift = in(reg) left_shift,
            borrow_out = out(reg) borrow_out,
            prev = out(reg) _,
            next = out(reg) _,
            shifted = out(reg) _,
            high = out(reg) _,
            minuend = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    borrow_out
}
