//! `s390x` (IBM Z) cross-limb shifted-high subtraction.
//!
//! Evaluates `dst -= ((src >> (64 - shift)) | (src[+1] << shift)) + borrow`
//! using `slbgr` borrow chains and CC-neutral address arithmetic (`la`).

use core::arch::asm;

use super::Limb;

/// Subtract a cross-limb shifted source span from `dst`, including `borrow`.
///
/// For every `i < len`, the subtrahend limb is:
///
/// ```text
///   (src[i] >> (64 - shift)) | (src[i + 1] << shift)
/// ```
///
/// with the out-of-range `src[len]` term defined as zero.
///
/// # Safety
///
/// - Both pointers must cover `len` aligned, initialized limbs in disjoint spans.
/// - `dst` requires exclusive access; each span's byte length must fit in `isize`.
/// - Zero length permits null pointers.
/// - `0 < shift < 64`.
/// - `borrow <= 1`.
#[expect(
    clippy::inline_always,
    clippy::as_conversions,
    reason = "inlining exposes the limb recurrence to its caller; the 1..64 shift counts widen exactly to 64-bit registers"
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
    let borrow_out: Limb;

    // SAFETY: len > 0 makes len - 1 exact. Each lower limb reads its successor
    // within the aligned initialized disjoint spans; the final limb uses zero
    // extension. The shift contract bounds both counts to 1..64. Logical
    // shifts, LA merges, CGIJ and BRCTG preserve CC; the fragments have
    // disjoint bit ranges, so their merge cannot overflow. Early outputs
    // preserve inputs, and default asm options declare CC modification.
    unsafe {
        let paired = len.unchecked_sub(1);
        let left_shift = shift as Limb;
        let right_shift = (Limb::BITS as Limb).unchecked_sub(left_shift);
        asm!(
            "lg {prev}, 0({src_ptr})",                   // Prime pipeline: load src[0]
            "lghi {scratch}, 0",                         // scratch = 0
            "slgr {scratch}, {borrow}",                  // Seed CC from (0 - borrow)
            "cgij {paired}, 0, 8, 1f",                   // len == 1 goes directly to the final limb without changing CC.

            ".p2align 4",
            // Main shifted subtraction loop
            "2:",
            "lg {next}, 8({src_ptr})",                   // Load next src limb
            "srlg {shifted}, {prev}, 0({right_shift})",  // shifted = prev >> (64 - shift)
            "sllg {high}, {next}, 0({left_shift})",      // high = next << shift
            "la {shifted}, 0({shifted},{high})",         // CC-neutral sum: shifted = shifted + high
            "lg {minuend}, 0({dst_ptr})",                // Load dst[j]
            "slbgr {minuend}, {shifted}",                // minuend = minuend - shifted - borrow (updates CC)
            "stg {minuend}, 0({dst_ptr})",               // Store updated dst[j]
            "lgr {prev}, {next}",                        // prev = next for next iteration
            "la {src_ptr}, 8({src_ptr})",                // Advance src pointer (+8)
            "la {dst_ptr}, 8({dst_ptr})",                // Advance dst pointer (+8)
            "brctg {paired}, 2b",                        // Decrement paired and branch if nonzero, preserving CC.

            // Final zero-extended high limb (src[len] is mathematically zero)
            "1:",
            "srlg {shifted}, {prev}, 0({right_shift})",  // Final high bits from previous limb
            "lg {minuend}, 0({dst_ptr})",                // Load last dst limb
            "slbgr {minuend}, {shifted}",                // Final subtraction with borrow
            "stg {minuend}, 0({dst_ptr})",               // Store last dst limb

            // Extract borrow out of CC: (0 - 0 - borrow) -> 0 or -1, then negate to get 0 or 1
            "lghi {scratch}, 0",                         // scratch = 0
            "slbgr {scratch}, {scratch}",                // scratch = 0 - 0 - borrow (0 or -1)
            "lcgr {scratch}, {scratch}",                 // scratch = -scratch (0 or 1)

            src_ptr = inout(reg_addr) src_ptr => _,
            dst_ptr = inout(reg_addr) dst_ptr => _,
            shifted = out(reg_addr) _,
            high = out(reg_addr) _,
            right_shift = in(reg_addr) right_shift,
            left_shift = in(reg_addr) left_shift,
            paired = inout(reg) paired => _,
            borrow = in(reg) borrow,
            prev = out(reg) _,
            next = out(reg) _,
            minuend = out(reg) _,
            scratch = out(reg) borrow_out,
            options(nostack)
        );
    }

    borrow_out
}
