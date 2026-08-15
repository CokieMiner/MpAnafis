//! `PowerPC32` cross-limb shifted-high subtraction.
//!
//! Evaluates `dst -= ((src >> (32 - shift)) | (src[+1] << shift)) + borrow`
//! using `subfic`/`subfe` borrow chains and non-flag-setting 32-bit shifts (`srw`/`slw`).

use core::arch::asm;

use super::Limb;

/// Subtract a cross-limb shifted source span from `dst`, including `borrow`.
///
/// For every `i < len`, the subtrahend limb is:
///
/// ```text
///   (src[i] >> (32 - shift)) | (src[i + 1] << shift)
/// ```
///
/// with the out-of-range `src[len]` term defined as zero.
///
/// # Microarchitectural Strategy
///
/// On PowerPC, the borrow lives in `XER[CA]` while comparisons write `CR` fields.
/// The loop counter, hardware CTR looping (`bdnz`), and non-recording `srw`/`slw`/`or`
/// leave `XER[CA]` untouched, allowing an uninterrupted single borrow chain.
///
/// # Safety
///
/// - `dst` and `src` must each point to valid memory for at least `len` initialized 32-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `0 < shift < 32`.
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
    let paired = len.wrapping_sub(1);
    let left_shift = shift as Limb;
    let right_shift = (Limb::BITS as Limb).wrapping_sub(left_shift);
    let borrow_out: Limb;

    // SAFETY:
    // 1. `dst` is valid for reads and writes of `len` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Pointer offsets remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "lwz {prev}, 0({src_ptr})",                   // Prime pipeline: load src[0]
            "subfic {borrow_out}, {borrow}, 0",          // CA = 1 iff borrow == 0 ("no borrow")
            "addi {dst_ptr}, {dst_ptr}, -4",             // Pre-bias dst_ptr for lwzu pre-increment
            "cmplwi {paired}, 0",                        // Check if len == 1 (paired == 0)
            "beq 1f",                                    // If len == 1, skip main loop (1f)
            "mtctr {paired}",                            // Load paired count into hardware CTR register

            ".p2align 4",
            // Main shifted subtraction loop
            "2:",
            "lwzu {next}, 4({src_ptr})",                 // Load next src limb and advance pointer (+4)
            "srw {shifted}, {prev}, {right_shift}",      // shifted = prev >> (32 - shift)
            "slw {high}, {next}, {left_shift}",          // high = next << shift
            "or {shifted}, {shifted}, {high}",           // Merge shifted bit fragments
            "lwzu {minuend}, 4({dst_ptr})",              // Load dst[j] and advance pointer (+4)
            "subfe {minuend}, {shifted}, {minuend}",     // minuend = minuend - shifted + CA - 1
            "stw {minuend}, 0({dst_ptr})",               // Store updated dst[j]
            "mr {prev}, {next}",                         // prev = next for next iteration
            "bdnz 2b",                                   // Decrement CTR and branch if != 0

            // Final zero-extended high limb (src[len] is mathematically zero)
            "1:",
            "srw {shifted}, {prev}, {right_shift}",      // Final high bits from previous limb
            "lwz {minuend}, 4({dst_ptr})",               // Load last dst limb
            "subfe {minuend}, {shifted}, {minuend}",     // Final subtraction with borrow
            "stw {minuend}, 4({dst_ptr})",               // Store last dst limb

            // Extract borrow: subfe on zero gives (CA - 1), negating gives 0 or 1
            "li {borrow_out}, 0",                        // borrow_out = 0
            "subfe {borrow_out}, {borrow_out}, {borrow_out}", // borrow_out = CA - 1 (0 or -1)
            "neg {borrow_out}, {borrow_out}",            // borrow_out = -(CA - 1) (0 or 1)

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
