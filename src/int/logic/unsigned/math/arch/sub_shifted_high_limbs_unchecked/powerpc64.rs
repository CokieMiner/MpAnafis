//! `PowerPC64` cross-limb shifted-high subtraction.
//!
//! Evaluates `dst -= ((src >> (64 - shift)) | (src[+1] << shift)) + borrow`
//! using `subfic`/`subfe` borrow chains and non-flag-setting 64-bit shifts (`srd`/`sld`).

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
/// # Microarchitectural Strategy
///
/// On PowerPC, the borrow lives in `XER[CA]` while comparisons write `CR` fields.
/// The loop counter, hardware CTR looping (`bdnz`), and non-recording `srd`/`sld`/`or`
/// leave `XER[CA]` untouched, allowing an uninterrupted single borrow chain.
///
/// # Safety
///
/// - `dst` and `src` must each point to valid memory for at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `0 < shift < 64`.
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
    let paired = len.wrapping_sub(1);
    let left_shift = shift as Limb;
    let right_shift = (Limb::BITS as Limb).wrapping_sub(left_shift);
    let borrow_out: Limb;

    // SAFETY:
    // 1. `dst` is valid for reads and writes of `len` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "ld {prev}, 0({src_ptr})",                   // Prime pipeline: load src[0]
            "subfic {borrow_out}, {borrow}, 0",          // CA = 1 iff borrow == 0 ("no borrow")
            "addi {dst_ptr}, {dst_ptr}, -8",             // Pre-bias dst_ptr for ldu pre-increment
            "cmpldi {paired}, 0",                        // Check if len == 1 (paired == 0)
            "beq 1f",                                    // If len == 1, skip main loop (1f)
            "mtctr {paired}",                            // Load paired count into hardware CTR register

            ".p2align 4",
            // Main shifted subtraction loop
            "2:",
            "ldu {next}, 8({src_ptr})",                  // Load next src limb and advance pointer (+8)
            "srd {shifted}, {prev}, {right_shift}",      // shifted = prev >> (64 - shift)
            "sld {high}, {next}, {left_shift}",          // high = next << shift
            "or {shifted}, {shifted}, {high}",           // Merge shifted bit fragments
            "ldu {minuend}, 8({dst_ptr})",               // Load dst[j] and advance pointer (+8)
            "subfe {minuend}, {shifted}, {minuend}",     // minuend = minuend - shifted + CA - 1
            "std {minuend}, 0({dst_ptr})",               // Store updated dst[j]
            "mr {prev}, {next}",                         // prev = next for next iteration
            "bdnz 2b",                                   // Decrement CTR and branch if != 0

            // Final zero-extended high limb (src[len] is mathematically zero)
            "1:",
            "srd {shifted}, {prev}, {right_shift}",      // Final high bits from previous limb
            "ld {minuend}, 8({dst_ptr})",                // Load last dst limb
            "subfe {minuend}, {shifted}, {minuend}",     // Final subtraction with borrow
            "std {minuend}, 8({dst_ptr})",               // Store last dst limb

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
            options(nostack)
        );
    }

    borrow_out
}
