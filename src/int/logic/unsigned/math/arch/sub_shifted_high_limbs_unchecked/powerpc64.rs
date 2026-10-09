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

    // SAFETY: len > 0 makes len - 1 exact. CTR processes those lower limbs
    // and loads one successor each time; the final limb uses zero extension.
    // Each access stays in the aligned initialized disjoint spans. The biased
    // destination is incremented before access. Shifts, merges, comparisons
    // and CTR control preserve XER.CA. Both shift counts are in 1..64; early
    // outputs preserve inputs and CTR, XER and CR0 clobbers are declared.
    unsafe {
        let paired = len.unchecked_sub(1);
        let left_shift = shift as Limb;
        let right_shift = (Limb::BITS as Limb).unchecked_sub(left_shift);
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
            out("cr0") _,
            options(nostack)
        );
    }

    borrow_out
}
