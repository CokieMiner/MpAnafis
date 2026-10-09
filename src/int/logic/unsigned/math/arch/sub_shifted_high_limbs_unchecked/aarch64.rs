//! `AArch64` cross-limb shifted-high subtraction.
//!
//! Evaluates `dst -= ((src >> (64 - shift)) | (src[+1] << shift)) + borrow`
//! using flag-preserving register shifts and unified `sbcs` borrow chains.

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
    let even_extra = usize::from(len.is_multiple_of(2));
    let borrow_out: Limb;

    // SAFETY: len > 0 makes len - 1 exact. Each pair consumes two of those
    // lower limbs and reads their two successors; even_extra consumes the
    // remaining lower limb before the final zero-extended limb. Every access
    // stays in the aligned initialized disjoint spans. The shift contract
    // makes 64 - shift exact. Non-S shifts, merges, pointer updates and CB(N)Z
    // preserve NZCV between SBCS operations; early outputs preserve live inputs.
    unsafe {
        let pair_count = len.unchecked_sub(1) >> 1;
        let left_shift = shift as Limb;
        let right_shift = (Limb::BITS as Limb).unchecked_sub(left_shift);
        asm!(
            "ldr {prev}, [{src_ptr}]",                   // Prime pipeline: load src[0]
            "cmp xzr, {borrow}",                         // Seed carry flag (C = 1 means no borrow on ARM)
            "cbz {pair_count}, 2f",                      // If pair_count == 0, skip to single tail (2f)

            ".p2align 4",
            // Main 2-way unrolled paired loop
            "1:",
            "ldp {next0}, {next1}, [{src_ptr}, #8]",     // Paired load: next0 = src[j+1], next1 = src[j+2]

            // [Limb 0 Subtrahend Assembly & Subtraction]
            "lsr {shifted}, {prev}, {right_shift}",      // shifted = prev >> (64 - shift)
            "lsl {high}, {next0}, {left_shift}",         // high = next0 << shift
            "orr {shifted}, {shifted}, {high}",          // Merge shifted bits
            "ldr {minuend}, [{dst_ptr}]",                // Load dst[j]
            "sbcs {minuend}, {minuend}, {shifted}",      // dst[j] -= shifted + borrow, update C flag
            "str {minuend}, [{dst_ptr}]",                // Store updated dst[j]

            // [Limb 1 Subtrahend Assembly & Subtraction]
            "lsr {shifted}, {next0}, {right_shift}",     // shifted = next0 >> (64 - shift)
            "lsl {high}, {next1}, {left_shift}",         // high = next1 << shift
            "orr {shifted}, {shifted}, {high}",          // Merge shifted bits
            "ldr {minuend}, [{dst_ptr}, #8]",            // Load dst[j+1]
            "sbcs {minuend}, {minuend}, {shifted}",      // dst[j+1] -= shifted + borrow, update C flag
            "str {minuend}, [{dst_ptr}, #8]",            // Store updated dst[j+1]

            "mov {prev}, {next1}",                       // prev = next1 for next iteration
            "add {src_ptr}, {src_ptr}, #16",             // Advance src pointer by 16 bytes
            "add {dst_ptr}, {dst_ptr}, #16",             // Advance dst pointer by 16 bytes
            "sub {pair_count}, {pair_count}, #1",        // Decrement pair counter
            "cbnz {pair_count}, 1b",                     // Repeat while pair_count != 0

            // Even-length extra limb (when len is even)
            "2:",
            "cbz {even_extra}, 3f",                      // If len is odd, skip to final limb (3f)
            "ldr {next0}, [{src_ptr}, #8]",              // Load single next limb
            "lsr {shifted}, {prev}, {right_shift}",      // Shift low bits
            "lsl {high}, {next0}, {left_shift}",         // Shift high bits
            "orr {shifted}, {shifted}, {high}",          // Merge
            "ldr {minuend}, [{dst_ptr}]",                // Load dst
            "sbcs {minuend}, {minuend}, {shifted}",      // Subtract with borrow
            "str {minuend}, [{dst_ptr}]",                // Store dst
            "mov {prev}, {next0}",                       // prev = next0
            "add {src_ptr}, {src_ptr}, #8",              // Advance src
            "add {dst_ptr}, {dst_ptr}, #8",              // Advance dst

            // Final zero-extended high limb (src[len] is mathematically zero)
            "3:",
            "lsr {shifted}, {prev}, {right_shift}",      // Final high bits from previous limb
            "ldr {minuend}, [{dst_ptr}]",                // Load last dst limb
            "sbcs {minuend}, {minuend}, {shifted}",      // Final subtraction with borrow
            "str {minuend}, [{dst_ptr}]",                // Store last dst limb

            // C = 0 denotes borrow; CSET CC returns its binary value.
            "cset {borrow_out}, cc",                     // borrow_out = 1 if final borrow occurred

            dst_ptr = inout(reg) dst_ptr => _,
            src_ptr = inout(reg) src_ptr => _,
            pair_count = inout(reg) pair_count => _,
            even_extra = in(reg) even_extra,
            borrow = in(reg) borrow,
            left_shift = in(reg) left_shift,
            right_shift = in(reg) right_shift,
            borrow_out = out(reg) borrow_out,
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
