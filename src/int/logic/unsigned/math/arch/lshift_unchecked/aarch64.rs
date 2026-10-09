//! `AArch64` multi-precision left shift kernel.
//!
//! Shifts an array of limbs left by `shift` bits in place, extracting the high overflow bits
//! and merging adjacent limbs using 64-bit `lsl`/`lsr`/`orr` sequences.

use core::arch::asm;

use super::Limb;

/// Left-shift `len` limbs in-place by `shift` bits (`0 < shift < 64`).
/// Returns the bits shifted out of the top limb.
///
/// Computes:
///
/// ```text
///   (carry_out, limbs[0..len]) = limbs[0..len] << shift
/// ```
///
/// Operates top-down (from `len - 1` down to 0) to allow strictly in-place modifications without
/// overwriting unshifted bits. Adjacent limbs are combined via register-form `lsl`+`lsr`+`orr`.
///
/// # Safety
///
/// - For nonzero `len`, `limbs` must cover `len` aligned, initialized, writable
///   limbs within `isize::MAX` bytes.
/// - `0 < shift < 64`.
#[expect(
    clippy::inline_always,
    reason = "Inlining keeps the normalization leaf at its arithmetic caller"
)]
#[inline(always)]
pub unsafe fn lshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    if len == 0 {
        return 0;
    }
    let mut carry_out: Limb;
    #[expect(
        clippy::as_conversions,
        reason = "Limb::BITS (<=64) fits in Limb on all targets; shift < Limb::BITS fits in Limb"
    )]
    // SAFETY: len > 0 and 0 < shift < 64 make both differences exact. Each
    // count fits Limb on this 64-bit backend.
    let (c_shift, count) = unsafe {
        ((Limb::BITS as Limb).unchecked_sub(shift as Limb), len.unchecked_sub(1))
    };
    // SAFETY: Caller guarantees `limbs` has `len` elements, so `limbs.add(len - 1)` is within bounds.
    let ptr = unsafe { limbs.add(count) };

    // SAFETY: the caller provides an aligned, initialized writable span and
    // 0 < shift < 64. The top load is at len-1; each of count=len-1 iterations
    // loads one lower limb before storing the higher output. The final store
    // reaches limb zero. Every modified register is declared as an output.
    unsafe {
        asm!(
            // Extract top carry: carry_out = top_limb >> (64 - shift)
            "ldr {top}, [{ptr}]",                        // Load highest limb limbs[len-1]
            "lsr {carry_out}, {top}, {c_shift}",         // Extract high overflow bits

            // Loop: top-down traversal from limbs[len-1] down to limbs[1]
            "cbz {count}, 2f",                           // If len == 1 (count == 0), skip to bottom limb (2f)

            "1:",
            "ldr {prev}, [{ptr}, #-8]!",                 // Load next lower limb and pre-decrement ptr
            "lsl {tmp}, {top}, {shift}",                 // tmp = top << shift
            "lsr {tmp2}, {prev}, {c_shift}",             // tmp2 = prev >> (64 - shift)
            "orr {top}, {tmp}, {tmp2}",                  // Merge bits into new limb value
            "str {top}, [{ptr}, #8]",                    // Store merged limb at original position
            "mov {top}, {prev}",                         // top = prev for next iteration
            "subs {count}, {count}, #1",                 // Decrement loop counter
            "b.ne 1b",                                   // Repeat while count != 0

            // Shift bottom limb in place (limbs[0] has no lower neighbour)
            "2:",
            "lsl {top}, {top}, {shift}",                 // limbs[0] = limbs[0] << shift
            "str {top}, [{ptr}]",                        // Store finalized bottom limb

            carry_out = out(reg) carry_out,
            ptr = inout(reg) ptr => _,
            count = inout(reg) count => _,
            shift = in(reg) u64::from(shift),
            c_shift = in(reg) c_shift,
            top = out(reg) _,
            prev = out(reg) _,
            tmp = out(reg) _,
            tmp2 = out(reg) _,
            options(nostack),
        );
    }
    carry_out
}
