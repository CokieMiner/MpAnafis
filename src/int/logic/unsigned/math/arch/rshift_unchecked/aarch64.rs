//! `AArch64` multi-precision right shift kernel.
//!
//! Shifts an array of limbs right by `shift` bits in place, extracting the low overflow bits
//! and merging adjacent limbs using 64-bit `lsl`/`lsr`/`orr` sequences.

use core::arch::asm;

use super::Limb;

/// Right-shift `len` limbs in-place by `shift` bits (`0 < shift < 64`).
/// Returns the bits shifted out of the bottom limb.
///
/// Computes:
///
/// ```text
///   (limbs[0..len], carry_out) = limbs[0..len] >> shift
/// ```
///
/// Operates bottom-up (from 0 up to `len - 1`) to allow strictly in-place modifications without
/// overwriting unshifted bits. Adjacent limbs are combined via register-form `lsl`+`lsr`+`orr`.
///
/// # Safety
///
/// - For nonzero `len`, `limbs` covers `len` aligned, initialized, writable
///   limbs within `isize::MAX` bytes.
/// - `0 < shift < 64`.
#[expect(
    clippy::inline_always,
    reason = "Inlining keeps the normalization leaf at its arithmetic caller"
)]
#[inline(always)]
pub unsafe fn rshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    if len == 0 {
        return 0;
    }
    let mut carry_out: Limb;
    #[expect(
        clippy::as_conversions,
        reason = "Limb::BITS (<=64) fits in Limb on all targets; shift < Limb::BITS fits in Limb"
    )]
    // SAFETY: len > 0 and 0 < shift < 64 make both differences exact. Both
    // counts fit Limb on this 64-bit backend.
    let (c_shift, count) = unsafe {
        ((Limb::BITS as Limb).unchecked_sub(shift as Limb), len.unchecked_sub(1))
    };
    let ptr = limbs;

    // SAFETY: the aligned initialized writable span contains len > 0 limbs.
    // The first load is at zero; each of len-1 iterations loads the next limb
    // before storing the lower result. The final store reaches len-1. Shift
    // counts are in 1..64, and every modified register is an output.
    unsafe {
        asm!(
            // Extract bottom carry: carry_out = bottom_limb << (64 - shift)
            "ldr {bot}, [{ptr}]",                        // Load lowest limb limbs[0]
            "lsl {carry_out}, {bot}, {c_shift}",         // Extract low overflow bits

            // Loop: bottom-up traversal from limbs[0] up to limbs[len-2]
            "cbz {count}, 2f",                           // If len == 1 (count == 0), skip to top limb (2f)

            "1:",
            "ldr {next}, [{ptr}, #8]!",                  // Load next higher limb and advance ptr
            "lsl {tmp}, {next}, {c_shift}",              // tmp = next << (64 - shift)
            "lsr {tmp2}, {bot}, {shift}",                // tmp2 = bot >> shift
            "orr {bot}, {tmp}, {tmp2}",                  // Merge bits into new limb value
            "str {bot}, [{ptr}, #-8]",                   // Store merged limb at previous position
            "mov {bot}, {next}",                         // bot = next for next iteration
            "subs {count}, {count}, #1",                 // Decrement loop counter
            "b.ne 1b",                                   // Repeat while count != 0

            // Shift top limb in place (limbs[len-1] has no higher neighbour)
            "2:",
            "lsr {bot}, {bot}, {shift}",                 // limbs[len-1] = limbs[len-1] >> shift
            "str {bot}, [{ptr}]",                        // Store finalized top limb

            carry_out = out(reg) carry_out,
            ptr = inout(reg) ptr => _,
            count = inout(reg) count => _,
            shift = in(reg) u64::from(shift),
            c_shift = in(reg) c_shift,
            bot = out(reg) _,
            next = out(reg) _,
            tmp = out(reg) _,
            tmp2 = out(reg) _,
            options(nostack),
        );
    }
    carry_out
}
