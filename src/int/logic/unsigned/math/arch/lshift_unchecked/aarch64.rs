//! `AArch64` architecture-specific shift kernels.
//!
//! Merges bits from adjacent registers using `lsl`/`lsr`/`orr`. A64's `extr`
//! takes an *immediate* `#lsb`, but our shift amounts are runtime values
//! (divisor normalization shifts, user-supplied shift counts), so the
//! register-form `lsl`+`lsr`+`orr` sequence is used instead. This is still
//! tighter than the compiler's shift+and+or split (no explicit `and` mask).

use core::arch::asm;

use super::Limb;

/// Left-shift `len` limbs in-place by `shift` bits (0 < shift < 64).
/// Returns the bits shifted out of the top limb.
///
/// # Safety
///
/// - `limbs` must be valid for reads and writes of `len` elements.
/// - `shift` must satisfy `0 < shift < LIMB_BITS`: the kernel computes
///   `LIMB_BITS - shift` and applies both shift amounts to each element, so
///   an out-of-range amount is undefined behavior.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn lshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    if len == 0 {
        return 0;
    }
    let mut carry_out: Limb;
    #[allow(
        clippy::as_conversions,
        reason = "Limb::BITS (≤64) fits in Limb on all targets; shift < Limb::BITS fits in Limb"
    )]
    let c_shift = (Limb::BITS as Limb).wrapping_sub(shift as Limb);
    let idx = len.wrapping_sub(1);
    // SAFETY: Caller guarantees `limbs` has `len` elements, shift in 1..63
    unsafe {
        asm!(
            // carry_out = top_limb >> (64 - shift)
            "ldr {top}, [{limbs}, {idx}, lsl #3]",
            "lsr {carry_out}, {top}, {c_shift}",

            // Loop: from i = len-1 down to 1
            "mov {i}, {idx}",
            "cbz {i}, 2f",                 // skip loop if len == 1
            "1:",
            "sub {i}, {i}, #1",
            "ldr {prev}, [{limbs}, {i}, lsl #3]",
            // limbs[i+1] = (top << shift) | (prev >> (64-shift))
            "lsl {tmp}, {top}, {shift}",
            "lsr {tmp2}, {prev}, {c_shift}",
            "orr {top}, {tmp}, {tmp2}",
            "add {j}, {i}, #1",
            "str {top}, [{limbs}, {j}, lsl #3]",
            "mov {top}, {prev}",
            "cbnz {i}, 1b",

            // Shift bottom limb in place. {top} already holds limbs[0] from
            // the last loop iteration (or the original load when len == 1),
            // so no reload is needed.
            "2:",
            "lsl {top}, {top}, {shift}",
            "str {top}, [{limbs}]",

            carry_out = out(reg) carry_out,
            limbs = in(reg) limbs,
            idx = in(reg) idx,
            shift = in(reg) u64::from(shift),
            c_shift = in(reg) c_shift,
            top = out(reg) _,
            prev = out(reg) _,
            i = out(reg) _,
            j = out(reg) _,
            tmp = out(reg) _,
            tmp2 = out(reg) _,
            options(nostack)
        );
    }
    carry_out
}
