//! `AArch64` architecture-specific shift kernels.
//!
//! Merges bits from adjacent registers using `lsl`/`lsr`/`orr`. A64's `extr`
//! takes an *immediate* `#lsb`, but our shift amounts are runtime values
//! (divisor normalization shifts, user-supplied shift counts), so the
//! register-form `lsl`+`lsr`+`orr` sequence is used instead. This is still
//! tighter than the compiler's shift+and+or split (no explicit `and` mask).

use core::arch::asm;

use super::Limb;

/// Right-shift `len` limbs in-place by `shift` bits (0 < shift < 64).
/// Returns the bits shifted out of the bottom limb.
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
pub unsafe fn rshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    if len == 0 {
        return 0;
    }
    let mut carry_out: Limb;
    #[allow(
        clippy::as_conversions,
        reason = "Limb::BITS (≤64) fits in Limb on all targets; shift < Limb::BITS fits in Limb"
    )]
    let c_shift = (Limb::BITS as Limb).wrapping_sub(shift as Limb);
    let last = len.wrapping_sub(1);
    // SAFETY: Caller guarantees `limbs` has `len` elements, shift in 1..63
    unsafe {
        asm!(
            // carry_out = bottom_limb << (64 - shift)
            "ldr {bot}, [{limbs}]",
            "lsl {carry_out}, {bot}, {c_shift}",

            // Loop: from i = 0 up to len-2
            "mov {i}, #0",
            "cmp {i}, {last}",
            "b.ge 2f",                     // skip loop if len == 1

            "1:",
            "add {j}, {i}, #1",
            "ldr {next}, [{limbs}, {j}, lsl #3]",
            // limbs[i] = (next:bot >> shift) = (next << (64-shift)) | (bot >> shift)
            "lsl {tmp}, {next}, {c_shift}",
            "lsr {tmp2}, {bot}, {shift}",
            "orr {bot}, {tmp}, {tmp2}",
            "str {bot}, [{limbs}, {i}, lsl #3]",
            "mov {bot}, {next}",
            "mov {i}, {j}",
            "cmp {i}, {last}",
            "b.lt 1b",

            // Shift top limb in place. {bot} already holds limbs[last] from
            // the last loop iteration (or the original load when len == 1),
            // so no reload is needed.
            "2:",
            "lsr {bot}, {bot}, {shift}",
            "str {bot}, [{limbs}, {last}, lsl #3]",

            carry_out = out(reg) carry_out,
            limbs = in(reg) limbs,
            last = in(reg) last,
            shift = in(reg) u64::from(shift),
            c_shift = in(reg) c_shift,
            bot = out(reg) _,
            next = out(reg) _,
            i = out(reg) _,
            j = out(reg) _,
            tmp = out(reg) _,
            tmp2 = out(reg) _,
            options(nostack)
        );
    }
    carry_out
}
