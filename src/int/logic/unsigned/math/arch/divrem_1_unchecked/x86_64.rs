//! `x86_64` division kernel using the `divq` instruction.
//!
//! Evaluates `(q, r) = (rem_hi:limb) / divisor` in a single hardware `divq` instruction.

use core::arch::asm;

use super::Limb;

/// Divide `(rem_hi << LIMB_BITS) | limb` by `divisor`.
///
/// Computes:
///
/// ```text
///   (quotient, remainder) = ((rem_hi << 64) | limb) / divisor
/// ```
///
/// `divq {d}` on `x86_64` divides the 128-bit unsigned value in `%rdx:%rax` by the 64-bit divisor `{d}`.
/// Returns quotient in `%rax` and remainder in `%rdx`.
///
/// # Safety
///
/// `divisor` must be non-zero and `rem_hi < divisor` (otherwise
/// the quotient would overflow 64 bits, causing a hardware `#DE` exception).
#[expect(
    clippy::inline_always,
    reason = "Inlining places the hardware divide at its arithmetic caller without a leaf-call boundary"
)]
#[inline(always)]
pub unsafe fn divrem_1_unchecked(limb: Limb, rem_hi: Limb, divisor: Limb) -> (Limb, Limb) {
    let mut q_val: Limb = limb;   // Placed in rax (low 64 bits of numerator)
    let mut r_val: Limb = rem_hi; // Placed in rdx (high 64 bits of numerator)

    // SAFETY: divisor > 0 and rem_hi < divisor make DIVQ's quotient fit in
    // RAX without a divide exception. RAX and RDX are read/write operands;
    // the divisor cannot share either early-clobbered register. No memory or
    // stack is accessed, and identical inputs produce identical outputs.
    unsafe {
        asm!(
            "divq {d}",                                  // %rax = (%rdx:%rax) / d, %rdx = (%rdx:%rax) % d
            d = in(reg) divisor,
            inout("rax") q_val,
            inout("rdx") r_val,
            options(pure, nomem, nostack, att_syntax)
        );
    }
    (q_val, r_val)
}
