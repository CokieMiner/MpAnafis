//! `AArch64` carry propagation kernel (inline assembly).
//!
//! Resolves the common first-limb stop before entering assembly, then uses CF
//! via `adcs` in a 4-way unrolled loop for a genuine carry chain.

use core::arch::asm;

use super::Limb;

/// Propagate carry through `dst` slice.
///
/// # Safety
/// Caller guarantees `dst` points to at least `len` valid limbs.
#[allow(
    unsafe_code,
    reason = "Hardware inline assembly natively requires unsafe code"
)]
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
#[inline(always)]
pub unsafe fn propagate_carry_unchecked(dst: *mut Limb, len: usize, mut carry: Limb) -> Limb {
    debug_assert!(carry <= 1, "carry propagation accepts a binary carry");
    if len == 0 {
        return carry;
    }
    // SAFETY: the caller guarantees dst is readable and writable for len > 0.
    let (first_sum, first_overflow) = unsafe { (*dst).overflowing_add(carry) };
    // SAFETY: the caller guarantees the first destination limb is writable.
    unsafe {
        *dst = first_sum;
    }
    if !first_overflow {
        return 0;
    }
    if len == 1 {
        return 1;
    }

    // SAFETY: len > 1 proves the one-limb offset remains within the allocation.
    let tail_dst = unsafe { dst.add(1) };
    let tail_len = len.wrapping_sub(1);
    let chunks = tail_len >> 2;
    let rem = tail_len & 3;
    // SAFETY: Caller guarantees `tail_dst` is valid for `tail_len` elements.
    unsafe {
        asm!(
            "cmp {carry}, #1",                      // set C flag if carry >= 1
            "cbz {chunks}, 1f",                      // skip main loop if chunks == 0
            ".p2align 4",
            "2:",
            "ldp {val0}, {val1}, [{dst}]",
            "ldp {val2}, {val3}, [{dst}, #16]",
            "adcs {val0}, {val0}, xzr",
            "adcs {val1}, {val1}, xzr",
            "adcs {val2}, {val2}, xzr",
            "adcs {val3}, {val3}, xzr",
            "stp {val0}, {val1}, [{dst}], #16",
            "stp {val2}, {val3}, [{dst}], #16",
            "b.cc 5f",                              // early exit if carry cleared
            "sub {chunks}, {chunks}, #1",
            "cbnz {chunks}, 2b",
            "1:",
            "cbz {rem}, 5f",
            ".p2align 4",
            "3:",
            "ldr {val0}, [{dst}]",
            "adcs {val0}, {val0}, xzr",
            "str {val0}, [{dst}], #8",
            "b.cc 5f",                              // early exit if carry cleared
            "sub {rem}, {rem}, #1",
            "cbnz {rem}, 3b",
            "5:",
            "cset {carry}, cs",                     // carry = 1 if C set, 0 otherwise
            carry = inout(reg) carry,
            dst = inout(reg) tail_dst => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            val0 = out(reg) _, val1 = out(reg) _, val2 = out(reg) _, val3 = out(reg) _,
            options(nostack)
        );
    }
    carry
}
