//! `x86_64` carry propagation kernel (inline assembly).
//!
//! Resolves the overwhelmingly common first-limb stop before entering assembly,
//! then uses CF via `adcq` in a 4-way unrolled loop for a genuine carry chain.

use core::arch::asm;

use super::Limb;

/// Propagate a carry through a raw limb pointer slice.
///
/// # Safety
///
/// `dst` must be valid for reading and writing `len` elements of type `Limb`.
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
    // A carry reaches the second limb iff dst[0] was Limb::MAX. Handling this
    // scalar first step here avoids entering the unrolled assembly for the
    // common probability 1 - 1/B, where B = 2^Limb::BITS.
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
    let idx = 0_usize;
    // SAFETY: Caller guarantees `tail_dst` is valid for `tail_len` elements.
    unsafe {
        asm!(
            "btq $0, {carry}",
            "decq {chunks}",
            "js 1f",
            ".p2align 4",
            "2:",
            "adcq $0, ({dst}, {idx}, 8)",
            "adcq $0, 8({dst}, {idx}, 8)",
            "adcq $0, 16({dst}, {idx}, 8)",
            "adcq $0, 24({dst}, {idx}, 8)",
            "jnc 5f",
            "leaq 4({idx}), {idx}",
            "decq {chunks}",
            "jns 2b",
            "1:",
            "decq {rem}",
            "js 5f",
            ".p2align 4",
            "3:",
            "adcq $0, ({dst}, {idx}, 8)",
            "jnc 5f",
            "leaq 1({idx}), {idx}",
            "decq {rem}",
            "jns 3b",
            "5:",
            "movl $0, {carry:e}",
            "adcq {carry}, {carry}",
            carry = inout(reg) carry,
            dst = in(reg) tail_dst,
            idx = inout(reg) idx => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            options(nostack, att_syntax)
        );
    }
    carry
}
