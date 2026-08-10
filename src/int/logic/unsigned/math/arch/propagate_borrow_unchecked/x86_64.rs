//! `x86_64` borrow propagation kernel (inline assembly).
//!
//! Resolves the overwhelmingly common first-limb stop before entering assembly,
//! then uses CF via `sbbq` in a 4-way unrolled loop for a genuine borrow chain.

use core::arch::asm;

use super::Limb;

/// Propagate a borrow through a raw limb pointer slice.
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
pub unsafe fn propagate_borrow_unchecked(dst: *mut Limb, len: usize, mut borrow: Limb) -> Limb {
    debug_assert!(borrow <= 1, "borrow propagation accepts a binary borrow");
    if len == 0 {
        return borrow;
    }
    // A borrow reaches the second limb iff dst[0] was zero. Handling this
    // scalar first step avoids entering assembly with probability 1 - 1/B,
    // where B = 2^Limb::BITS.
    // SAFETY: the caller guarantees dst is readable and writable for len > 0.
    let (first_difference, first_underflow) = unsafe { (*dst).overflowing_sub(borrow) };
    // SAFETY: the caller guarantees the first destination limb is writable.
    unsafe {
        *dst = first_difference;
    }
    if !first_underflow {
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
            "btq $0, {borrow}",
            "decq {chunks}",
            "js 1f",
            ".p2align 4",
            "2:",
            "sbbq $0, ({dst}, {idx}, 8)",
            "sbbq $0, 8({dst}, {idx}, 8)",
            "sbbq $0, 16({dst}, {idx}, 8)",
            "sbbq $0, 24({dst}, {idx}, 8)",
            "jnc 5f",
            "leaq 4({idx}), {idx}",
            "decq {chunks}",
            "jns 2b",
            "1:",
            "decq {rem}",
            "js 5f",
            ".p2align 4",
            "3:",
            "sbbq $0, ({dst}, {idx}, 8)",
            "jnc 5f",
            "leaq 1({idx}), {idx}",
            "decq {rem}",
            "jns 3b",
            "5:",
            "movl $0, {borrow:e}",
            "adcq {borrow}, {borrow}",
            borrow = inout(reg) borrow,
            dst = in(reg) tail_dst,
            idx = inout(reg) idx => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            options(nostack, att_syntax)
        );
    }
    borrow
}
