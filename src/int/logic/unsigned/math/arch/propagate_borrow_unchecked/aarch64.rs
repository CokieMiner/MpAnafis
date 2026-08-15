//! `AArch64` borrow propagation kernel (inline assembly).
//!
//! Propagates a borrow through an array of limbs in place, utilizing 4-way unrolled
//! paired loads/stores (`ldp`/`stp`) and `sbcs` with early-exit on cleared borrow (`b.cs`).

use core::arch::asm;

use super::Limb;

/// Propagate borrow through `dst` slice in-place.
///
/// Returns the final borrow-out (0 or 1).
///
/// # Microarchitectural Strategy
///
/// Handles the common first-limb early termination in fast Rust code before entering assembly.
/// The assembly kernel executes 4-way unrolled paired loads and stores, checking for early exit
/// with `b.cs` (C=1 means no borrow on ARM) if the borrow chain clears.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `borrow <= 1`.
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

    // SAFETY:
    // 1. `tail_dst` is valid for reads and writes of `tail_len` 64-bit `Limb` elements.
    // 2. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "cmp xzr, {borrow}",                         // Seed carry flag (C = 0 if borrow == 1, C = 1 if borrow == 0)
            "cbz {chunks}, 1f",                          // If chunks == 0, skip unrolled loop (1f)

            ".p2align 4",                                // Align loop header for optimal branch target decoding
            // 4-way unrolled paired loop
            "2:",                                        // Loop head label
            "ldp {val0}, {val1}, [{dst}]",               // Load limbs [j, j+1]
            "ldp {val2}, {val3}, [{dst}, #16]",          // Load limbs [j+2, j+3]
            "sbcs {val0}, {val0}, xzr",                  // val0 -= 0 + borrow (updates C flag)
            "sbcs {val1}, {val1}, xzr",                  // val1 -= 0 + borrow
            "sbcs {val2}, {val2}, xzr",                  // val2 -= 0 + borrow
            "sbcs {val3}, {val3}, xzr",                  // val3 -= 0 + borrow
            "stp {val0}, {val1}, [{dst}], #16",          // Store updated limbs [j, j+1]
            "stp {val2}, {val3}, [{dst}], #16",          // Store updated limbs [j+2, j+3]
            "b.cs 5f",                                   // Early exit if no borrow (C == 1)
            "sub {chunks}, {chunks}, #1",                // Decrement chunk counter
            "cbnz {chunks}, 2b",                         // Repeat while chunks != 0

            // Remainder 1-limb loop (0 to 3 limbs)
            "1:",                                        // Remainder entry label
            "cbz {rem}, 5f",                             // If rem == 0, skip to exit (5f)

            ".p2align 4",                                // Align remainder loop header
            "3:",                                        // Tail loop label
            "ldr {val0}, [{dst}]",                       // Load single limb
            "sbcs {val0}, {val0}, xzr",                  // Subtract borrow
            "str {val0}, [{dst}], #8",                   // Store limb and advance pointer (+8)
            "b.cs 5f",                                   // Early exit if no borrow (C == 1)
            "sub {rem}, {rem}, #1",                      // Decrement remainder
            "cbnz {rem}, 3b",                            // Repeat while rem != 0

            // Exit point: capture final borrow condition
            "5:",                                        // Exit label
            "cset {borrow}, cc",                         // borrow = 1 if C == 0 (borrow occurred), 0 otherwise

            borrow = inout(reg) borrow,
            dst = inout(reg) tail_dst => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            val0 = out(reg) _,
            val1 = out(reg) _,
            val2 = out(reg) _,
            val3 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
