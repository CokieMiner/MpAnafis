//! `AArch64` carry propagation kernel (inline assembly).
//!
//! Propagates a carry through an array of limbs in place, utilizing 4-way unrolled
//! paired loads/stores (`ldp`/`stp`) and `adcs` with early-exit on cleared carry (`b.cc`).

#![expect(unsafe_code, reason = "The kernel accesses caller-proven raw spans and AArch64 assembly")]

use core::arch::asm;

use super::Limb;

/// Propagate carry through `dst` slice in-place.
///
/// Returns the final carry-out (0 or 1).
///
/// Handles first-limb termination before entering assembly.
/// The assembly kernel executes 4-way unrolled paired loads and stores, checking for early exit
/// with `b.cc` if the carry chain clears before the end of the slice.
///
/// # Safety
///
/// - For nonzero `len`, `dst` covers `len` aligned, initialized, writable limbs
///   within `isize::MAX` bytes.
/// - `carry <= 1`.
#[expect(clippy::inline_always, reason = "Inlining retains the first-limb stopping path at its arithmetic caller")]
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
    // SAFETY: len > 1 makes the one-limb length reduction exact.
    let tail_len = unsafe { len.unchecked_sub(1) };
    let chunks = tail_len >> 2;
    let rem = tail_len & 3;

    // SAFETY: the initialized writable tail contains tail_len limbs. Blocks
    // cover four limbs and the tail covers tail_len % 4, so all accesses stay
    // in the span. The first addition established carry=1; ADCS propagates it
    // through C, and flag-neutral counter updates preserve C. Stopping after
    // C=0 leaves following limbs unchanged. All modified registers are outputs.
    unsafe {
        asm!(
            "cmp {carry}, #1",                           // Seed carry flag (C = 1 if carry >= 1)
            "cbz {chunks}, 1f",                          // If chunks == 0, skip unrolled loop (1f)

            ".p2align 4",                                // Align the loop header
            // 4-way unrolled paired loop
            "2:",                                        // Loop head label
            "ldp {val0}, {val1}, [{dst}]",               // Load limbs [j, j+1]
            "ldp {val2}, {val3}, [{dst}, #16]",          // Load limbs [j+2, j+3]
            "adcs {val0}, {val0}, xzr",                  // val0 += C flag + 0
            "adcs {val1}, {val1}, xzr",                  // val1 += C flag + 0
            "adcs {val2}, {val2}, xzr",                  // val2 += C flag + 0
            "adcs {val3}, {val3}, xzr",                  // val3 += C flag + 0
            "stp {val0}, {val1}, [{dst}], #16",          // Store updated limbs [j, j+1]
            "stp {val2}, {val3}, [{dst}], #16",          // Store updated limbs [j+2, j+3]
            "b.cc 5f",                                   // Early exit if carry cleared (C = 0)
            "sub {chunks}, {chunks}, #1",                // Decrement chunk counter
            "cbnz {chunks}, 2b",                         // Repeat while chunks != 0

            // Remainder 1-limb loop (0 to 3 limbs)
            "1:",                                        // Remainder entry label
            "cbz {rem}, 5f",                             // If rem == 0, skip to exit (5f)

            ".p2align 4",                                // Align remainder loop header
            "3:",                                        // Tail loop label
            "ldr {val0}, [{dst}]",                       // Load single limb
            "adcs {val0}, {val0}, xzr",                  // Add carry
            "str {val0}, [{dst}], #8",                   // Store limb and advance pointer (+8)
            "b.cc 5f",                                   // Early exit if carry cleared
            "sub {rem}, {rem}, #1",                      // Decrement remainder
            "cbnz {rem}, 3b",                            // Repeat while rem != 0

            // Exit point: capture final carry condition
            "5:",                                        // Exit label
            "cset {carry}, cs",                          // carry = 1 if C flag set (carry out), 0 otherwise

            carry = inout(reg) carry,
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
    carry
}
