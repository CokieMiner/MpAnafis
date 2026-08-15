//! `s390x` (IBM Z) borrow propagation kernel (inline assembly).
//!
//! Propagates a borrow through an array of limbs in place, utilizing 2-way unrolled
//! `slbgr` (subtract logical with borrow) and CC-neutral `brctg` branching.

use core::arch::asm;

use super::Limb;

/// Propagate borrow through `dst` slice in-place.
///
/// Returns the final borrow-out (0 or 1).
///
/// # Microarchitectural Strategy
///
/// Uses `slbgr` to propagate borrow across destination limbs, seeding the Condition Code (CC)
/// via `slgr` with `0 - borrow`. `brctg` decrements the loop counter without modifying CC.
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
    if borrow == 0 || len == 0 {
        return borrow;
    }
    let chunks = len >> 1;
    let rem = len & 1;
    let zero_const: Limb = 0;

    // SAFETY:
    // 1. `dst` is valid for reads and writes of `len` 64-bit `Limb` elements.
    // 2. Pointer offsets remain within allocated bounds.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",                   // If chunks == 0, skip main loop (1f)

            // Seed CC from borrow: (0 - borrow) produces CC borrow iff borrow == 1
            "lghi {cc_seed}, 0",                         // cc_seed = 0
            "slgr {cc_seed}, {borrow}",                  // Set Condition Code (CC)

            ".p2align 4",                                // Align loop header for branch prediction
            // 2-way unrolled main loop
            "2:",                                        // Loop head label
            "lg {val0}, 0({dst})",                       // Load dst[j]
            "lg {val1}, 8({dst})",                       // Load dst[j+1]
            "slbgr {val0}, {zero}",                      // val0 -= 0 + borrow_from_CC
            "slbgr {val1}, {zero}",                      // val1 -= 0 + borrow_from_CC
            "stg {val0}, 0({dst})",                      // Store updated dst[j]
            "stg {val1}, 8({dst})",                      // Store updated dst[j+1]
            "la {dst}, 16({dst})",                       // Advance dst pointer (+16)
            "brctg {chunks}, 2b",                        // Decrement chunks and branch if > 0 (CC-neutral)
            "j 4f",                                      // Jump to tail/exit

            // Remainder entry seed path
            "1:",                                        // Remainder entry label
            "lghi {cc_seed}, 0",                         // cc_seed = 0
            "slgr {cc_seed}, {borrow}",                  // Set CC

            // 1-limb tail
            "4:",                                        // Tail loop label
            "brctg {rem}, 3f",                           // If rem == 0, skip tail (3f)
            "lg {val0}, 0({dst})",                       // Load single limb
            "slbgr {val0}, {zero}",                      // Subtract borrow
            "stg {val0}, 0({dst})",                      // Store limb
            "3:",                                        // Remainder exit label

            // Capture final borrow out of CC: (0 - 0 - borrow) -> 0 or -1, then negate to get 0 or 1
            "5:",                                        // Exit label
            "lghi {borrow}, 0",                          // borrow = 0
            "slbgr {borrow}, {borrow}",                  // borrow = 0 - 0 - borrow (0 or -1)
            "lcgr {borrow}, {borrow}",                   // borrow = -borrow (0 or 1)

            borrow = inout(reg) borrow,
            dst = inout(reg_addr) dst => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            zero = inout(reg) zero_const => _,
            cc_seed = out(reg) _,
            val0 = out(reg) _,
            val1 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
