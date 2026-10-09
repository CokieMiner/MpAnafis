//! `s390x` (IBM Z) borrow propagation kernel (inline assembly).
//!
//! Propagates a borrow through an array of limbs in place, utilizing 2-way unrolled
//! `slbgr` (subtract logical with borrow) and CC-neutral `brctg` branching.

#![expect(unsafe_code, reason = "The kernel accesses caller-proven raw spans and s390x assembly")]

use core::arch::asm;

use super::Limb;

/// Propagate borrow through `dst` slice in-place.
///
/// Returns the final borrow-out (0 or 1).
///
/// Uses `slbgr` to propagate borrow across destination limbs, seeding the Condition Code (CC)
/// via `slgr` with `0 - borrow`. `brctg` decrements the loop counter without modifying CC.
///
/// # Safety
///
/// - For nonzero `len`, `dst` covers `len` aligned, initialized, writable limbs
///   within `isize::MAX` bytes.
/// - `borrow <= 1`.
#[expect(clippy::inline_always, reason = "Inlining keeps carry propagation within its arithmetic caller")]
#[inline(always)]
pub unsafe fn propagate_borrow_unchecked(dst: *mut Limb, len: usize, mut borrow: Limb) -> Limb {
    if borrow == 0 || len == 0 {
        return borrow;
    }
    let chunks = len >> 1;
    let rem = len & 1;
    let zero_const: Limb = 0;

    // SAFETY: the caller provides len aligned initialized writable limbs and
    // a binary borrow. Each block consumes two limbs, with len % 2 remaining.
    // SLGR seeds the borrow condition; SLBGR propagates it while BRCTG preserves
    // CC. Once cleared, subtraction by zero leaves later limbs unchanged.
    // Every modified general register is declared as an output.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",                   // If chunks == 0, skip main loop (1f)

            // Seed CC from borrow: (0 - borrow) produces CC borrow iff borrow == 1
            "lghi {cc_seed}, 0",                         // cc_seed = 0
            "slgr {cc_seed}, {borrow}",                  // Set Condition Code (CC)

            ".p2align 4",                                // Align the loop header
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
