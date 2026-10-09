//! `s390x` (IBM Z) carry propagation kernel (inline assembly).
//!
//! Propagates a carry through an array of limbs in place, utilizing 2-way unrolled
//! `alcgr` (add logical with carry) and CC-neutral `brctg` branching.

#![expect(unsafe_code, reason = "The kernel accesses caller-proven raw spans and s390x assembly")]

use core::arch::asm;

use super::Limb;

/// Propagate carry through `dst` slice in-place.
///
/// Returns the final carry-out (0 or 1).
///
/// Uses `alcgr` to propagate carry across destination limbs, seeding the Condition Code (CC)
/// via `algr` with `u64::MAX`. `brctg` decrements the loop counter without modifying CC.
///
/// # Safety
///
/// - For nonzero `len`, `dst` covers `len` aligned, initialized, writable limbs
///   within `isize::MAX` bytes.
/// - `carry <= 1`.
#[expect(clippy::inline_always, reason = "Inlining keeps carry propagation within its arithmetic caller")]
#[inline(always)]
pub unsafe fn propagate_carry_unchecked(dst: *mut Limb, len: usize, mut carry: Limb) -> Limb {
    if carry == 0 || len == 0 {
        return carry;
    }
    let chunks = len >> 1;
    let rem = len & 1;
    let zero_const: Limb = 0;

    // SAFETY: the caller provides len aligned initialized writable limbs and
    // a binary carry. Each block consumes two limbs, with len % 2 remaining.
    // ALGR seeds the carry condition; ALCGR propagates it while BRCTG preserves
    // CC. Once cleared, addition by zero leaves later limbs unchanged.
    // Every modified general register is declared as an output.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",                   // If chunks == 0, skip main loop (1f)

            // Seed CC from carry: (u64::MAX + carry) produces CC carry iff carry == 1
            "lghi {cc_seed}, -1",                        // cc_seed = u64::MAX
            "algr {cc_seed}, {carry}",                   // Set Condition Code (CC)

            ".p2align 4",                                // Align the loop header
            // 2-way unrolled main loop
            "2:",                                        // Loop head label
            "lg {val0}, 0({dst})",                       // Load dst[j]
            "lg {val1}, 8({dst})",                       // Load dst[j+1]
            "alcgr {val0}, {zero}",                      // val0 += CC carry + 0
            "alcgr {val1}, {zero}",                      // val1 += CC carry + 0
            "stg {val0}, 0({dst})",                      // Store updated dst[j]
            "stg {val1}, 8({dst})",                      // Store updated dst[j+1]
            "la {dst}, 16({dst})",                       // Advance dst pointer (+16)
            "brctg {chunks}, 2b",                        // Decrement chunks and branch if > 0 (CC-neutral)
            "j 4f",                                      // Jump to tail/exit

            // Remainder entry seed path
            "1:",                                        // Remainder entry label
            "lghi {cc_seed}, -1",                        // cc_seed = u64::MAX
            "algr {cc_seed}, {carry}",                   // Set CC

            // 1-limb tail
            "4:",                                        // Tail loop label
            "brctg {rem}, 3f",                           // If rem == 0, skip tail (3f)
            "lg {val0}, 0({dst})",                       // Load single limb
            "alcgr {val0}, {zero}",                      // Add carry
            "stg {val0}, 0({dst})",                      // Store limb
            "3:",                                        // Remainder exit label

            // Capture final carry out of CC
            "5:",                                        // Exit label
            "lghi {carry}, 0",                           // carry = 0
            "alcgr {carry}, {carry}",                    // carry = 0 + 0 + CC carry (0 or 1)

            carry = inout(reg) carry,
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
    carry
}
