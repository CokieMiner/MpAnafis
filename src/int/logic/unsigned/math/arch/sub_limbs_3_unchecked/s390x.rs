//! `s390x` (IBM Z) subtraction kernels (inline assembly).
//!
//! Two-limb SLBGR blocks carry the borrow in the condition code (CC), followed
//! by a single-limb tail. CC=0/1 means borrow; CC=2/3 means no borrow.
//! SLGR of a register with itself seeds CC=2. LA and BRCTG preserve CC.

use core::arch::asm;

use super::Limb;

/// Compute `dst[i] = src1[i] - src2[i] - borrow` for `len` limbs,
/// returning the final borrow.
///
/// # Safety
///
/// Both sources must cover `len` aligned initialized readable limbs and may
/// alias each other. `dst` must cover `len` aligned writable limbs, disjoint
/// from both sources; its contents may be uninitialized. Each byte span must
/// fit in `isize::MAX`. `len == 0` performs no pointer access.
#[expect(clippy::inline_always, reason = "Inlining exposes the two-limb assembly borrow chain to callers")]
#[inline(always)]
pub unsafe fn sub_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    let mut borrow: Limb;
    let chunks = len >> 1;
    let rem = len & 1;
    // SAFETY: two-limb blocks and the len&1 tail access exactly len initialized
    // aligned source limbs and disjoint writable output limbs. Sources may
    // coincide. SLGR seeds no borrow, and LA/BRCTG preserve CC until extraction.
    // Every changed register is an early output; dst and stack memory are not read.
    unsafe {
        asm!(
            "cgij {chunks}, 0, 8, 1f",          // skip main loop if chunks == 0

            // chunks > 0: seed no borrow before the main loop
            "lghi {borrow}, 0",
            "slgr {borrow}, {borrow}",          // CC = 2 (no borrow)
            ".p2align 4",                          // 16-byte loop alignment
            "2:",
            "lg {src1_val0}, 0({src1})",        // load src1[0]
            "lg {src2_val0}, 0({src2})",        // load src2[0]
            "slbgr {src1_val0}, {src2_val0}",   // src1_val0 = src1[0] - src2[0] - borrow
            "stg {src1_val0}, 0({dst})",        // store dst[0]
            "lg {src1_val1}, 8({src1})",        // load src1[1]
            "lg {src2_val1}, 8({src2})",        // load src2[1]
            "slbgr {src1_val1}, {src2_val1}",   // src1_val1 = src1[1] - src2[1] - borrow
            "stg {src1_val1}, 8({dst})",        // store dst[1]
            "la {src1}, 16({src1})",            // src1 += 16
            "la {src2}, 16({src2})",            // src2 += 16
            "la {dst}, 16({dst})",              // dst  += 16
            "brctg {chunks}, 2b",               // --chunks; branch if != 0 (preserves CC)
            "j 4f",                              // skip CC re-init (main loop done, CC already set)

            "1:",                                 // chunks == 0: seed no borrow for tail
            "lghi {borrow}, 0",
            "slgr {borrow}, {borrow}",           // CC = 2 (no borrow)

            "4:",                                 // common: CC is ready
            "brctg {rem}, 3f",                  // if rem was 0 -> skip tail
            "lg {src1_val0}, 0({src1})",        // load last limb
            "lg {src2_val0}, 0({src2})",        // load last src2
            "slbgr {src1_val0}, {src2_val0}",   // src1 - src2 - borrow
            "stg {src1_val0}, 0({dst})",        // store result
            "3:",
            "lghi {borrow}, 0",
            "slbgr {borrow}, {borrow}",         // borrow = -borrow_from_CC (0 or -1)
            "lcgr {borrow}, {borrow}",          // two's complement -> 0 or 1
            borrow = out(reg) borrow,
            dst = inout(reg_addr) dst => _,
            src1 = inout(reg_addr) src1 => _,
            src2 = inout(reg_addr) src2 => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1_val0 = out(reg) _, src2_val0 = out(reg) _,
            src1_val1 = out(reg) _, src2_val1 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
