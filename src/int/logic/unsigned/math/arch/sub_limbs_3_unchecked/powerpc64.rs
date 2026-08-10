//! `PowerPC64` subtraction kernels (inline assembly).
//!
//! Uses the XER[CA] (carry) bit for borrow propagation:
//!
//! ```text
//!   li     borrow, 0
//!   subfc  borrow, borrow, borrow    // CA = 1  (no initial borrow)
//!   ...
//!   subfe  dst, src, dst            // dst = dst − src + CA − 1
//!   ...
//!   li     borrow, 0
//!   subfe  borrow, borrow, borrow   // borrow = CA − 1  (0 or −1)
//!   neg    borrow, borrow           // borrow = 0 or 1
//! ```
//!
//! `subfe` (subtract from extended) computes `¬RA + RB + CA`, effectively
//! `RB − RA + CA − 1`.  When CA = 1 (no prior borrow) this gives `RB − RA`;
//! when CA = 0 (prior borrow) this subtracts an extra 1.
//!
//! The loop uses the CTR (count register) with `bdnz` for zero-overhead
//! loop control.
//!
//! ## Loop structure
//!
//! 4-way unrolled (`len >> 2`) with a single-limb tail for the remainder.

use core::arch::asm;

use super::Limb;

// ── sub_n_3 (3-operand) ───────────────────────────────────────────────────

/// Compute `dst[i] = src1[i] − src2[i] − borrow` for `len` limbs,
/// returning the final borrow.
///
/// # Safety
///
/// `dst`, `src1`, and `src2` must each be valid for `len` elements.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
#[inline(always)]
pub unsafe fn sub_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    let mut borrow: Limb;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: Assembly block accesses `len` elements from `dst`, `src1`, and `src2`, which caller guarantees are valid.
    unsafe {
        asm!(
            "subfc {borrow}, {borrow}, {borrow}",   // CA = 1 (no initial borrow)

            "cmpldi {chunks}, 0",                   // test chunks
            "beq 1f",                                // skip chunk loop if chunks == 0
            "mtctr {chunks}",                        // CTR = chunks
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "ld {src1_v0}, 0({src1})",              // load src1[0..3]
            "ld {src1_v1}, 8({src1})",
            "ld {src1_v2}, 16({src1})",
            "ld {src1_v3}, 24({src1})",
            "ld {src2_v0}, 0({src2})",              // load src2[0..3]
            "ld {src2_v1}, 8({src2})",
            "ld {src2_v2}, 16({src2})",
            "ld {src2_v3}, 24({src2})",
            "subfe {src1_v0}, {src2_v0}, {src1_v0}", // src1[i] = src1[i] − src2[i] + CA − 1
            "subfe {src1_v1}, {src2_v1}, {src1_v1}",
            "subfe {src1_v2}, {src2_v2}, {src1_v2}",
            "subfe {src1_v3}, {src2_v3}, {src1_v3}",
            "std {src1_v0}, 0({dst})",              // store dst[0..3]
            "std {src1_v1}, 8({dst})",
            "std {src1_v2}, 16({dst})",
            "std {src1_v3}, 24({dst})",
            "addi {src1}, {src1}, 32",              // src1 += 32
            "addi {src2}, {src2}, 32",              // src2 += 32
            "addi {dst}, {dst}, 32",                // dst  += 32
            "bdnz 2b",                               // --CTR; branch if CTR != 0

            // --- Tail ---
            "1:",
            "cmpldi {rem}, 0",                      // test rem
            "beq 3f",                                // skip tail if rem == 0
            "mtctr {rem}",                           // CTR = rem
            "addi {src1}, {src1}, -8",
            "addi {src2}, {src2}, -8",
            "addi {dst}, {dst}, -8",
            ".p2align 4",                          // align loop header for fetch efficiency
            "4:",
            "ldu {src1_v0}, 8({dst})",              // advance dst (throwaway load)
            "ldu {src1_v0}, 8({src1})",             // load src1[i], advance src1
            "ldu {src2_v0}, 8({src2})",             // load src2[i], advance src2
            "subfe {src1_v0}, {src2_v0}, {src1_v0}", // src1[i] −= src2[i] + CA − 1
            "std {src1_v0}, 0({dst})",              // store dst[i]
            "bdnz 4b",                               // --CTR; branch if CTR != 0

            "3:",
            "li {borrow}, 0",
            "subfe {borrow}, {borrow}, {borrow}",   // borrow = CA − 1  (0 or −1)
            "neg {borrow}, {borrow}",                // borrow = 0 or 1
            borrow = out(reg) borrow,
            dst = inout(reg_nonzero) dst => _,
            src1 = inout(reg_nonzero) src1 => _,
            src2 = inout(reg_nonzero) src2 => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1_v0 = out(reg) _, src1_v1 = out(reg) _, src1_v2 = out(reg) _, src1_v3 = out(reg) _,
            src2_v0 = out(reg) _, src2_v1 = out(reg) _, src2_v2 = out(reg) _, src2_v3 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    borrow
}
