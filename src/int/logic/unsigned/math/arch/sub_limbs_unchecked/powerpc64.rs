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

// ── sub_n ──────────────────────────────────────────────────────────────────

/// Subtract `len` limbs of `src` from `dst` and return the final borrow.
///
/// ```text
///   (borrow, dst[0..len]) = dst[0..len] − src[0..len]
/// ```
///
/// # Safety
///
/// `dst` and `src` must each be valid for `len` elements of type `u64`.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow: Limb;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    unsafe {
        asm!(
            "subfc {borrow}, {borrow}, {borrow}",   // CA = 1 (no initial borrow)
            "cmpldi {chunks}, 0",                   // test chunks
            "beq 1f",                                // skip chunk loop if chunks == 0
            "mtctr {chunks}",                        // CTR = chunks
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "ld {src_v0}, 0({src})",                // load src[0..3]
            "ld {src_v1}, 8({src})",
            "ld {src_v2}, 16({src})",
            "ld {src_v3}, 24({src})",
            "ld {dst_v0}, 0({dst})",                // load dst[0..3]
            "ld {dst_v1}, 8({dst})",
            "ld {dst_v2}, 16({dst})",
            "ld {dst_v3}, 24({dst})",
            "subfe {dst_v0}, {src_v0}, {dst_v0}",   // dst[i] = dst[i] − src[i] + CA − 1
            "subfe {dst_v1}, {src_v1}, {dst_v1}",
            "subfe {dst_v2}, {src_v2}, {dst_v2}",
            "subfe {dst_v3}, {src_v3}, {dst_v3}",
            "std {dst_v0}, 0({dst})",               // store dst[0..3]
            "std {dst_v1}, 8({dst})",
            "std {dst_v2}, 16({dst})",
            "std {dst_v3}, 24({dst})",
            "addi {src}, {src}, 32",                // src += 32
            "addi {dst}, {dst}, 32",                // dst += 32
            "bdnz 2b",                               // --CTR; branch if CTR != 0
            // --- Tail ---
            "1:",
            "cmpldi {rem}, 0",                      // test rem
            "beq 3f",                                // skip tail if rem == 0
            "mtctr {rem}",                           // CTR = rem
            "addi {src}, {src}, -8",
            "addi {dst}, {dst}, -8",
            ".p2align 4",                          // align loop header for fetch efficiency
            "4:",
            "ldu {src_v0}, 8({src})",               // load src[i], advance src
            "ldu {dst_v0}, 8({dst})",               // load dst[i], advance dst
            "subfe {dst_v0}, {src_v0}, {dst_v0}",   // dst[i] −= src[i] + CA − 1
            "std {dst_v0}, 0({dst})",               // store dst[i]
            "bdnz 4b",                               // --CTR; branch if CTR != 0
            "3:",
            "li {borrow}, 0",
            "subfe {borrow}, {borrow}, {borrow}",   // borrow = CA − 1  (0 or −1)
            "neg {borrow}, {borrow}",                // borrow = 0 or 1
            borrow = out(reg) borrow,
            dst = inout(reg_nonzero) dst => _,
            src = inout(reg_nonzero) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src_v0 = out(reg) _, src_v1 = out(reg) _, src_v2 = out(reg) _, src_v3 = out(reg) _,
            dst_v0 = out(reg) _, dst_v1 = out(reg) _, dst_v2 = out(reg) _, dst_v3 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    borrow
}
