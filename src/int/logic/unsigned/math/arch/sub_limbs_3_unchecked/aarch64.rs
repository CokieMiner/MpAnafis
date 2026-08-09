//! `AArch64` subtraction kernels (inline assembly).
//!
//! Uses the inverted carry flag (C) via `sbcs`/`cset`:
//!
//! ```text
//!   cmp   xzr, xzr          // C = 1  (no borrow)
//!   sbcs  dst, dst, src     // dst = dst − src − ¬C
//!   cset  borrow, cc        // borrow = 1 iff C == 0
//! ```
//!
//! `AArch64` `sbcs` subtracts with inverted carry: the carry flag is 1 when
//! no borrow occurred and 0 when borrow occurred.  `cmp xzr, xzr` sets
//! C = 1 for the initial state.  `cset cc` (carry clear = C == 0)
//! extracts the final borrow.
//!
//! ## Loop structure
//!
//! 4-way unrolled (`len >> 2`) using 128-bit loads/stores (`ldp`/`stp`).
//! The remainder uses single-limb `ldr`/`str`.

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
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: Assembly block accesses `len` elements from `dst`, `src1`, and `src2`, which caller guarantees are valid.
    unsafe {
        asm!(
            "cmp xzr, xzr",                    // C = 1 (no initial borrow)
            "cbz {chunks}, 1f",                // skip chunk loop if len < 4
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "ldp {src1_v0}, {src1_v1}, [{src1}], #16",  // load src1[0..1], src1 += 16
            "ldp {src2_v0}, {src2_v1}, [{src2}], #16",  // load src2[0..1], src2 += 16
            "ldp {src1_v2}, {src1_v3}, [{src1}], #16",  // load src1[2..3], src1 += 16
            "ldp {src2_v2}, {src2_v3}, [{src2}], #16",  // load src2[2..3], src2 += 16
            "sbcs {src1_v0}, {src1_v0}, {src2_v0}",     // src1[0] −= src2[0] + ¬C
            "sbcs {src1_v1}, {src1_v1}, {src2_v1}",     // src1[1] −= src2[1] + ¬C
            "sbcs {src1_v2}, {src1_v2}, {src2_v2}",     // src1[2] −= src2[2] + ¬C
            "sbcs {src1_v3}, {src1_v3}, {src2_v3}",     // src1[3] −= src2[3] + ¬C
            "stp {src1_v0}, {src1_v1}, [{dst}], #16",   // store dst[0..1], dst += 16
            "stp {src1_v2}, {src1_v3}, [{dst}], #16",   // store dst[2..3], dst += 16
            "sub {chunks}, {chunks}, #1",               // --chunks
            "cbnz {chunks}, 2b",                        // repeat if chunks != 0

            // --- Tail ---
            "1:",
            "cbz {rem}, 3f",                   // skip tail if rem == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "4:",
            "ldr {src1_v0}, [{src1}], #8",      // load src1[i], src1 += 8
            "ldr {src2_v0}, [{src2}], #8",      // load src2[i], src2 += 8
            "sbcs {src1_v0}, {src1_v0}, {src2_v0}", // src1[i] −= src2[i] + ¬C
            "str {src1_v0}, [{dst}], #8",        // store dst[i], dst += 8
            "sub {rem}, {rem}, #1",              // --rem
            "cbnz {rem}, 4b",                    // repeat if rem != 0
            "3:",
            "cset {borrow}, cc",                 // borrow = (C == 0) ? 1 : 0
            borrow = inout(reg) borrow,
            dst = inout(reg) dst => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1_v0 = out(reg) _, src1_v1 = out(reg) _, src1_v2 = out(reg) _, src1_v3 = out(reg) _,
            src2_v0 = out(reg) _, src2_v1 = out(reg) _, src2_v2 = out(reg) _, src2_v3 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
