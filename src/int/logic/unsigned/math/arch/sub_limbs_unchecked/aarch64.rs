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

// ── sub_n ──────────────────────────────────────────────────────────────────

/// Subtract `len` limbs of `src` from `dst` and return the final borrow.
///
/// ```text
///   (borrow, dst[0..len]) = dst[0..len] − src[0..len]
/// ```
///
/// # Safety
///
/// `dst` and `src` must each be valid for `len` elements of type `Limb`.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    unsafe {
        asm!(
            "cmp xzr, xzr",                    // C = 1 (no initial borrow)
            "cbz {chunks}, 1f",                // skip chunk loop if len < 4
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "ldp {src_v0}, {src_v1}, [{src}], #16",   // load src[0..1], src += 16
            "ldp {dst_v0}, {dst_v1}, [{dst}]",         // load dst[0..1]
            "ldp {src_v2}, {src_v3}, [{src}], #16",    // load src[2..3], src += 16
            "ldp {dst_v2}, {dst_v3}, [{dst}, #16]",    // load dst[2..3] (offset 16)
            "sbcs {dst_v0}, {dst_v0}, {src_v0}",       // dst[0] −= src[0] + ¬C
            "sbcs {dst_v1}, {dst_v1}, {src_v1}",       // dst[1] −= src[1] + ¬C
            "sbcs {dst_v2}, {dst_v2}, {src_v2}",       // dst[2] −= src[2] + ¬C
            "sbcs {dst_v3}, {dst_v3}, {src_v3}",       // dst[3] −= src[3] + ¬C
            "stp {dst_v0}, {dst_v1}, [{dst}], #16",    // store dst[0..1], dst += 16
            "stp {dst_v2}, {dst_v3}, [{dst}], #16",    // store dst[2..3], dst += 16
            "sub {chunks}, {chunks}, #1",              // --chunks
            "cbnz {chunks}, 2b",                       // repeat if chunks != 0

            // --- Tail ---
            "1:",
            "cbz {rem}, 3f",                   // skip tail if rem == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "4:",
            "ldr {src_v0}, [{src}], #8",       // load src[i], src += 8
            "ldr {dst_v0}, [{dst}]",            // load dst[i]
            "sbcs {dst_v0}, {dst_v0}, {src_v0}", // dst[i] −= src[i] + ¬C
            "str {dst_v0}, [{dst}], #8",        // store dst[i], dst += 8
            "sub {rem}, {rem}, #1",             // --rem
            "cbnz {rem}, 4b",                   // repeat if rem != 0
            "3:",
            "cset {borrow}, cc",                // borrow = (C == 0) ? 1 : 0
            borrow = inout(reg) borrow,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src_v0 = out(reg) _, src_v1 = out(reg) _, src_v2 = out(reg) _, src_v3 = out(reg) _,
            dst_v0 = out(reg) _, dst_v1 = out(reg) _, dst_v2 = out(reg) _, dst_v3 = out(reg) _,
            options(nostack)
        );
    }
    borrow
}
