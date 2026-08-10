//! `AArch64` addition kernels (inline assembly).
//!
//! Uses the carry flag (C) via `adds` / `adcs` instructions.
//! Small lengths (`len == 1` and `2..=4`) use dedicated straight-line carry
//! chains with `ldp` / `stp` (load/store pairs). Lengths above four use a
//! **4‑way unrolled** loop (`len >> 2`) for maximum throughput.
//!
//! ## Carry tracking
//!
//! `adds xzr, xzr, xzr` clears the C flag (0 + 0 = 0, no carry).
//! Each `adcs` instruction adds its operands plus the C flag and
//! writes the new C flag (carry out).  The final carry is extracted
//! with `cset {carry}, cs` (conditionally set to 1 if C is set).
//!
//! ## Register usage
//!
//! Four source values (`src_v0`–`src_v3`) and four destination values
//! (`dst_v0`–`dst_v3`) are kept in registers within the loop body.

use core::arch::asm;

use super::Limb;

/// Add `len` limbs from `src` into `dst` and return the final carry.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint: the kernel
///   reads `src[i]` and `dst[i]` and then writes `dst[i]`, so a partial
///   overlap is a data race.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    if len == 1 {
        // SAFETY: The caller guarantees both pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*dst).overflowing_add(*src) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if (2..=4).contains(&len) {
        // SAFETY: The caller guarantees both pointers cover `len` limbs, and
        // this branch proves the fixed kernel's `2..=4` length precondition.
        return unsafe { add_small_unchecked(dst, src, len) };
    }
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    unsafe {
        asm!(
            "adds xzr, xzr, xzr",                   // clear C flag (0 + 0 = 0)
            // ── 4‑way unrolled loop ────────────────────────────────────
            "cbz {chunks}, 1f",                      // skip main loop if chunks == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "ldp {src_v0}, {src_v1}, [{src}], #16",  // load src[0], src[1]; src += 16
            "ldp {dst_v0}, {dst_v1}, [{dst}]",        // load dst[0], dst[1]
            "ldp {src_v2}, {src_v3}, [{src}], #16",  // load src[2], src[3]; src += 16
            "ldp {dst_v2}, {dst_v3}, [{dst}, #16]",  // load dst[2], dst[3] (offset 16)
            "adcs {dst_v0}, {dst_v0}, {src_v0}",     // dst[0] += src[0] + C
            "adcs {dst_v1}, {dst_v1}, {src_v1}",     // dst[1] += src[1] + C
            "adcs {dst_v2}, {dst_v2}, {src_v2}",     // dst[2] += src[2] + C
            "adcs {dst_v3}, {dst_v3}, {src_v3}",     // dst[3] += src[3] + C
            "stp {dst_v0}, {dst_v1}, [{dst}], #16",  // store dst[0], dst[1]; dst += 16
            "stp {dst_v2}, {dst_v3}, [{dst}], #16",  // store dst[2], dst[3]; dst += 16
            "sub {chunks}, {chunks}, #1",             // decrement chunk counter
            "cbnz {chunks}, 2b",                      // loop back if chunks != 0

            // ── Tail: single‑limb remainder loop ───────────────────────
            "1:",
            "cbz {rem}, 3f",                          // skip tail if rem == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "4:",
            "ldr {src_v0}, [{src}], #8",              // load src limb; src += 8
            "ldr {dst_v0}, [{dst}]",                   // load dst limb
            "adcs {dst_v0}, {dst_v0}, {src_v0}",      // dst += src + C
            "str {dst_v0}, [{dst}], #8",               // store result; dst += 8
            "sub {rem}, {rem}, #1",                    // decrement remainder counter
            "cbnz {rem}, 4b",                          // loop back if rem != 0
            "3:",
            "cset {carry}, cs",                        // carry = 1 if C set, 0 otherwise
            carry = inout(reg) carry,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src_v0 = out(reg) _, src_v1 = out(reg) _, src_v2 = out(reg) _, src_v3 = out(reg) _,
            dst_v0 = out(reg) _, dst_v1 = out(reg) _, dst_v2 = out(reg) _, dst_v3 = out(reg) _,
            options(nostack)
        );
    }
    carry
}

/// Straight-line `dst[i] = dst[i] + src[i] + carry` chain for `len` in
/// `2..=4`.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint: the kernel
///   reads `src[i]` and `dst[i]` and then writes `dst[i]`, so a partial
///   overlap is a data race.
#[allow(
    clippy::inline_always,
    reason = "The fixed-size carry chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut carry: Limb;
    match len {
        2 => {
            // SAFETY: The caller guarantees `dst` and `src` have at least 2 limbs.
            unsafe {
                asm!(
                    "ldp {s0}, {s1}, [{src}]",
                    "ldp {d0}, {d1}, [{dst}]",
                    "adds {d0}, {d0}, {s0}",
                    "adcs {d1}, {d1}, {s1}",
                    "stp {d0}, {d1}, [{dst}]",
                    "cset {carry}, cs",
                    src = in(reg) src,
                    dst = in(reg) dst,
                    s0 = out(reg) _, s1 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
        }
        3 => {
            // SAFETY: The caller guarantees `dst` and `src` have at least 3 limbs.
            unsafe {
                asm!(
                    "ldp {s0}, {s1}, [{src}]",
                    "ldp {d0}, {d1}, [{dst}]",
                    "ldr {s2}, [{src}, #16]",
                    "ldr {d2}, [{dst}, #16]",
                    "adds {d0}, {d0}, {s0}",
                    "adcs {d1}, {d1}, {s1}",
                    "adcs {d2}, {d2}, {s2}",
                    "stp {d0}, {d1}, [{dst}]",
                    "str {d2}, [{dst}, #16]",
                    "cset {carry}, cs",
                    src = in(reg) src,
                    dst = in(reg) dst,
                    s0 = out(reg) _, s1 = out(reg) _, s2 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _, d2 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
        }
        _ => {
            // SAFETY: The caller guarantees `dst` and `src` have at least 4 limbs.
            unsafe {
                asm!(
                    "ldp {s0}, {s1}, [{src}]",
                    "ldp {d0}, {d1}, [{dst}]",
                    "ldp {s2}, {s3}, [{src}, #16]",
                    "ldp {d2}, {d3}, [{dst}, #16]",
                    "adds {d0}, {d0}, {s0}",
                    "adcs {d1}, {d1}, {s1}",
                    "adcs {d2}, {d2}, {s2}",
                    "adcs {d3}, {d3}, {s3}",
                    "stp {d0}, {d1}, [{dst}]",
                    "stp {d2}, {d3}, [{dst}, #16]",
                    "cset {carry}, cs",
                    src = in(reg) src,
                    dst = in(reg) dst,
                    s0 = out(reg) _, s1 = out(reg) _, s2 = out(reg) _, s3 = out(reg) _,
                    d0 = out(reg) _, d1 = out(reg) _, d2 = out(reg) _, d3 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
        }
    }
    carry
}
