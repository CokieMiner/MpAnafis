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

/// Compute `dst[i] = src1[i] + src2[i] + carry` for `len` limbs, returning
/// the final carry.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span: the kernel writes `dst`
///   while it reads `src1` and `src2`.
/// - `src1` and `src2` are read-only and may alias each other.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 1 {
        // SAFETY: The caller guarantees all pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if (2..=4).contains(&len) {
        // SAFETY: The caller guarantees all pointers cover `len` limbs, and
        // this branch proves the fixed kernel's `2..=4` length precondition.
        return unsafe { add_small_3_unchecked(dst, src1, src2, len) };
    }
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: Assembly block accesses `len` elements from `dst`, `src1`, and `src2`, which caller guarantees are valid.
    unsafe {
        asm!(
            "adds xzr, xzr, xzr",                        // clear C flag
            // ── 4‑way unrolled loop ────────────────────────────────────
            "cbz {chunks}, 1f",                           // skip main loop if chunks == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "ldp {src1_v0}, {src1_v1}, [{src1}], #16",   // load src1[0], src1[1]; src1 += 16
            "ldp {src2_v0}, {src2_v1}, [{src2}], #16",   // load src2[0], src2[1]; src2 += 16
            "ldp {src1_v2}, {src1_v3}, [{src1}], #16",   // load src1[2], src1[3]; src1 += 16
            "ldp {src2_v2}, {src2_v3}, [{src2}], #16",   // load src2[2], src2[3]; src2 += 16
            "adcs {src1_v0}, {src1_v0}, {src2_v0}",      // src1[0] += src2[0] + C
            "adcs {src1_v1}, {src1_v1}, {src2_v1}",      // src1[1] += src2[1] + C
            "adcs {src1_v2}, {src1_v2}, {src2_v2}",      // src1[2] += src2[2] + C
            "adcs {src1_v3}, {src1_v3}, {src2_v3}",      // src1[3] += src2[3] + C
            "stp {src1_v0}, {src1_v1}, [{dst}], #16",    // store dst[0], dst[1]; dst += 16
            "stp {src1_v2}, {src1_v3}, [{dst}], #16",    // store dst[2], dst[3]; dst += 16
            "sub {chunks}, {chunks}, #1",                  // decrement chunk counter
            "cbnz {chunks}, 2b",                           // loop back if chunks != 0

            // ── Tail: single‑limb remainder loop ───────────────────────
            "1:",
            "cbz {rem}, 3f",                               // skip tail if rem == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "4:",
            "ldr {src1_v0}, [{src1}], #8",                // load src1 limb; src1 += 8
            "ldr {src2_v0}, [{src2}], #8",                // load src2 limb; src2 += 8
            "adcs {src1_v0}, {src1_v0}, {src2_v0}",       // src1 += src2 + C
            "str {src1_v0}, [{dst}], #8",                  // store result; dst += 8
            "sub {rem}, {rem}, #1",                         // decrement remainder
            "cbnz {rem}, 4b",                               // loop back if rem != 0
            "3:",
            "cset {carry}, cs",                             // carry = C (condition code CS)
            carry = inout(reg) carry,
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
    carry
}

/// Straight-line `dst[i] = src1[i] + src2[i] + carry` chain for `len` in
/// `2..=4`.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span: it is written while `src1`
///   and `src2` are read.
/// - `src1` and `src2` are read-only and may alias each other.
#[allow(
    clippy::inline_always,
    reason = "The fixed-size carry chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn add_small_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    let mut carry: Limb;
    match len {
        2 => {
            // SAFETY: The caller guarantees `dst`, `src1`, and `src2` have at least 2 limbs.
            unsafe {
                asm!(
                    "ldp {s1_0}, {s1_1}, [{src1}]",
                    "ldp {s2_0}, {s2_1}, [{src2}]",
                    "adds {s1_0}, {s1_0}, {s2_0}",
                    "adcs {s1_1}, {s1_1}, {s2_1}",
                    "stp {s1_0}, {s1_1}, [{dst}]",
                    "cset {carry}, cs",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    s1_0 = out(reg) _, s1_1 = out(reg) _,
                    s2_0 = out(reg) _, s2_1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
        }
        3 => {
            // SAFETY: The caller guarantees `dst`, `src1`, and `src2` have at least 3 limbs.
            unsafe {
                asm!(
                    "ldp {s1_0}, {s1_1}, [{src1}]",
                    "ldp {s2_0}, {s2_1}, [{src2}]",
                    "ldr {s1_2}, [{src1}, #16]",
                    "ldr {s2_2}, [{src2}, #16]",
                    "adds {s1_0}, {s1_0}, {s2_0}",
                    "adcs {s1_1}, {s1_1}, {s2_1}",
                    "adcs {s1_2}, {s1_2}, {s2_2}",
                    "stp {s1_0}, {s1_1}, [{dst}]",
                    "str {s1_2}, [{dst}, #16]",
                    "cset {carry}, cs",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    s1_0 = out(reg) _, s1_1 = out(reg) _, s1_2 = out(reg) _,
                    s2_0 = out(reg) _, s2_1 = out(reg) _, s2_2 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
        }
        _ => {
            // SAFETY: The caller guarantees `dst`, `src1`, and `src2` have at least 4 limbs.
            unsafe {
                asm!(
                    "ldp {s1_0}, {s1_1}, [{src1}]",
                    "ldp {s2_0}, {s2_1}, [{src2}]",
                    "ldp {s1_2}, {s1_3}, [{src1}, #16]",
                    "ldp {s2_2}, {s2_3}, [{src2}, #16]",
                    "adds {s1_0}, {s1_0}, {s2_0}",
                    "adcs {s1_1}, {s1_1}, {s2_1}",
                    "adcs {s1_2}, {s1_2}, {s2_2}",
                    "adcs {s1_3}, {s1_3}, {s2_3}",
                    "stp {s1_0}, {s1_1}, [{dst}]",
                    "stp {s1_2}, {s1_3}, [{dst}, #16]",
                    "cset {carry}, cs",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
                    dst = in(reg) dst,
                    s1_0 = out(reg) _, s1_1 = out(reg) _, s1_2 = out(reg) _, s1_3 = out(reg) _,
                    s2_0 = out(reg) _, s2_1 = out(reg) _, s2_2 = out(reg) _, s2_3 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
        }
    }
    carry
}
