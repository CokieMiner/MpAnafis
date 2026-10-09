//! `AArch64` in-place addition with four-limb carry chains.
//!
//! `adds` initializes C, `adcs` propagates it, and `cset` extracts the final
//! carry. Each block loads its source and destination before writing results.

use core::arch::asm;

use super::Limb;

/// Adds `src` to `dst` and returns the binary carry.
///
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// Nonempty spans must be aligned, initialized, and cover `len` limbs in
/// live allocations of at most `isize::MAX` bytes. The destination must be
/// writable. The two spans must be disjoint or exactly identical.
#[expect(
    clippy::inline_always,
    reason = "Keep the hardware carry chain in the selected arithmetic caller"
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
    let carry: Limb;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: 4 * chunks + rem == len bounds all aligned, initialized accesses.
    // Each block reads its inputs before writing, permitting exact aliasing.
    // Empty spans skip all memory access; cset defines carry on every path.
    unsafe {
        asm!(
            "adds xzr, xzr, xzr",                   // clear C flag (0 + 0 = 0)
            // -- 4-way unrolled loop ------------------------------------
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

            // -- Tail: single-limb remainder loop -----------------------
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
            carry = out(reg) carry,
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

/// Adds two to four limbs with a straight carry chain.
///
/// # Safety
///
/// The pointer contract of [`add_limbs_unchecked`] applies and `2 <= len <= 4`.
#[expect(
    clippy::inline_always,
    reason = "The fixed-size carry chains must inline into the public kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut carry: Limb;
    match len {
        2 => {
            // SAFETY: len == 2 bounds both initialized spans; paired loads
            // precede stores, including when the pointers are identical.
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
            // SAFETY: len == 3 bounds both initialized spans; all source
            // and destination loads precede stores, permitting exact aliasing.
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
            // SAFETY: 2 <= len <= 4 and the prior arms imply len == 4.
            // All aligned input limbs are loaded before any result is stored.
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
