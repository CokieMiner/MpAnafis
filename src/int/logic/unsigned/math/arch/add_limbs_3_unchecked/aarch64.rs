//! `AArch64` addition into a disjoint destination.
//!
//! `adcs` propagates C through four-limb blocks and a scalar tail.
//! `adds xzr, xzr, xzr` initializes C to zero; `cset` extracts the final carry.
//! Pair loads and stores handle fixed lengths two through four.

use core::arch::asm;

use super::Limb;

/// Writes the low `len` limbs of `src1 + src2` and returns the binary carry.
///
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// For nonempty spans, all pointers must be aligned and cover `len` limbs.
/// Each span must lie within one live allocation and have at most `isize::MAX` bytes.
/// The sources must be initialized and readable; the destination must be
/// writable and disjoint from both sources. Its previous contents are unused.
/// The two source spans may overlap each other.
#[expect(
    clippy::inline_always,
    reason = "Keep the hardware carry chain in the selected arithmetic caller"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 1 {
        // SAFETY: len == 1 bounds both aligned, initialized source spans.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: len == 1 bounds the aligned, writable destination span.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if (2..=4).contains(&len) {
        // SAFETY: The caller guarantees all pointers cover `len` limbs, and
        // this branch proves 2 <= len <= 4; dst is disjoint from both sources.
        return unsafe { add_small_3_unchecked(dst, src1, src2, len) };
    }
    let carry: Limb;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: 4 * chunks + rem == len bounds every aligned read and write.
    // Zero counts skip memory access. Both sources are initialized and may
    // overlap; dst is disjoint and only written. Pointer advances stop at the
    // span ends, and every modified GPR is declared as an output.
    unsafe {
        asm!(
            "adds xzr, xzr, xzr",                        // clear C flag
            // -- 4-way unrolled loop ------------------------------------
            "cbz {chunks}, 1f",                           // skip main loop if chunks == 0
            ".p2align 4",                          // align the loop to 16 bytes
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

            // -- Tail: single-limb remainder loop -----------------------
            "1:",
            "cbz {rem}, 3f",                               // skip tail if rem == 0
            ".p2align 4",                          // align the loop to 16 bytes
            "4:",
            "ldr {src1_v0}, [{src1}], #8",                // load src1 limb; src1 += 8
            "ldr {src2_v0}, [{src2}], #8",                // load src2 limb; src2 += 8
            "adcs {src1_v0}, {src1_v0}, {src2_v0}",       // src1 += src2 + C
            "str {src1_v0}, [{dst}], #8",                  // store result; dst += 8
            "sub {rem}, {rem}, #1",                         // decrement remainder
            "cbnz {rem}, 4b",                               // loop back if rem != 0
            "3:",
            "cset {carry}, cs",                             // carry = C (condition code CS)
            carry = out(reg) carry,
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

/// Writes a sum with a straight carry chain for `2 <= len <= 4`.
///
/// # Safety
///
/// `len` must be in `2..=4`. Both aligned sources must contain `len`
/// initialized limbs. The aligned destination must cover `len` writable limbs
/// and be disjoint from both sources; the sources may overlap each other.
#[expect(
    clippy::inline_always,
    reason = "Keep fixed-size carry chains inside the selected kernel"
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
            // SAFETY: len == 2 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
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
            // SAFETY: len == 3 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
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
            // SAFETY: the caller establishes 2 <= len <= 4, so this branch
            // has len == 4. Aligned reads and disjoint writes cover four limbs.
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
