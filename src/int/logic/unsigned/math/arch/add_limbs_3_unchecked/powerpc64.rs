//! `PowerPC64` addition into a disjoint destination.
//!
//! `adde` propagates XER[CA] through four-limb blocks and a scalar tail.
//! `addic` initializes CA to zero; `addze` extracts the final binary carry.
//! CTR counts blocks without modifying CA.

use core::{arch::asm, hint::unreachable_unchecked};

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
    if len == 0 {
        return 0;
    }
    if len == 1 {
        // SAFETY: len == 1 bounds both aligned, initialized source spans.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: len == 1 bounds the aligned, writable destination span.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if len <= 4 {
        // SAFETY: prior returns and len <= 4 prove 2 <= len <= 4. The caller
        // supplies aligned sources and a disjoint writable destination.
        return unsafe { add_small_3_unchecked(dst, src1, src2, len) };
    }
    let mut carry: Limb;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: len > 4 gives chunks > 0; 4 * chunks + rem == len bounds every
    // aligned read and write. Tail predecrements stay within the processed
    // prefix. The sources are initialized and may overlap; dst is disjoint
    // and only written. All modified GPRs, CTR, XER, and CR0 are declared.
    unsafe {
        asm!(
            "addic {carry}, {rem}, 0",                // XER[CA] = 0
            // -- 4-way unrolled loop (using CTR) ------------------------
            "mtctr {chunks}",                       // CTR = chunks
            ".p2align 4",                          // align the loop to 16 bytes
            "2:",
            "ld {src1_v0}, 0({src1})",            // load src1[0]
            "ld {src1_v1}, 8({src1})",            // load src1[1]
            "ld {src1_v2}, 16({src1})",           // load src1[2]
            "ld {src1_v3}, 24({src1})",           // load src1[3]
            "ld {src2_v0}, 0({src2})",            // load src2[0]
            "ld {src2_v1}, 8({src2})",            // load src2[1]
            "ld {src2_v2}, 16({src2})",           // load src2[2]
            "ld {src2_v3}, 24({src2})",           // load src2[3]
            "adde {src1_v0}, {src1_v0}, {src2_v0}", // src1[0] += src2[0] + XER[CA]
            "adde {src1_v1}, {src1_v1}, {src2_v1}", // src1[1] += src2[1] + XER[CA]
            "adde {src1_v2}, {src1_v2}, {src2_v2}", // src1[2] += src2[2] + XER[CA]
            "adde {src1_v3}, {src1_v3}, {src2_v3}", // src1[3] += src2[3] + XER[CA]
            "std {src1_v0}, 0({dst})",            // store dst[0]
            "std {src1_v1}, 8({dst})",            // store dst[1]
            "std {src1_v2}, 16({dst})",           // store dst[2]
            "std {src1_v3}, 24({dst})",           // store dst[3]
            "addi {src1}, {src1}, 32",            // advance src1 by 32 bytes
            "addi {src2}, {src2}, 32",            // advance src2 by 32 bytes
            "addi {dst}, {dst}, 32",              // advance dst by 32 bytes
            "bdnz 2b",                             // CTR--, branch if CTR != 0
            // -- Tail: single-limb remainder loop -----------------------
            "1:",
            "cmpldi {rem}, 0",                     // compare rem with 0
            "beq 3f",                               // skip tail if rem == 0
            "mtctr {rem}",                          // CTR = rem
            "addi {src1}, {src1}, -8",
            "addi {src2}, {src2}, -8",
            "addi {dst}, {dst}, -8",
            ".p2align 4",                          // align the loop to 16 bytes
            "4:",
            "ldu {src1_v0}, 8({src1})",           // load src1 limb, advance src1
            "ldu {src2_v0}, 8({src2})",           // load src2 limb, advance src2
            "adde {src1_v0}, {src1_v0}, {src2_v0}", // src1 += src2 + XER[CA]
            "stdu {src1_v0}, 8({dst})",           // write result and advance dst without reading it
            "bdnz 4b",                              // CTR--, branch if CTR != 0
            "3:",
            "li {carry}, 0",                        // carry = 0
            "addze {carry}, {carry}",               // carry = 0 + 0 + XER[CA]
            carry = out(reg) carry,
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
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: len == 2 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
            unsafe {
                asm!(
                    "ld {a0}, 0({src1})",
                    "ld {a1}, 8({src1})",
                    "ld {b0}, 0({src2})",
                    "ld {b1}, 8({src2})",
                    "addc {a0}, {a0}, {b0}",
                    "adde {a1}, {a1}, {b1}",
                    "std {a0}, 0({dst})",
                    "std {a1}, 8({dst})",
                    "addze {carry}, {zero}",
                    src1 = inout(reg_nonzero) src1 => _,
                    src2 = inout(reg_nonzero) src2 => _,
                    dst = inout(reg_nonzero) dst => _,
                    zero = inout(reg) 0_usize => _,
                    a0 = out(reg) _, a1 = out(reg) _,
                    b0 = out(reg) _, b1 = out(reg) _,
                    carry = out(reg) carry,
                    out("xer") _,
                    options(nostack)
                );
            }
            carry
        }
        3 => {
            let mut carry: Limb;
            // SAFETY: len == 3 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
            unsafe {
                asm!(
                    "ld {a0}, 0({src1})",
                    "ld {a1}, 8({src1})",
                    "ld {a2}, 16({src1})",
                    "ld {b0}, 0({src2})",
                    "ld {b1}, 8({src2})",
                    "ld {b2}, 16({src2})",
                    "addc {a0}, {a0}, {b0}",
                    "adde {a1}, {a1}, {b1}",
                    "adde {a2}, {a2}, {b2}",
                    "std {a0}, 0({dst})",
                    "std {a1}, 8({dst})",
                    "std {a2}, 16({dst})",
                    "addze {carry}, {zero}",
                    src1 = inout(reg_nonzero) src1 => _,
                    src2 = inout(reg_nonzero) src2 => _,
                    dst = inout(reg_nonzero) dst => _,
                    zero = inout(reg) 0_usize => _,
                    a0 = out(reg) _, a1 = out(reg) _, a2 = out(reg) _,
                    b0 = out(reg) _, b1 = out(reg) _, b2 = out(reg) _,
                    carry = out(reg) carry,
                    out("xer") _,
                    options(nostack)
                );
            }
            carry
        }
        4 => {
            let mut carry: Limb;
            // SAFETY: len == 4 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
            unsafe {
                asm!(
                    "ld {a0}, 0({src1})",
                    "ld {a1}, 8({src1})",
                    "ld {a2}, 16({src1})",
                    "ld {a3}, 24({src1})",
                    "ld {b0}, 0({src2})",
                    "ld {b1}, 8({src2})",
                    "ld {b2}, 16({src2})",
                    "ld {b3}, 24({src2})",
                    "addc {a0}, {a0}, {b0}",
                    "adde {a1}, {a1}, {b1}",
                    "adde {a2}, {a2}, {b2}",
                    "adde {a3}, {a3}, {b3}",
                    "std {a0}, 0({dst})",
                    "std {a1}, 8({dst})",
                    "std {a2}, 16({dst})",
                    "std {a3}, 24({dst})",
                    "addze {carry}, {zero}",
                    src1 = inout(reg_nonzero) src1 => _,
                    src2 = inout(reg_nonzero) src2 => _,
                    dst = inout(reg_nonzero) dst => _,
                    zero = inout(reg) 0_usize => _,
                    a0 = out(reg) _, a1 = out(reg) _, a2 = out(reg) _, a3 = out(reg) _,
                    b0 = out(reg) _, b1 = out(reg) _, b2 = out(reg) _, b3 = out(reg) _,
                    carry = out(reg) carry,
                    out("xer") _,
                    options(nostack)
                );
            }
            carry
        }
        // SAFETY: the caller establishes 2 <= len <= 4, all matched above.
        _ => unsafe { unreachable_unchecked() },
    }
}
