//! `PowerPC32` addition into a disjoint destination.
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
    // aligned read and write. Both sources are initialized and may overlap;
    // dst is disjoint and only written. All modified GPRs, CTR, XER, and CR0
    // are declared; pointer advances stop at the span ends.
    unsafe {
        asm!(
            "addic {carry}, {rem}, 0",                   // Clear XER[CA] bit (CA = 0)

            // Main 4-way unrolled loop using CTR
            "mtctr {chunks}",                            // Load chunk count into CTR register
            ".p2align 4",
            "2:",
            // [Load 4 Limbs from src1 and src2]
            "lwz {src1_v0}, 0({src1})",                  // Load src1[0]
            "lwz {src1_v1}, 4({src1})",                  // Load src1[1]
            "lwz {src1_v2}, 8({src1})",                  // Load src1[2]
            "lwz {src1_v3}, 12({src1})",                 // Load src1[3]
            "lwz {src2_v0}, 0({src2})",                  // Load src2[0]
            "lwz {src2_v1}, 4({src2})",                  // Load src2[1]
            "lwz {src2_v2}, 8({src2})",                  // Load src2[2]
            "lwz {src2_v3}, 12({src2})",                 // Load src2[3]

            // [Add with Carry]
            "adde {t0}, {src1_v0}, {src2_v0}",           // t0 = src1[0] + src2[0] + CA
            "adde {t1}, {src1_v1}, {src2_v1}",           // t1 = src1[1] + src2[1] + CA
            "adde {t2}, {src1_v2}, {src2_v2}",           // t2 = src1[2] + src2[2] + CA
            "adde {t3}, {src1_v3}, {src2_v3}",           // t3 = src1[3] + src2[3] + CA

            // [Store 4 Limbs to dst]
            "stw {t0}, 0({dst})",                        // Store dst[0]
            "stw {t1}, 4({dst})",                        // Store dst[1]
            "stw {t2}, 8({dst})",                        // Store dst[2]
            "stw {t3}, 12({dst})",                       // Store dst[3]

            // Advance pointers by 16 bytes and loop via CTR
            "addi {src1}, {src1}, 16",                   // Advance src1 pointer
            "addi {src2}, {src2}, 16",                   // Advance src2 pointer
            "addi {dst}, {dst}, 16",                     // Advance dst pointer
            "bdnz 2b",                                   // Decrement CTR and branch if != 0

            // Remainder entry point (0 to 3 limbs)
            "1:",
            "cmpwi {rem}, 0",                            // Check if rem == 0
            "beq 3f",                                    // If rem == 0, exit (3f)
            "mtctr {rem}",                               // Load remainder count into CTR
            ".p2align 4",

            // 1-limb tail loop
            "4:",
            "lwz {src1_v0}, 0({src1})",                  // Load single src1 limb
            "lwz {src2_v0}, 0({src2})",                  // Load single src2 limb
            "adde {t0}, {src1_v0}, {src2_v0}",           // Add with carry
            "stw {t0}, 0({dst})",                        // Store single dst limb
            "addi {src1}, {src1}, 4",                    // Advance src1
            "addi {src2}, {src2}, 4",                    // Advance src2
            "addi {dst}, {dst}, 4",                      // Advance dst
            "bdnz 4b",                                   // Decrement CTR and branch if != 0

            // Exit: capture final carry bit from XER[CA]
            "3:",
            "li {carry}, 0",                             // carry = 0
            "addze {carry}, {carry}",                    // carry = 0 + CA (0 or 1)

            carry = out(reg) carry,
            dst = inout(reg_nonzero) dst => _,
            src1 = inout(reg_nonzero) src1 => _,
            src2 = inout(reg_nonzero) src2 => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1_v0 = out(reg) _, src1_v1 = out(reg) _, src1_v2 = out(reg) _, src1_v3 = out(reg) _,
            src2_v0 = out(reg) _, src2_v1 = out(reg) _, src2_v2 = out(reg) _, src2_v3 = out(reg) _,
            t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _, t3 = out(reg) _,
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
                    "lwz {a0}, 0({src1})",               // Load src1[0]
                    "lwz {a1}, 4({src1})",               // Load src1[1]
                    "lwz {b0}, 0({src2})",               // Load src2[0]
                    "lwz {b1}, 4({src2})",               // Load src2[1]
                    "addc {a0}, {a0}, {b0}",             // a0 = src1[0] + src2[0], set XER[CA]
                    "adde {a1}, {a1}, {b1}",             // a1 = src1[1] + src2[1] + CA
                    "stw {a0}, 0({dst})",                // Store dst[0]
                    "stw {a1}, 4({dst})",                // Store dst[1]
                    "addze {carry}, {zero}",             // carry = 0 + CA (0 or 1)
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
                    "lwz {a0}, 0({src1})",               // Load src1[0]
                    "lwz {a1}, 4({src1})",               // Load src1[1]
                    "lwz {a2}, 8({src1})",               // Load src1[2]
                    "lwz {b0}, 0({src2})",               // Load src2[0]
                    "lwz {b1}, 4({src2})",               // Load src2[1]
                    "lwz {b2}, 8({src2})",               // Load src2[2]
                    "addc {a0}, {a0}, {b0}",             // a0 = src1[0] + src2[0], set XER[CA]
                    "adde {a1}, {a1}, {b1}",             // a1 = src1[1] + src2[1] + CA
                    "adde {a2}, {a2}, {b2}",             // a2 = src1[2] + src2[2] + CA
                    "stw {a0}, 0({dst})",                // Store dst[0]
                    "stw {a1}, 4({dst})",                // Store dst[1]
                    "stw {a2}, 8({dst})",                // Store dst[2]
                    "addze {carry}, {zero}",             // carry = 0 + CA
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
                    "lwz {a0}, 0({src1})",               // Load src1[0]
                    "lwz {a1}, 4({src1})",               // Load src1[1]
                    "lwz {a2}, 8({src1})",               // Load src1[2]
                    "lwz {a3}, 12({src1})",              // Load src1[3]
                    "lwz {b0}, 0({src2})",               // Load src2[0]
                    "lwz {b1}, 4({src2})",               // Load src2[1]
                    "lwz {b2}, 8({src2})",               // Load src2[2]
                    "lwz {b3}, 12({src2})",              // Load src2[3]
                    "addc {a0}, {a0}, {b0}",             // a0 = src1[0] + src2[0], set XER[CA]
                    "adde {a1}, {a1}, {b1}",             // a1 = src1[1] + src2[1] + CA
                    "adde {a2}, {a2}, {b2}",             // a2 = src1[2] + src2[2] + CA
                    "adde {a3}, {a3}, {b3}",             // a3 = src1[3] + src2[3] + CA
                    "stw {a0}, 0({dst})",                // Store dst[0]
                    "stw {a1}, 4({dst})",                // Store dst[1]
                    "stw {a2}, 8({dst})",                // Store dst[2]
                    "stw {a3}, 12({dst})",               // Store dst[3]
                    "addze {carry}, {zero}",             // carry = 0 + CA
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
