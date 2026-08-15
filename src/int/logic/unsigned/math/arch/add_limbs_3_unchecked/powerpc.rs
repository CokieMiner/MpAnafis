//! PowerPC 32-bit 3-way addition kernels (inline assembly).
//!
//! Evaluates `dst = src1 + src2` using 4-way unrolled loops with hardware carry `XER[CA]`
//! via `adde` (add extended) and CTR hardware branch looping (`bdnz`).

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Compute `dst[i] = src1[i] + src2[i] + carry` for `len` limbs, returning
/// the final carry.
///
/// # Microarchitectural Strategy
///
/// `PowerPC32` maintains the carry in `XER[CA]` across `adde` instructions.
/// The 4-way unrolled loop loads 4 words each from `src1` and `src2`, adds them with carry,
/// and uses hardware CTR register (`bdnz`) to eliminate branch latency.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for reads and writes of `len` 32-bit limbs.
/// - `dst` must not alias `src1` or `src2`.
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
    // SAFETY: The caller guarantees both pointers cover `len` elements.
    if len == 0 {
        return 0;
    }
    if len == 1 {
        // SAFETY: The caller guarantees all pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if len <= 4 {
        // SAFETY: Caller guarantees `dst`, `src1`, `src2` valid for `len in 2..=4`.
        return unsafe { add_small_3_unchecked(dst, src1, src2, len) };
    }
    let mut carry: Limb;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst`, `src1`, `src2` are valid for `len` 32-bit `Limb` elements.
    // 2. Memory spans are non-overlapping.
    // 3. Pointer offsets remain within allocated bounds.
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

/// Straight-line `dst[i] = src1[i] + src2[i] + carry` chain for `len` in
/// `2..=4`.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span.
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
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 2 limbs.
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
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 3 limbs.
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
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 4 limbs.
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
        // SAFETY: Caller guarantees `len in 2..=4`.
        _ => unsafe { unreachable_unchecked() },
    }
}
