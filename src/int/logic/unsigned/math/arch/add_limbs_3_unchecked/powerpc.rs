//! PowerPC 32-bit 3-way addition kernels (inline assembly).
//!
//! Uses the XER[CA] carry bit via `adde` (add extended) instructions.
//! The loop is **4‑way unrolled** (`len >> 2`) with the CTR register
//! for counting (`bdnz` branch‑decrement‑non‑zero).

use core::{arch::asm, hint::unreachable_unchecked};

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

    // SAFETY: Caller guarantees pointers are valid for `len` elements.
    unsafe {
        asm!(
            "addic {carry}, {rem}, 0",              // XER[CA] = 0 (add immediate carrying)

            // ── 4‑way unrolled loop (using CTR) ────────────────────────
            "mtctr {chunks}",
            ".p2align 4",                          // align loop header for fetch efficiency
            "2:",
            "lwz {src1_v0}, 0({src1})",
            "lwz {src1_v1}, 4({src1})",
            "lwz {src1_v2}, 8({src1})",
            "lwz {src1_v3}, 12({src1})",
            "lwz {src2_v0}, 0({src2})",
            "lwz {src2_v1}, 4({src2})",
            "lwz {src2_v2}, 8({src2})",
            "lwz {src2_v3}, 12({src2})",
            "adde {t0}, {src1_v0}, {src2_v0}",  // t0 = src1[0] + src2[0] + XER[CA]
            "adde {t1}, {src1_v1}, {src2_v1}",
            "adde {t2}, {src1_v2}, {src2_v2}",
            "adde {t3}, {src1_v3}, {src2_v3}",
            "stw {t0}, 0({dst})",
            "stw {t1}, 4({dst})",
            "stw {t2}, 8({dst})",
            "stw {t3}, 12({dst})",
            "addi {src1}, {src1}, 16",
            "addi {src2}, {src2}, 16",
            "addi {dst}, {dst}, 16",
            "bdnz 2b",

            // ── Tail: single‑limb remainder loop ───────────────────────
            "1:",
            "cmpwi {rem}, 0",
            "beq 3f",
            "mtctr {rem}",
            ".p2align 4",                          // align loop header for fetch efficiency
            "4:",
            "lwz {src1_v0}, 0({src1})",
            "lwz {src2_v0}, 0({src2})",
            "adde {t0}, {src1_v0}, {src2_v0}",
            "stw {t0}, 0({dst})",
            "addi {src1}, {src1}, 4",
            "addi {src2}, {src2}, 4",
            "addi {dst}, {dst}, 4",
            "bdnz 4b",
            "3:",
            "li {carry}, 0",
            "addze {carry}, {carry}",           // carry = XER[CA]

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
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 2 limbs.
            unsafe {
                asm!(
                    "lwz {a0}, 0({src1})",
                    "lwz {a1}, 4({src1})",
                    "lwz {b0}, 0({src2})",
                    "lwz {b1}, 4({src2})",
                    "addc {a0}, {a0}, {b0}",
                    "adde {a1}, {a1}, {b1}",
                    "stw {a0}, 0({dst})",
                    "stw {a1}, 4({dst})",
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
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 3 limbs.
            unsafe {
                asm!(
                    "lwz {a0}, 0({src1})",
                    "lwz {a1}, 4({src1})",
                    "lwz {a2}, 8({src1})",
                    "lwz {b0}, 0({src2})",
                    "lwz {b1}, 4({src2})",
                    "lwz {b2}, 8({src2})",
                    "addc {a0}, {a0}, {b0}",
                    "adde {a1}, {a1}, {b1}",
                    "adde {a2}, {a2}, {b2}",
                    "stw {a0}, 0({dst})",
                    "stw {a1}, 4({dst})",
                    "stw {a2}, 8({dst})",
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
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 4 limbs.
            unsafe {
                asm!(
                    "lwz {a0}, 0({src1})",
                    "lwz {a1}, 4({src1})",
                    "lwz {a2}, 8({src1})",
                    "lwz {a3}, 12({src1})",
                    "lwz {b0}, 0({src2})",
                    "lwz {b1}, 4({src2})",
                    "lwz {b2}, 8({src2})",
                    "lwz {b3}, 12({src2})",
                    "addc {a0}, {a0}, {b0}",
                    "adde {a1}, {a1}, {b1}",
                    "adde {a2}, {a2}, {b2}",
                    "adde {a3}, {a3}, {b3}",
                    "stw {a0}, 0({dst})",
                    "stw {a1}, 4({dst})",
                    "stw {a2}, 8({dst})",
                    "stw {a3}, 12({dst})",
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
        // SAFETY: Caller guarantees `len in 2..=4`.
        _ => unsafe { unreachable_unchecked() },
    }
}
