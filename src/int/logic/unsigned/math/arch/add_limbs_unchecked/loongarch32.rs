//! `LoongArch32` addition kernels (inline assembly).
//!
//! `LoongArch32` uses the same carry-tracking idiom as RISC-V 32:
//! `add.w` for addition and `sltu` (set-less-than unsigned) to detect
//! carry (wrap-around).  There is no carry flag in the ISA.
//!
//! The loop is **4-way unrolled** (`len >> 2`) for maximum throughput.

use core::{arch::asm, hint::unreachable_unchecked};

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
    // SAFETY: The caller guarantees both pointers cover `len` elements.
    if len == 0 {
        return 0;
    }
    if len == 1 {
        // SAFETY: The caller guarantees both pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*dst).overflowing_add(*src) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    if len <= 4 {
        // SAFETY: Caller guarantees `dst` and `src` are valid for `len in 2..=4`.
        return unsafe { add_small_unchecked(dst, src, len) };
    }
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`, which caller guarantees are valid.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",
            // ── 4-way unrolled loop ────────────────────────────────────
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            // Limb 0
            "ld.w {t0}, {src}, 0",
            "ld.w {t1}, {dst}, 0",
            "add.w {t1}, {t1}, {t0}",
            "sltu {c0}, {t1}, {t0}",
            "add.w {t1}, {t1}, {carry}",
            "sltu {c1}, {t1}, {carry}",
            "or {carry}, {c0}, {c1}",
            "st.w {t1}, {dst}, 0",

            // Limb 1
            "ld.w {t0}, {src}, 4",
            "ld.w {t1}, {dst}, 4",
            "add.w {t1}, {t1}, {t0}",
            "sltu {c0}, {t1}, {t0}",
            "add.w {t1}, {t1}, {carry}",
            "sltu {c1}, {t1}, {carry}",
            "or {carry}, {c0}, {c1}",
            "st.w {t1}, {dst}, 4",

            // Limb 2
            "ld.w {t0}, {src}, 8",
            "ld.w {t1}, {dst}, 8",
            "add.w {t1}, {t1}, {t0}",
            "sltu {c0}, {t1}, {t0}",
            "add.w {t1}, {t1}, {carry}",
            "sltu {c1}, {t1}, {carry}",
            "or {carry}, {c0}, {c1}",
            "st.w {t1}, {dst}, 8",

            // Limb 3
            "ld.w {t0}, {src}, 12",
            "ld.w {t1}, {dst}, 12",
            "add.w {t1}, {t1}, {t0}",
            "sltu {c0}, {t1}, {t0}",
            "add.w {t1}, {t1}, {carry}",
            "sltu {c1}, {t1}, {carry}",
            "or {carry}, {c0}, {c1}",
            "st.w {t1}, {dst}, 12",

            "addi.w {src}, {src}, 16",
            "addi.w {dst}, {dst}, 16",
            "addi.w {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            // ── Tail: single-limb remainder loop ───────────────────────
            "2:",
            "beqz {rem}, 4f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "ld.w {t0}, {src}, 0",
            "ld.w {t1}, {dst}, 0",
            "add.w {t1}, {t1}, {t0}",
            "sltu {c0}, {t1}, {t0}",
            "add.w {t1}, {t1}, {carry}",
            "sltu {c1}, {t1}, {carry}",
            "or {carry}, {c0}, {c1}",
            "st.w {t1}, {dst}, 0",

            "addi.w {src}, {src}, 4",
            "addi.w {dst}, {dst}, 4",
            "addi.w {rem}, {rem}, -1",
            "bnez {rem}, 3b",
            "4:",

            carry = inout(reg) carry,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            c0 = out(reg) _,
            c1 = out(reg) _,
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
    clippy::too_many_lines,
    reason = "The fixed-size carry chains must remain visibly unrolled and inline into the public hot kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst` and `src` are valid for 2 limbs.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld.w {t0}, {src}, 0",
                    "ld.w {t1}, {dst}, 0",
                    "add.w {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "st.w {t1}, {dst}, 0",
                    // Limb 1
                    "ld.w {t0}, {src}, 4",
                    "ld.w {t1}, {dst}, 4",
                    "add.w {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.w {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.w {t1}, {dst}, 4",
                    src = in(reg) src,
                    dst = in(reg) dst,
                    t0 = out(reg) _, t1 = out(reg) _,
                    c0 = out(reg) _, c1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        3 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst` and `src` are valid for 3 limbs.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld.w {t0}, {src}, 0",
                    "ld.w {t1}, {dst}, 0",
                    "add.w {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "st.w {t1}, {dst}, 0",
                    // Limb 1
                    "ld.w {t0}, {src}, 4",
                    "ld.w {t1}, {dst}, 4",
                    "add.w {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.w {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.w {t1}, {dst}, 4",
                    // Limb 2
                    "ld.w {t0}, {src}, 8",
                    "ld.w {t1}, {dst}, 8",
                    "add.w {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.w {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.w {t1}, {dst}, 8",
                    src = in(reg) src,
                    dst = in(reg) dst,
                    t0 = out(reg) _, t1 = out(reg) _,
                    c0 = out(reg) _, c1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        4 => {
            let mut carry: Limb;
            // SAFETY: Caller guarantees `dst` and `src` are valid for 4 limbs.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld.w {t0}, {src}, 0",
                    "ld.w {t1}, {dst}, 0",
                    "add.w {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "st.w {t1}, {dst}, 0",
                    // Limb 1
                    "ld.w {t0}, {src}, 4",
                    "ld.w {t1}, {dst}, 4",
                    "add.w {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.w {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.w {t1}, {dst}, 4",
                    // Limb 2
                    "ld.w {t0}, {src}, 8",
                    "ld.w {t1}, {dst}, 8",
                    "add.w {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.w {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.w {t1}, {dst}, 8",
                    // Limb 3
                    "ld.w {t0}, {src}, 12",
                    "ld.w {t1}, {dst}, 12",
                    "add.w {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.w {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.w {t1}, {dst}, 12",
                    src = in(reg) src,
                    dst = in(reg) dst,
                    t0 = out(reg) _, t1 = out(reg) _,
                    c0 = out(reg) _, c1 = out(reg) _,
                    carry = out(reg) carry,
                    options(nostack)
                );
            }
            carry
        }
        // SAFETY: Caller guarantees `len in 2..=4`.
        _ => unsafe { unreachable_unchecked() },
    }
}
