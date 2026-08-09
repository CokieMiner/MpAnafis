//! `LoongArch64` addition kernels (inline assembly).
//!
//! `LoongArch64` uses the same carry-tracking idiom as RISC-V 64:
//! `add.d` for addition and `sltu` (set-less-than unsigned) to detect
//! carry (wrap-around).  There is no carry flag in the ISA.
//!
//! ## Per-limb carry detection
//!
//! ```text
//!   add.d   tmp, dst, src          // tmp = dst + src  (may wrap)
//!   sltu    c0, tmp, src           // c0 = 1 if tmp < src  (overflow from dst+src)
//!   add.d   result, tmp, carry     // result = tmp + carry
//!   sltu    c1, result, carry      // c1 = 1 if result < carry  (overflow from +carry)
//!   or      carry, c0, c1          // combined carry for next limb
//! ```
//!
//! The overflow-from-first-add (`sltu`) must be computed BEFORE the
//! second addition, because the second addition can change the value
//! in a way that defeats transitive detection (e.g. both additions
//! overflow but the net result wraps back above the operand).
//!
//! ## Loop structure
//!
//! The loop is **4-way unrolled** (`len >> 2`) for maximum throughput.
//! Four limbs are loaded, computed, and stored per iteration.
//! Any remaining limbs (len & 3) are handled by a tail loop.

use core::{arch::asm, hint::unreachable_unchecked};

use super::Limb;

/// Compute `dst[i] = src1[i] + src2[i] + carry` for `len` limbs,
/// returning the final carry.
///
/// # Safety
///
/// `dst`, `src1`, and `src2` must each be valid for `len` elements.
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
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: Assembly block accesses `len` elements from `dst`, `src1`, and `src2`, which caller guarantees are valid.
    unsafe {
        asm!(
            "beqz {chunks}, 2f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            // Limb 0: t2 = src1[0] + src2[0] + carry
            "ld.d {t0}, {src1}, 0",         // t0 = src1[0]
            "ld.d {t1}, {src2}, 0",         // t1 = src2[0]
            "add.d {t2}, {t0}, {t1}",        // t2 = src1 + src2 (may wrap)
            "sltu {c0}, {t2}, {t0}",         // c0 = overflow from src1+src2 (t2 < t0 means wrap)
            "add.d {t2}, {t2}, {carry}",     // t2 += previous carry
            "sltu {c1}, {t2}, {carry}",      // c1 = overflow from adding carry (t2 < carry)
            "or {carry}, {c0}, {c1}",        // combined carry
            "st.d {t2}, {dst}, 0",

            // Limb 1
            "ld.d {t0}, {src1}, 8",
            "ld.d {t1}, {src2}, 8",
            "add.d {t2}, {t0}, {t1}",        // t2 = src1 + src2 (may wrap)
            "sltu {c0}, {t2}, {t0}",         // c0 = overflow from src1+src2
            "add.d {t2}, {t2}, {carry}",     // t2 += previous carry
            "sltu {c1}, {t2}, {carry}",      // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",        // combined carry
            "st.d {t2}, {dst}, 8",

            // Limb 2
            "ld.d {t0}, {src1}, 16",
            "ld.d {t1}, {src2}, 16",
            "add.d {t2}, {t0}, {t1}",        // t2 = src1 + src2 (may wrap)
            "sltu {c0}, {t2}, {t0}",         // c0 = overflow from src1+src2
            "add.d {t2}, {t2}, {carry}",     // t2 += previous carry
            "sltu {c1}, {t2}, {carry}",      // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",        // combined carry
            "st.d {t2}, {dst}, 16",

            // Limb 3
            "ld.d {t0}, {src1}, 24",
            "ld.d {t1}, {src2}, 24",
            "add.d {t2}, {t0}, {t1}",        // t2 = src1 + src2 (may wrap)
            "sltu {c0}, {t2}, {t0}",         // c0 = overflow from src1+src2
            "add.d {t2}, {t2}, {carry}",     // t2 += previous carry
            "sltu {c1}, {t2}, {carry}",      // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",        // combined carry
            "st.d {t2}, {dst}, 24",

            "addi.d {src1}, {src1}, 32",
            "addi.d {src2}, {src2}, 32",
            "addi.d {dst}, {dst}, 32",
            "addi.d {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            "2:",
            "beqz {rem}, 4f",
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "ld.d {t0}, {src1}, 0",
            "ld.d {t1}, {src2}, 0",
            "add.d {t2}, {t0}, {t1}",        // t2 = src1 + src2 (may wrap)
            "sltu {c0}, {t2}, {t0}",         // c0 = overflow from src1+src2
            "add.d {t2}, {t2}, {carry}",     // t2 += previous carry
            "sltu {c1}, {t2}, {carry}",      // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",        // combined carry
            "st.d {t2}, {dst}, 0",
            "addi.d {src1}, {src1}, 8",
            "addi.d {src2}, {src2}, 8",
            "addi.d {dst}, {dst}, 8",
            "addi.d {rem}, {rem}, -1",
            "bnez {rem}, 3b",
            "4:",

            carry = inout(reg) carry,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            dst = inout(reg) dst => _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            c0 = out(reg) _,
            c1 = out(reg) _,
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
    clippy::too_many_lines,
    reason = "The fixed-size carry chains must remain visibly unrolled and inline into the public hot kernel"
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
                    // Limb 0 (carry-in = 0)
                    "ld.d {t0}, {src1}, 0",
                    "ld.d {t1}, {src2}, 0",
                    "add.d {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "st.d {t1}, {dst}, 0",
                    // Limb 1
                    "ld.d {t0}, {src1}, 8",
                    "ld.d {t1}, {src2}, 8",
                    "add.d {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.d {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.d {t1}, {dst}, 8",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
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
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 3 limbs.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld.d {t0}, {src1}, 0",
                    "ld.d {t1}, {src2}, 0",
                    "add.d {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "st.d {t1}, {dst}, 0",
                    // Limb 1
                    "ld.d {t0}, {src1}, 8",
                    "ld.d {t1}, {src2}, 8",
                    "add.d {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.d {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.d {t1}, {dst}, 8",
                    // Limb 2
                    "ld.d {t0}, {src1}, 16",
                    "ld.d {t1}, {src2}, 16",
                    "add.d {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.d {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.d {t1}, {dst}, 16",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
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
            // SAFETY: Caller guarantees `dst`, `src1`, `src2` are valid for 4 limbs.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld.d {t0}, {src1}, 0",
                    "ld.d {t1}, {src2}, 0",
                    "add.d {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "st.d {t1}, {dst}, 0",
                    // Limb 1
                    "ld.d {t0}, {src1}, 8",
                    "ld.d {t1}, {src2}, 8",
                    "add.d {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.d {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.d {t1}, {dst}, 8",
                    // Limb 2
                    "ld.d {t0}, {src1}, 16",
                    "ld.d {t1}, {src2}, 16",
                    "add.d {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.d {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.d {t1}, {dst}, 16",
                    // Limb 3
                    "ld.d {t0}, {src1}, 24",
                    "ld.d {t1}, {src2}, 24",
                    "add.d {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add.d {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "st.d {t1}, {dst}, 24",
                    src1 = in(reg) src1,
                    src2 = in(reg) src2,
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
