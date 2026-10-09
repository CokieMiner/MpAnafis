//! RISC-V64 addition into a disjoint destination.
//!
//! With `B = 2^64`, let `s = (a + b) mod B` and `d = (s + c) mod B`,
//! where `c` is zero or one. `sltu` detects `s < a` and `d < c`; their OR is
//! the outgoing carry. These overflows are mutually exclusive: the first
//! implies `s <= B - 2`, while the second requires `s == B - 1` and `c == 1`.
//! Four-limb blocks precede a scalar tail.

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
    reason = "Keep the carry recurrence in the selected arithmetic caller"
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
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;
    // SAFETY: len > 4 gives chunks > 0; 4 * chunks + rem == len bounds every
    // aligned read and write. Both sources are initialized and may overlap;
    // dst is disjoint and only written. All modified GPRs are declared, and
    // pointer advances stop at the span ends.
    unsafe {
        asm!(
            // -- 4-way unrolled loop ------------------------------------
            ".p2align 4",                          // align the loop to 16 bytes
            "1:",
            "ld {t0}, 0({src1})",          // t0 = src1[0]
            "ld {t1}, 0({src2})",          // t1 = src2[0]
            "add {t2}, {t0}, {t1}",        // t2 = src1 + src2 (may wrap)
            "sltu {c0}, {t2}, {t0}",       // c0 = overflow from src1+src2 (t2 < t0)
            "add {t2}, {t2}, {carry}",     // t2 += previous carry
            "sltu {c1}, {t2}, {carry}",    // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",      // combined carry for next limb
            "sd {t2}, 0({dst})",           // store result

            "ld {t0}, 8({src1})",          // t0 = src1[1]
            "ld {t1}, 8({src2})",          // t1 = src2[1]
            "add {t2}, {t0}, {t1}",        // t2 = src1 + src2 (may wrap)
            "sltu {c0}, {t2}, {t0}",       // c0 = overflow from src1+src2
            "add {t2}, {t2}, {carry}",     // t2 += previous carry
            "sltu {c1}, {t2}, {carry}",    // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",      // combined carry
            "sd {t2}, 8({dst})",           // store result

            "ld {t0}, 16({src1})",         // t0 = src1[2]
            "ld {t1}, 16({src2})",         // t1 = src2[2]
            "add {t2}, {t0}, {t1}",        // t2 = src1 + src2 (may wrap)
            "sltu {c0}, {t2}, {t0}",       // c0 = overflow from src1+src2
            "add {t2}, {t2}, {carry}",     // t2 += previous carry
            "sltu {c1}, {t2}, {carry}",    // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",      // combined carry
            "sd {t2}, 16({dst})",          // store result

            "ld {t0}, 24({src1})",         // t0 = src1[3]
            "ld {t1}, 24({src2})",         // t1 = src2[3]
            "add {t2}, {t0}, {t1}",        // t2 = src1 + src2 (may wrap)
            "sltu {c0}, {t2}, {t0}",       // c0 = overflow from src1+src2
            "add {t2}, {t2}, {carry}",     // t2 += previous carry
            "sltu {c1}, {t2}, {carry}",    // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",      // combined carry
            "sd {t2}, 24({dst})",          // store result

            "addi {src1}, {src1}, 32",     // advance src1 by 32 bytes (4 x u64)
            "addi {src2}, {src2}, 32",     // advance src2 by 32 bytes
            "addi {dst}, {dst}, 32",       // advance dst by 32 bytes
            "addi {chunks}, {chunks}, -1", // decrement chunk counter
            "bnez {chunks}, 1b",           // loop back if chunks != 0

            // -- Tail: single-limb remainder loop -----------------------
            "2:",
            "beqz {rem}, 4f",              // skip tail if rem == 0
            ".p2align 4",                          // align the loop to 16 bytes
            "3:",
            "ld {t0}, 0({src1})",          // t0 = src1[i]
            "ld {t1}, 0({src2})",          // t1 = src2[i]
            "add {t2}, {t0}, {t1}",        // t2 = src1 + src2 (may wrap)
            "sltu {c0}, {t2}, {t0}",       // c0 = overflow from src1+src2 (t2 < t0)
            "add {t2}, {t2}, {carry}",     // t2 += previous carry
            "sltu {c1}, {t2}, {carry}",    // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",      // combined carry for next limb
            "sd {t2}, 0({dst})",           // store result
            "addi {src1}, {src1}, 8",      // advance src1 by 8 bytes
            "addi {src2}, {src2}, 8",      // advance src2 by 8 bytes
            "addi {dst}, {dst}, 8",        // advance dst by 8 bytes
            "addi {rem}, {rem}, -1",       // decrement remainder counter
            "bnez {rem}, 3b",              // loop back if rem != 0
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

/// Writes a sum with a straight carry chain for `2 <= len <= 4`.
///
/// # Safety
///
/// `len` must be in `2..=4`. Both aligned sources must contain `len`
/// initialized limbs. The aligned destination must cover `len` writable limbs
/// and be disjoint from both sources; the sources may overlap each other.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "Keep the three fixed-size carry chains in one inlined kernel"
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
                    // Limb 0 (carry-in = 0)
                    "ld {t0}, 0({src1})",
                    "ld {t1}, 0({src2})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "sd {t1}, 0({dst})",
                    // Limb 1
                    "ld {t0}, 8({src1})",
                    "ld {t1}, 8({src2})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sd {t1}, 8({dst})",
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
            // SAFETY: len == 3 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld {t0}, 0({src1})",
                    "ld {t1}, 0({src2})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "sd {t1}, 0({dst})",
                    // Limb 1
                    "ld {t0}, 8({src1})",
                    "ld {t1}, 8({src2})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sd {t1}, 8({dst})",
                    // Limb 2
                    "ld {t0}, 16({src1})",
                    "ld {t1}, 16({src2})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sd {t1}, 16({dst})",
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
            // SAFETY: len == 4 bounds each aligned span; both sources are
            // initialized and the disjoint destination is only written.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "ld {t0}, 0({src1})",
                    "ld {t1}, 0({src2})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "sd {t1}, 0({dst})",
                    // Limb 1
                    "ld {t0}, 8({src1})",
                    "ld {t1}, 8({src2})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sd {t1}, 8({dst})",
                    // Limb 2
                    "ld {t0}, 16({src1})",
                    "ld {t1}, 16({src2})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sd {t1}, 16({dst})",
                    // Limb 3
                    "ld {t0}, 24({src1})",
                    "ld {t1}, 24({src2})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sd {t1}, 24({dst})",
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
        // SAFETY: the caller establishes 2 <= len <= 4, all matched above.
        _ => unsafe { unreachable_unchecked() },
    }
}
