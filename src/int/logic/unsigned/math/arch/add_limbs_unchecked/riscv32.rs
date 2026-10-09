//! RISC-V 32-bit addition kernels (inline assembly).
//!
//! RISC-V has no carry flag, so carry is tracked manually with `sltu`
//! (set-less-than unsigned) and `or`.
//!
//! Four-limb blocks precede a scalar tail. With a binary incoming carry,
//! the two addition overflow flags cannot both be set; their OR is binary.

use core::{arch::asm, hint::unreachable_unchecked};

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
    reason = "Keep the carry recurrence in the selected arithmetic caller"
)]
#[inline(always)]
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
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
    // SAFETY: len > 4 proves chunks > 0; 4 * chunks + rem == len bounds
    // every aligned, initialized access. Each input pair is read before its
    // result is written, permitting exact aliasing.
    unsafe {
        asm!(
            // -- 4-way unrolled loop ------------------------------------
            ".p2align 4",                          // align loop header for fetch efficiency
            "1:",
            "lw {t0}, 0({src})",           // t0 = src[0]
            "lw {t1}, 0({dst})",           // t1 = dst[0]
            "add {t1}, {t1}, {t0}",        // t1 = dst + src (may wrap)
            "sltu {c0}, {t1}, {t0}",       // c0 = overflow from dst+src (t1 < t0)
            "add {t1}, {t1}, {carry}",     // t1 += previous carry
            "sltu {c1}, {t1}, {carry}",    // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",      // combined carry for next limb
            "sw {t1}, 0({dst})",           // store result

            "lw {t0}, 4({src})",           // t0 = src[1]
            "lw {t1}, 4({dst})",           // t1 = dst[1]
            "add {t1}, {t1}, {t0}",        // t1 = dst + src (may wrap)
            "sltu {c0}, {t1}, {t0}",       // c0 = overflow from dst+src
            "add {t1}, {t1}, {carry}",     // t1 += previous carry
            "sltu {c1}, {t1}, {carry}",    // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",      // combined carry
            "sw {t1}, 4({dst})",           // store result

            "lw {t0}, 8({src})",           // t0 = src[2]
            "lw {t1}, 8({dst})",           // t1 = dst[2]
            "add {t1}, {t1}, {t0}",        // t1 = dst + src (may wrap)
            "sltu {c0}, {t1}, {t0}",       // c0 = overflow from dst+src
            "add {t1}, {t1}, {carry}",     // t1 += previous carry
            "sltu {c1}, {t1}, {carry}",    // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",      // combined carry
            "sw {t1}, 8({dst})",           // store result

            "lw {t0}, 12({src})",          // t0 = src[3]
            "lw {t1}, 12({dst})",          // t1 = dst[3]
            "add {t1}, {t1}, {t0}",        // t1 = dst + src (may wrap)
            "sltu {c0}, {t1}, {t0}",       // c0 = overflow from dst+src
            "add {t1}, {t1}, {carry}",     // t1 += previous carry
            "sltu {c1}, {t1}, {carry}",    // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",      // combined carry
            "sw {t1}, 12({dst})",          // store result

            "addi {src}, {src}, 16",       // advance src by 16 bytes (4 x u32)
            "addi {dst}, {dst}, 16",       // advance dst by 16 bytes
            "addi {chunks}, {chunks}, -1", // decrement chunk counter
            "bnez {chunks}, 1b",           // loop back if chunks != 0

            // -- Tail: single-limb remainder loop -----------------------
            "2:",
            "beqz {rem}, 4f",              // skip tail if rem == 0
            ".p2align 4",                          // align loop header for fetch efficiency
            "3:",
            "lw {t0}, 0({src})",           // t0 = src[i]
            "lw {t1}, 0({dst})",           // t1 = dst[i]
            "add {t1}, {t1}, {t0}",        // t1 = dst + src (may wrap)
            "sltu {c0}, {t1}, {t0}",       // c0 = overflow from dst+src (t1 < t0)
            "add {t1}, {t1}, {carry}",     // t1 += previous carry
            "sltu {c1}, {t1}, {carry}",    // c1 = overflow from adding carry
            "or {carry}, {c0}, {c1}",      // combined carry for next limb
            "sw {t1}, 0({dst})",           // store result
            "addi {src}, {src}, 4",        // advance src by 4 bytes
            "addi {dst}, {dst}, 4",        // advance dst by 4 bytes
            "addi {rem}, {rem}, -1",       // decrement remainder counter
            "bnez {rem}, 3b",              // loop back if rem != 0
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
/// The pointer contract of [`add_limbs_unchecked`] applies and `2 <= len <= 4`.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "The fixed-size carry chains must remain visibly unrolled and inline into the public hot kernel"
)]
#[inline(always)]
unsafe fn add_small_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    match len {
        2 => {
            let mut carry: Limb;
            // SAFETY: len == 2 bounds the aligned, initialized spans.
            // Each input pair is read before its store, permitting exact aliasing.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "lw {t0}, 0({src})",
                    "lw {t1}, 0({dst})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "sw {t1}, 0({dst})",
                    // Limb 1
                    "lw {t0}, 4({src})",
                    "lw {t1}, 4({dst})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 4({dst})",
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
            // SAFETY: len == 3 bounds the aligned, initialized spans.
            // Each input pair is read before its store, permitting exact aliasing.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "lw {t0}, 0({src})",
                    "lw {t1}, 0({dst})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "sw {t1}, 0({dst})",
                    // Limb 1
                    "lw {t0}, 4({src})",
                    "lw {t1}, 4({dst})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 4({dst})",
                    // Limb 2
                    "lw {t0}, 8({src})",
                    "lw {t1}, 8({dst})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 8({dst})",
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
            // SAFETY: len == 4 bounds the aligned, initialized spans.
            // Each input pair is read before its store, permitting exact aliasing.
            unsafe {
                asm!(
                    // Limb 0 (carry-in = 0)
                    "lw {t0}, 0({src})",
                    "lw {t1}, 0({dst})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {carry}, {t1}, {t0}",
                    "sw {t1}, 0({dst})",
                    // Limb 1
                    "lw {t0}, 4({src})",
                    "lw {t1}, 4({dst})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 4({dst})",
                    // Limb 2
                    "lw {t0}, 8({src})",
                    "lw {t1}, 8({dst})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 8({dst})",
                    // Limb 3
                    "lw {t0}, 12({src})",
                    "lw {t1}, 12({dst})",
                    "add {t1}, {t1}, {t0}",
                    "sltu {c0}, {t1}, {t0}",
                    "add {t1}, {t1}, {carry}",
                    "sltu {c1}, {t1}, {carry}",
                    "or {carry}, {c0}, {c1}",
                    "sw {t1}, 12({dst})",
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
        // SAFETY: the caller establishes 2 <= len <= 4, all matched above.
        _ => unsafe { unreachable_unchecked() },
    }
}
