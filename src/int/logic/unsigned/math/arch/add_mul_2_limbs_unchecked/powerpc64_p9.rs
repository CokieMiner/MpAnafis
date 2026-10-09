//! `PowerPC64` (POWER9 / POWER10 ISA 3.0+) fused dual-row multiply-add kernel.
//!
//! `maddld`/`maddhdu` compute the low and high halves of each product plus
//! destination limb. The overlapping intermediate limb remains in a register.

use core::arch::asm;

use super::Limb;

/// Accumulates two interleaved scalar-product rows and returns their separate carries.
///
/// Empty spans return `(0, 0)` without accessing pointers.
///
/// # Safety
///
/// For nonzero `len`, `src` must cover `len` aligned, initialized limbs and
/// `dst` must cover `len + 1` aligned, initialized, writable limbs. The spans
/// must be disjoint and remain within live allocations of at most `isize::MAX` bytes.
/// The executing CPU must support the POWER9 multiply-add instructions.
#[expect(
    clippy::inline_always,
    reason = "Keep both hardware row recurrences in the selected multiplication caller"
)]
#[inline(always)]
pub unsafe fn add_mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) -> (Limb, Limb) {
    let mut c0: Limb = 0;
    let mut c1: Limb = 0;

    // SAFETY: zero length skips the initial load and final store. Otherwise CTR
    // runs exactly len iterations over source[j] and destination[j+1], followed
    // by the store to destination[len]. Eight-byte advances stay aligned within
    // the live, disjoint spans. Backend selection establishes POWER9 support.
    unsafe {
        asm!(
            "cmpldi {len}, 0",                           // Compare length with 0
            "beq 2f",                                    // If len == 0, skip to end (2f)
            "ld {d_cur}, 0({dst})",                      // Prime pipeline: load dst[0]
            "mtctr {len}",                               // Load loop counter into hardware CTR register

            ".p2align 4",
            // Main dual-row accumulation loop
            "1:",
            "ld {s}, 0({src})",                          // Load src[j]
            "ld {d_next}, 8({dst})",                     // Pre-load dst[j+1] for row 1 accumulation

            // [Row 0 Fused MAC: (s * s0 + d_cur)]
            "maddld {t0}, {s}, {s0}, {d_cur}",           // t0 = (src[j] * s0 + d_cur).lo
            "maddhdu {hi0}, {s}, {s0}, {d_cur}",         // hi0 = (src[j] * s0 + d_cur).hi
            "addc {d_cur}, {t0}, {c0}",                  // d_cur = t0 + c0, set CA bit in XER
            "addze {c0}, {hi0}",                         // c0 = hi0 + CA bit (row 0 carry)
            "std {d_cur}, 0({dst})",                     // Store finalized dst[j]

            // [Row 1 Fused MAC: (s * s1 + d_next)]
            "maddld {t1}, {s}, {s1}, {d_next}",          // t1 = (src[j] * s1 + d_next).lo
            "maddhdu {hi1}, {s}, {s1}, {d_next}",        // hi1 = (src[j] * s1 + d_next).hi
            "addc {d_cur}, {t1}, {c1}",                  // d_cur = t1 + c1, set CA bit in XER
            "addze {c1}, {hi1}",                         // c1 = hi1 + CA bit (row 1 carry)

            // Advance pointers and loop via CTR
            "addi {src}, {src}, 8",                      // Advance src pointer by 8 bytes
            "addi {dst}, {dst}, 8",                      // Advance dst pointer by 8 bytes
            "bdnz 1b",                                   // Decrement CTR and branch if CTR != 0

            // [Final Store: Write high accumulated limb dst[len]]
            "std {d_cur}, 0({dst})",                     // Flush carry-forwarded dst[len]

            // Completion
            "2:",

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg_nonzero) src => _,
            dst = inout(reg_nonzero) dst => _,
            len = in(reg) len,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            s = out(reg) _,
            d_cur = out(reg) _,
            d_next = out(reg) _,
            t0 = out(reg) _,
            hi0 = out(reg) _,
            t1 = out(reg) _,
            hi1 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    (c0, c1)
}
