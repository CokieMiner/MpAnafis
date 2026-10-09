//! `PowerPC64` fused dual-row multiply-add kernel.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 64)`)
//! using 64-bit multipliers (`mulld`/`mulhdu`) and hardware CTR loop branching (`bdnz`).

use core::arch::asm;

use super::Limb;

/// Accumulates two interleaved scalar-product rows and returns their separate carries.
///
/// The intermediate `dst[j+1]` remains in a register for the next row-0 update.
/// Empty spans return `(0, 0)` without accessing pointers.
///
/// # Safety
///
/// For nonzero `len`, `src` must cover `len` aligned, initialized limbs and
/// `dst` must cover `len + 1` aligned, initialized, writable limbs. The spans
/// must be disjoint and remain within live allocations of at most `isize::MAX` bytes.
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
    // the live spans. All writes target the initialized, disjoint destination.
    unsafe {
        asm!(
            "cmpldi {len}, 0",                           // Compare length with 0
            "beq 2f",                                    // If len == 0, skip to end (2f)
            "ld {d_cur}, 0({dst})",                      // Prime pipeline: load dst[0]
            "mtctr {len}",                               // Load loop count into hardware CTR register

            ".p2align 4",
            // Main dual-row accumulation loop
            "1:",
            "ld {s}, 0({src})",                          // Load src[j]
            "ld {d_next}, 8({dst})",                     // Pre-load dst[j+1] for row 1 accumulation

            // Low and high halves of the two scalar products.
            "mulld {p_lo0}, {s}, {s0}",                  // Low 64 bits of src[j] * s0
            "mulhdu {p_hi0}, {s}, {s0}",                 // High 64 bits of src[j] * s0
            "mulld {p_lo1}, {s}, {s1}",                  // Low 64 bits of src[j] * s1
            "mulhdu {p_hi1}, {s}, {s1}",                 // High 64 bits of src[j] * s1

            // [Row 0 Carry Chain: Finalize dst[j]]
            "addc {t_lo0}, {p_lo0}, {c0}",               // t_lo0 = p_lo0 + c0, set CA bit in XER
            "addze {p_hi0}, {p_hi0}",                    // p_hi0 += CA bit
            "addc {d_cur}, {t_lo0}, {d_cur}",            // d_cur += t_lo0, set CA bit
            "addze {c0}, {p_hi0}",                       // c0 = p_hi0 + CA bit (row 0 carry)
            "std {d_cur}, 0({dst})",                     // Store finalized dst[j]

            // [Row 1 Carry Chain: Compute dst[j+1] and carry-forward in d_cur]
            "addc {t_lo1}, {p_lo1}, {c1}",               // t_lo1 = p_lo1 + c1, set CA bit
            "addze {p_hi1}, {p_hi1}",                    // p_hi1 += CA bit
            "addc {d_cur}, {t_lo1}, {d_next}",           // d_cur = d_next + t_lo1, set CA bit
            "addze {c1}, {p_hi1}",                       // c1 = p_hi1 + CA bit (row 1 carry)

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
            p_lo0 = out(reg) _,
            p_hi0 = out(reg) _,
            t_lo0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi1 = out(reg) _,
            t_lo1 = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    (c0, c1)
}
