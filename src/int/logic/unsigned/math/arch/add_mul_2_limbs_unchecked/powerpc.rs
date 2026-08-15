//! `PowerPC32` fused dual-row multiply-add kernel.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 32)`)
//! using 32-bit multipliers (`mullw`/`mulhwu`) and hardware CTR loop branching (`bdnz`).

use core::arch::asm;

use super::Limb;

/// Fused dual-row multiply-add kernel for PowerPC 32-bit.
///
/// Computes:
///
/// ```text
///   dst[0..len] += src[0..len] * s0 + c0
///   dst[1..len+1] += src[0..len] * s1 + c1
/// ```
///
/// # Microarchitectural Strategy
///
/// PowerPC 32-bit hoists all four 32×32→64-bit multipliers (`mullw`/`mulhwu`) across both rows,
/// executes sequential carry additions with `addc`/`addze`, and carries forward `dst[j+1]` in register
/// `{d_cur}` to avoid store-to-load forwarding penalties across loop iterations. Hardware `CTR`
/// loop control via `bdnz` provides 0-cycle branch execution.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len + 1` initialized 32-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 32-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical inner loop for 2-row multi-precision Karatsuba and basecase multiplication"
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

    // SAFETY:
    // 1. `dst` is valid for reads and writes of `len + 1` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `4`) remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cmpwi {len}, 0",                            // Compare length with 0
            "beq 2f",                                    // If len == 0, skip to end (2f)
            "lwz {d_cur}, 0({dst})",                     // Prime pipeline: load dst[0]
            "mtctr {len}",                               // Load loop count into hardware CTR register

            ".p2align 4",
            // Main dual-row accumulation loop
            "1:",
            "lwz {s}, 0({src})",                         // Load src[j]
            "lwz {d_next}, 4({dst})",                    // Pre-load dst[j+1] for row 1 accumulation

            // [Hoisted Superscalar Multipliers: 4 independent 32x32->64 products]
            "mullw {p_lo0}, {s}, {s0}",                  // Low 32 bits of src[j] * s0
            "mulhwu {p_hi0}, {s}, {s0}",                 // High 32 bits of src[j] * s0
            "mullw {p_lo1}, {s}, {s1}",                  // Low 32 bits of src[j] * s1
            "mulhwu {p_hi1}, {s}, {s1}",                 // High 32 bits of src[j] * s1

            // [Row 0 Carry Chain: Finalize dst[j]]
            "addc {t_lo0}, {p_lo0}, {c0}",               // t_lo0 = p_lo0 + c0, set CA bit in XER
            "addze {p_hi0}, {p_hi0}",                    // p_hi0 += CA bit
            "addc {d_cur}, {t_lo0}, {d_cur}",            // d_cur += t_lo0, set CA bit
            "addze {c0}, {p_hi0}",                       // c0 = p_hi0 + CA bit (row 0 carry)
            "stw {d_cur}, 0({dst})",                     // Store finalized dst[j]

            // [Row 1 Carry Chain: Compute dst[j+1] and carry-forward in d_cur]
            "addc {t_lo1}, {p_lo1}, {c1}",               // t_lo1 = p_lo1 + c1, set CA bit
            "addze {p_hi1}, {p_hi1}",                    // p_hi1 += CA bit
            "addc {d_cur}, {t_lo1}, {d_next}",           // d_cur = d_next + t_lo1, set CA bit
            "addze {c1}, {p_hi1}",                       // c1 = p_hi1 + CA bit (row 1 carry)

            // Advance pointers and loop via CTR
            "addi {src}, {src}, 4",                      // Advance src pointer by 4 bytes
            "addi {dst}, {dst}, 4",                      // Advance dst pointer by 4 bytes
            "bdnz 1b",                                   // Decrement CTR and branch if CTR != 0

            // [Final Store: Write high accumulated limb dst[len]]
            "stw {d_cur}, 0({dst})",                     // Flush carry-forwarded dst[len]

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
