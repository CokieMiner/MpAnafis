//! `AArch64` (ARMv8-A / ARMv9-A) fused dual-row multiply-add kernel.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 64)`)
//! in a single interleaved pass, eliminating store-to-load forwarding stalls.

use core::arch::asm;

use super::Limb;

/// Fused dual-row multiply-add kernel for `AArch64`.
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
/// Multi-precision basecase multiplication processes two multiplier scalars (`s0` and `s1`)
/// simultaneously. The kernel interleaves four 64×64→128-bit multipliers (`mul`/`umulh`) across
/// both rows and carries forward the accumulated `dst[j+1]` in register `{d_cur}`, eliminating
/// store-to-load forwarding (STLF) latencies across loop iterations.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len + 1` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
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

    if len == 0 {
        return (0, 0);
    }

    // SAFETY:
    // 1. `dst` is valid for reads and writes of `len + 1` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer advances remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "ldr {d_cur}, [{dst}]",                      // Prime register pipeline: load dst[0]

            // Main dual-row accumulation loop
            "1:",
            "ldr {src_val}, [{src}], #8",                // Load src[j] and advance src pointer
            "ldr {d_next}, [{dst}, #8]",                 // Pre-load dst[j+1] for row 1 accumulation

            // [Hoisted Superscalar Multipliers: 4 concurrent 64x64->128 products]
            "mul {lo0}, {src_val}, {s0}",                // Low 64 bits of src[j] * s0
            "mul {lo1}, {src_val}, {s1}",                // Low 64 bits of src[j] * s1
            "umulh {hi0}, {src_val}, {s0}",              // High 64 bits of src[j] * s0
            "umulh {hi1}, {src_val}, {s1}",              // High 64 bits of src[j] * s1

            // [Row 0 Carry Chain: Finalize dst[j]]
            "adds {lo0}, {lo0}, {c0}",                   // lo0 += c0, set C flag
            "adc {hi0}, {hi0}, xzr",                     // hi0 += C flag + 0
            "adds {d_cur}, {d_cur}, {lo0}",              // d_cur += lo0, set C flag
            "adc {c0}, {hi0}, xzr",                      // c0 = hi0 + C flag (carry for next row 0 limb)
            "str {d_cur}, [{dst}], #8",                  // Store finalized dst[j] and advance dst pointer

            // [Row 1 Carry Chain: Compute dst[j+1] and carry-forward in d_cur]
            "adds {lo1}, {lo1}, {c1}",                   // lo1 += c1, set C flag
            "adc {hi1}, {hi1}, xzr",                     // hi1 += C flag + 0
            "adds {d_cur}, {d_next}, {lo1}",             // d_cur = d_next + lo1, set C flag
            "adc {c1}, {hi1}, xzr",                      // c1 = hi1 + C flag (carry for next row 1 limb)

            "subs {len}, {len}, #1",                     // Decrement remaining limbs
            "b.ne 1b",                                   // Loop while len != 0

            // [Final Store: Write high accumulated limb dst[len]]
            "str {d_cur}, [{dst}]",                      // Store carry-forwarded dst[len]

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = inout(reg) len => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            src_val = out(reg) _,
            d_cur = out(reg) _,
            d_next = out(reg) _,
            lo0 = out(reg) _,
            hi0 = out(reg) _,
            lo1 = out(reg) _,
            hi1 = out(reg) _,
            options(nostack)
        );
    }
    (c0, c1)
}
