//! `LoongArch64` fused dual-row multiply-add kernel.
//!
//! Evaluates two simultaneous multiplication rows (`dst += src * s0 + (src * s1 << 64)`)
//! using 64×64→128-bit multipliers (`mul.d`/`mulh.du`) and branchless `sltu` carry capture.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by two scalars `s0` and `s1` simultaneously,
/// accumulating each result into two overlapping rows of `dst`:
///
/// ```text
///   (c0, dst[0..len])   = dst[0..len]   + src[0..len] × s0 + c0_in
///   (c1, dst[1..len+1]) = dst[1..len+1] + src[0..len] × s1 + c1_in
/// ```
///
/// Returns the two final carry-out values `(c0, c1)`.
///
/// # Microarchitectural Strategy
///
/// Evaluates four 64×64→128-bit products per limb across both rows (`s0` and `s1`),
/// captures two independent carry chains branchlessly with `sltu`, and updates memory in place.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len + 1` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in 64-bit LoongArch dual-row multiplication"
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
    // 1. `dst` is valid for reads and writes of `len + 1` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`) remain within allocated bounds.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "beqz {len}, 2f",                            // If len == 0, skip to end (2f)

            // Main dual-row accumulation loop
            "1:",
            "ld.d {s}, {src}, 0",                        // Load src[j]

            // [Hoisted Multipliers for Both Rows]
            "mul.d {p_lo0}, {s}, {s0}",                  // Low 64 bits of src[j] * s0
            "mulh.du {p_hi0}, {s}, {s0}",                // High 64 bits of src[j] * s0
            "mul.d {p_lo1}, {s}, {s1}",                  // Low 64 bits of src[j] * s1
            "mulh.du {p_hi1}, {s}, {s1}",                // High 64 bits of src[j] * s1

            // [Add Row Carries into Low Products]
            "add.d {p_lo0}, {p_lo0}, {c0}",              // p_lo0 += c0
            "sltu {t0}, {p_lo0}, {c0}",                  // t0 = 1 if addition wrapped
            "add.d {p_lo1}, {p_lo1}, {c1}",              // p_lo1 += c1
            "sltu {t1}, {p_lo1}, {c1}",                  // t1 = 1 if addition wrapped

            // [Propagate Wrap Overflows to High Products]
            "add.d {p_hi0}, {p_hi0}, {t0}",              // p_hi0 += t0
            "add.d {p_hi1}, {p_hi1}, {t1}",              // p_hi1 += t1

            // [Load Destination Limbs]
            "ld.d {d0}, {dst}, 0",                       // Load dst[j]
            "ld.d {d1}, {dst}, 8",                       // Load dst[j+1]

            // [Accumulate into Destination Limbs]
            "add.d {d0}, {d0}, {p_lo0}",                 // dst[j] += p_lo0
            "sltu {t0}, {d0}, {p_lo0}",                  // t0 = 1 if dst[j] addition wrapped
            "add.d {d1}, {d1}, {p_lo1}",                 // dst[j+1] += p_lo1
            "sltu {t1}, {d1}, {p_lo1}",                  // t1 = 1 if dst[j+1] addition wrapped

            // [Update Running Carries for Next Limb]
            "add.d {c0}, {p_hi0}, {t0}",                 // c0 = p_hi0 + t0 (row 0 carry)
            "add.d {c1}, {p_hi1}, {t1}",                 // c1 = p_hi1 + t1 (row 1 carry)

            // [Store Updated Destination Limbs]
            "st.d {d0}, {dst}, 0",                       // Store finalized dst[j]
            "st.d {d1}, {dst}, 8",                       // Store intermediate dst[j+1]

            // Advance pointers and loop
            "addi.d {src}, {src}, 8",                    // Advance src pointer by 8 bytes
            "addi.d {dst}, {dst}, 8",                    // Advance dst pointer by 8 bytes
            "addi.d {len}, {len}, -1",                   // Decrement remaining count
            "bnez {len}, 1b",                            // Repeat while len != 0

            // Completion
            "2:",

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = inout(reg) len => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            s = out(reg) _,
            d0 = out(reg) _,
            d1 = out(reg) _,
            p_lo0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi0 = out(reg) _,
            p_hi1 = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            options(nostack)
        );
    }
    (c0, c1)
}
