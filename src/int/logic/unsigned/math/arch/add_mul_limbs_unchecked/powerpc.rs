//! PowerPC 32-bit (G3 / G4 / e500) fused multiply-add limb kernel.
//!
//! Uses 32×32→64-bit integer multipliers (`mullw`/`mulhwu`), carry-chain arithmetic
//! (`addc`/`addze`), and hardware CTR zero-overhead loop branches (`bdnz`).

use core::arch::asm;

use super::Limb;

/// Multiply `len` 32-bit limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// Computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// PowerPC 32-bit architectures feature dual execution units for 32-bit integer multiplication.
/// The kernel hoists all 8 `mullw`/`mulhwu` operations across 4 limbs upfront, saturating
/// the superscalar execution pipelines while memory loads complete. Carry bits (`CA` in `XER`)
/// are chained sequentially via `addc` and `addze`. The hardware `CTR` register provides
/// 0-cycle branch execution via `bdnz`.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 32-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 32-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in 32-bit PowerPC multi-precision hot paths"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry: Limb = 0;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 32-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 32-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `4`, `8`, `12`, `16`) remain within `len * 4` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cmpwi {chunks}, 0",          // Compare 4-limb chunks count with 0
            "beq 1f",                     // If chunks == 0, skip to remainder loop (1f)
            "mtctr {chunks}",             // Load loop counter into hardware CTR register

            ".p2align 4",
            // Main 4-way unrolled loop body
            "2:",

            // [Load 4 Source and 4 Destination Limbs (32-bit each)]
            "lwz {src_v0}, 0({src})",     // Load src[0]
            "lwz {src_v1}, 4({src})",     // Load src[1]
            "lwz {src_v2}, 8({src})",     // Load src[2]
            "lwz {src_v3}, 12({src})",    // Load src[3]
            "lwz {dst_v0}, 0({dst})",     // Load dst[0]
            "lwz {dst_v1}, 4({dst})",     // Load dst[1]
            "lwz {dst_v2}, 8({dst})",     // Load dst[2]
            "lwz {dst_v3}, 12({dst})",    // Load dst[3]

            // [Hoisted Superscalar Multipliers: 4 independent 32x32->64 product pairs]
            "mullw {p_lo0}, {src_v0}, {scalar}",          // Low 32 bits of src[0] * scalar
            "mulhwu {p_hi0}, {src_v0}, {scalar}",         // High 32 bits of src[0] * scalar
            "mullw {p_lo1}, {src_v1}, {scalar}",          // Low 32 bits of src[1] * scalar
            "mulhwu {p_hi1}, {src_v1}, {scalar}",         // High 32 bits of src[1] * scalar
            "mullw {p_lo2}, {src_v2}, {scalar}",          // Low 32 bits of src[2] * scalar
            "mulhwu {p_hi2}, {src_v2}, {scalar}",         // High 32 bits of src[2] * scalar
            "mullw {p_lo3}, {src_v3}, {scalar}",          // Low 32 bits of src[3] * scalar
            "mulhwu {p_hi3}, {src_v3}, {scalar}",         // High 32 bits of src[3] * scalar

            // [Limb 0 Carry Chain Accumulation]
            "addc {p_lo0}, {p_lo0}, {carry}",             // p_lo0 += carry, set CA bit in XER
            "addze {p_hi0}, {p_hi0}",                     // p_hi0 += CA bit (propagate carry to high product)
            "addc {dst_v0}, {dst_v0}, {p_lo0}",           // dst_v0 += p_lo0, set CA bit in XER
            "addze {carry}, {p_hi0}",                     // carry = p_hi0 + CA bit
            "stw {dst_v0}, 0({dst})",                     // Store accumulated dst[0]

            // [Limb 1 Carry Chain Accumulation]
            "addc {p_lo1}, {p_lo1}, {carry}",             // p_lo1 += carry, set CA bit
            "addze {p_hi1}, {p_hi1}",                     // p_hi1 += CA bit
            "addc {dst_v1}, {dst_v1}, {p_lo1}",           // dst_v1 += p_lo1, set CA bit
            "addze {carry}, {p_hi1}",                     // carry = p_hi1 + CA bit
            "stw {dst_v1}, 4({dst})",                     // Store accumulated dst[1]

            // [Limb 2 Carry Chain Accumulation]
            "addc {p_lo2}, {p_lo2}, {carry}",             // p_lo2 += carry, set CA bit
            "addze {p_hi2}, {p_hi2}",                     // p_hi2 += CA bit
            "addc {dst_v2}, {dst_v2}, {p_lo2}",           // dst_v2 += p_lo2, set CA bit
            "addze {carry}, {p_hi2}",                     // carry = p_hi2 + CA bit
            "stw {dst_v2}, 8({dst})",                     // Store accumulated dst[2]

            // [Limb 3 Carry Chain Accumulation]
            "addc {p_lo3}, {p_lo3}, {carry}",             // p_lo3 += carry, set CA bit
            "addze {p_hi3}, {p_hi3}",                     // p_hi3 += CA bit
            "addc {dst_v3}, {dst_v3}, {p_lo3}",           // dst_v3 += p_lo3, set CA bit
            "addze {carry}, {p_hi3}",                     // carry = p_hi3 + CA bit
            "stw {dst_v3}, 12({dst})",                    // Store accumulated dst[3]

            // Advance pointers by 4 limbs (16 bytes) and loop via CTR
            "addi {src}, {src}, 16",
            "addi {dst}, {dst}, 16",
            "bdnz 2b",                                    // Decrement CTR and branch if CTR != 0

            // Remainder processing entry point (0 to 3 limbs)
            "1:",
            "cmpwi {rem}, 0",
            "beq 3f",
            "mtctr {rem}",                                // Load remainder count into CTR
            "addi {src}, {src}, -4",
            "addi {dst}, {dst}, -4",

            ".p2align 4",
            // 1-limb unrolled tail loop using load-update (`lwzu`)
            "4:",
            "lwzu {src_v0}, 4({src})",                    // Load src limb and update pointer (+4)
            "lwzu {dst_v0}, 4({dst})",                    // Load dst limb and update pointer (+4)
            "mullw {p_lo0}, {src_v0}, {scalar}",          // Low 32-bit product
            "mulhwu {p_hi0}, {src_v0}, {scalar}",         // High 32-bit product
            "addc {p_lo0}, {p_lo0}, {carry}",             // Add incoming carry, sets CA bit
            "addze {p_hi0}, {p_hi0}",                     // Propagate carry bit
            "addc {dst_v0}, {dst_v0}, {p_lo0}",           // Accumulate into destination limb
            "addze {carry}, {p_hi0}",                     // Update running carry
            "stw {dst_v0}, 0({dst})",                     // Store updated limb

            "bdnz 4b",                                    // Decrement CTR and branch if CTR != 0

            // Tail completion
            "3:",

            carry = inout(reg) carry,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            scalar = in(reg) scalar,
            src_v0 = out(reg) _,
            src_v1 = out(reg) _,
            src_v2 = out(reg) _,
            src_v3 = out(reg) _,
            dst_v0 = out(reg) _,
            dst_v1 = out(reg) _,
            dst_v2 = out(reg) _,
            dst_v3 = out(reg) _,
            p_lo0 = out(reg) _,
            p_hi0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi1 = out(reg) _,
            p_lo2 = out(reg) _,
            p_hi2 = out(reg) _,
            p_lo3 = out(reg) _,
            p_hi3 = out(reg) _,
            options(nostack)
        );
    }
    carry
}
