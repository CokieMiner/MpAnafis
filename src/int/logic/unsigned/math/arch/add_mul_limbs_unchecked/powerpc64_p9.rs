//! `PowerPC64` (POWER9 / POWER10 ISA 3.0+) multiply-add limb kernel.
//!
//! Uses hardware 3-operand fused multiply-add instructions (`maddld`/`maddhdu`),
//! reducing arithmetic from 6 instructions down to 4 instructions per limb.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
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
/// PowerPC ISA 3.0 introduces `maddld` and `maddhdu`, which compute $a \times b + c$ directly
/// into 64-bit low and high halves respectively without separate addition steps.
/// The loop is 4-way unrolled (32 bytes per iteration), maintaining zero-overhead loop control
/// through the hardware `CTR` register via `bdnz`.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in 64-bit PowerPC POWER9+ hot paths"
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
    // 1. `dst` is valid for writes of `len` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`, `24`, `32`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "cmpldi {chunks}, 0",         // Compare chunks count with 0
            "beq 1f",                     // If chunks == 0, skip to remainder loop (1f)
            "mtctr {chunks}",             // Load loop counter into hardware CTR register

            // Main 4-way unrolled loop body
            "2:",

            // [Load 4 Source and 4 Destination Limbs]
            "ld {src_v0}, 0({src})",      // Load src[0]
            "ld {src_v1}, 8({src})",      // Load src[1]
            "ld {src_v2}, 16({src})",     // Load src[2]
            "ld {src_v3}, 24({src})",     // Load src[3]
            "ld {dst_v0}, 0({dst})",      // Load dst[0]
            "ld {dst_v1}, 8({dst})",      // Load dst[1]
            "ld {dst_v2}, 16({dst})",     // Load dst[2]
            "ld {dst_v3}, 24({dst})",     // Load dst[3]

            // [Limb 0: Fused 3-operand Multiply-Add & Carry Chain]
            "maddld {t0}, {src_v0}, {scalar}, {dst_v0}",  // t0 = (src[0] * scalar + dst[0]).lo
            "maddhdu {c0}, {src_v0}, {scalar}, {dst_v0}", // c0 = (src[0] * scalar + dst[0]).hi
            "addc {dst_v0}, {t0}, {carry}",               // dst_v0 = t0 + carry, sets CA in XER
            "addze {carry}, {c0}",                         // carry = c0 + CA
            "std {dst_v0}, 0({dst})",                     // Store accumulated dst[0]

            // [Limb 1: Fused Multiply-Add & Carry Chain]
            "maddld {t1}, {src_v1}, {scalar}, {dst_v1}",  // t1 = (src[1] * scalar + dst[1]).lo
            "maddhdu {c1}, {src_v1}, {scalar}, {dst_v1}", // c1 = (src[1] * scalar + dst[1]).hi
            "addc {dst_v1}, {t1}, {carry}",               // dst_v1 = t1 + carry, sets CA
            "addze {carry}, {c1}",                         // carry = c1 + CA
            "std {dst_v1}, 8({dst})",                     // Store accumulated dst[1]

            // [Limb 2: Fused Multiply-Add & Carry Chain]
            "maddld {t2}, {src_v2}, {scalar}, {dst_v2}",  // t2 = (src[2] * scalar + dst[2]).lo
            "maddhdu {c2}, {src_v2}, {scalar}, {dst_v2}", // c2 = (src[2] * scalar + dst[2]).hi
            "addc {dst_v2}, {t2}, {carry}",               // dst_v2 = t2 + carry, sets CA
            "addze {carry}, {c2}",                         // carry = c2 + CA
            "std {dst_v2}, 16({dst})",                    // Store accumulated dst[2]

            // [Limb 3: Fused Multiply-Add & Carry Chain]
            "maddld {t3}, {src_v3}, {scalar}, {dst_v3}",  // t3 = (src[3] * scalar + dst[3]).lo
            "maddhdu {c3}, {src_v3}, {scalar}, {dst_v3}", // c3 = (src[3] * scalar + dst[3]).hi
            "addc {dst_v3}, {t3}, {carry}",               // dst_v3 = t3 + carry, sets CA
            "addze {carry}, {c3}",                         // carry = c3 + CA
            "std {dst_v3}, 24({dst})",                    // Store accumulated dst[3]

            // Advance pointers by 4 limbs (32 bytes) and loop via hardware CTR
            "addi {src}, {src}, 32",
            "addi {dst}, {dst}, 32",
            "bdnz 2b",                                    // Decrement CTR and branch if CTR != 0

            // Remainder processing entry point (0 to 3 limbs)
            "1:",
            "cmpldi {rem}, 0",
            "beq 3f",
            "mtctr {rem}",                                // Load remainder count into CTR

            // 1-limb unrolled tail loop
            "4:",
            "ld {src_v0}, 0({src})",                      // Load single src limb
            "ld {dst_v0}, 0({dst})",                      // Load single dst limb
            "maddld {t0}, {src_v0}, {scalar}, {dst_v0}",  // Fused multiply-add low
            "maddhdu {c0}, {src_v0}, {scalar}, {dst_v0}", // Fused multiply-add high
            "addc {dst_v0}, {t0}, {carry}",               // Add carry, sets CA
            "addze {carry}, {c0}",                         // Propagate carry
            "std {dst_v0}, 0({dst})",                     // Store updated limb
            "addi {src}, {src}, 8",                       // Advance src pointer by 8 bytes
            "addi {dst}, {dst}, 8",                       // Advance dst pointer by 8 bytes
            "bdnz 4b",                                    // Decrement CTR and loop if != 0

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
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            c0 = out(reg) _,
            c1 = out(reg) _,
            c2 = out(reg) _,
            c3 = out(reg) _,
            options(nostack)
        );
    }
    carry
}
