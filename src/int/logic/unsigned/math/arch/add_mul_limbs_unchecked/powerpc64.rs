//! `PowerPC64` (POWER8 / POWER9 / POWER10) fused multiply-add limb kernel.
//!
//! Uses 64-bit multipliers (`mulld`/`mulhdu`), carry-chain arithmetic
//! (`addc`/`addze`), and CTR loop control (`bdnz`).

use core::arch::asm;

use super::Limb;

/// Accumulates `src * scalar` into `dst` and returns the high carry limb.
///
/// Four product pairs precede their sequential carry additions in each block.
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// Nonempty spans must cover `len` aligned, initialized limbs in disjoint live
/// allocations of at most `isize::MAX` bytes. The destination must be writable.
#[expect(
    clippy::inline_always,
    reason = "Keep the unrolled hardware row recurrence in the selected multiplication caller"
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

    // SAFETY: 4*chunks + rem == len bounds every aligned read and write in
    // the initialized, disjoint spans. Eight-byte advances stay within live
    // allocations. Tail address prebiasing performs no access; LDU advances
    // to the current limb before loading. Only the writable destination changes.
    // Zero chunks and remainder skip all pointer accesses.
    unsafe {
        asm!(
            "cmpldi {chunks}, 0",         // Compare chunks count with 0
            "beq 1f",                     // If chunks == 0, skip to remainder loop (1f)
            "mtctr {chunks}",             // Load loop counter into hardware CTR register
            ".p2align 4",

            // Main 4-way unrolled loop body
            "2:",

            // [Dual Memory Loads across 4 Limbs]
            "ld {src_v0}, 0({src})",      // Load src[0]
            "ld {src_v1}, 8({src})",      // Load src[1]
            "ld {src_v2}, 16({src})",     // Load src[2]
            "ld {src_v3}, 24({src})",     // Load src[3]
            "ld {dst_v0}, 0({dst})",      // Load dst[0]
            "ld {dst_v1}, 8({dst})",      // Load dst[1]
            "ld {dst_v2}, 16({dst})",     // Load dst[2]
            "ld {dst_v3}, 24({dst})",     // Load dst[3]

            // Four independent low/high product pairs.
            "mulld {p_lo0}, {src_v0}, {scalar}",          // Low 64-bit product 0
            "mulhdu {p_hi0}, {src_v0}, {scalar}",         // High 64-bit product 0
            "mulld {p_lo1}, {src_v1}, {scalar}",          // Low 64-bit product 1
            "mulhdu {p_hi1}, {src_v1}, {scalar}",         // High 64-bit product 1
            "mulld {p_lo2}, {src_v2}, {scalar}",          // Low 64-bit product 2
            "mulhdu {p_hi2}, {src_v2}, {scalar}",         // High 64-bit product 2
            "mulld {p_lo3}, {src_v3}, {scalar}",          // Low 64-bit product 3
            "mulhdu {p_hi3}, {src_v3}, {scalar}",         // High 64-bit product 3

            // [Limb 0 Carry Chain Accumulation]
            "addc {p_lo0}, {p_lo0}, {carry}",             // p_lo0 += carry, set CA bit in XER
            "addze {p_hi0}, {p_hi0}",                     // p_hi0 += CA bit (propagate carry to high product)
            "addc {dst_v0}, {dst_v0}, {p_lo0}",           // dst_v0 += p_lo0, set CA bit in XER
            "addze {carry}, {p_hi0}",                     // carry = p_hi0 + CA bit
            "std {dst_v0}, 0({dst})",                     // Store accumulated dst[0]

            // [Limb 1 Carry Chain Accumulation]
            "addc {p_lo1}, {p_lo1}, {carry}",             // p_lo1 += carry, set CA bit
            "addze {p_hi1}, {p_hi1}",                     // p_hi1 += CA bit
            "addc {dst_v1}, {dst_v1}, {p_lo1}",           // dst_v1 += p_lo1, set CA bit
            "addze {carry}, {p_hi1}",                     // carry = p_hi1 + CA bit
            "std {dst_v1}, 8({dst})",                     // Store accumulated dst[1]

            // [Limb 2 Carry Chain Accumulation]
            "addc {p_lo2}, {p_lo2}, {carry}",             // p_lo2 += carry, set CA bit
            "addze {p_hi2}, {p_hi2}",                     // p_hi2 += CA bit
            "addc {dst_v2}, {dst_v2}, {p_lo2}",           // dst_v2 += p_lo2, set CA bit
            "addze {carry}, {p_hi2}",                     // carry = p_hi2 + CA bit
            "std {dst_v2}, 16({dst})",                    // Store accumulated dst[2]

            // [Limb 3 Carry Chain Accumulation]
            "addc {p_lo3}, {p_lo3}, {carry}",             // p_lo3 += carry, set CA bit
            "addze {p_hi3}, {p_hi3}",                     // p_hi3 += CA bit
            "addc {dst_v3}, {dst_v3}, {p_lo3}",           // dst_v3 += p_lo3, set CA bit
            "addze {carry}, {p_hi3}",                     // carry = p_hi3 + CA bit
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
            "addi {src}, {src}, -8",
            "addi {dst}, {dst}, -8",

            ".p2align 4",
            // 1-limb unrolled tail loop using load-update (`ldu`)
            "4:",
            "ldu {src_v0}, 8({src})",                     // Load src limb and update pointer (+8)
            "ldu {dst_v0}, 8({dst})",                     // Load dst limb and update pointer (+8)
            "mulld {p_lo0}, {src_v0}, {scalar}",          // Low 64-bit product
            "mulhdu {p_hi0}, {src_v0}, {scalar}",         // High 64-bit product
            "addc {p_lo0}, {p_lo0}, {carry}",             // Add incoming carry, set CA bit
            "addze {p_hi0}, {p_hi0}",                     // Propagate carry bit
            "addc {dst_v0}, {dst_v0}, {p_lo0}",           // Accumulate into destination limb
            "addze {carry}, {p_hi0}",                     // Update running carry
            "std {dst_v0}, 0({dst})",                     // Store updated limb

            "bdnz 4b",                                    // Decrement CTR and branch if CTR != 0

            // Tail completion
            "3:",

            carry = inout(reg) carry,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg_nonzero) src => _,
            dst = inout(reg_nonzero) dst => _,
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
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
    }
    carry
}
