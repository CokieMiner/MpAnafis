//! ARM 32-bit (`ARMv6` / ARMv7-A) fused dual-row multiply-add kernel using `umaal`.
//!
//! `umaal` computes `(carry:limb) = limb + carry + src * scalar`.
//! Two source limbs are processed per block, followed by an optional scalar tail.

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

    if len == 0 {
        return (0, 0);
    }

    let chunks = len >> 1;
    let rem = len & 1;

    // SAFETY: 2*chunks + rem == len bounds every source[j] and destination[j+1]
    // access. Four-byte advances preserve alignment within the initialized
    // spans. The destination is writable and disjoint from the source.
    unsafe {
        asm!(
            "cmp {chunks}, #0",                          // Compare 2-limb chunk counter with 0
            "beq 2f",                                    // If chunks == 0, skip to remainder loop (2f)
            ".p2align 2",

            // Main 2-way unrolled loop body
            "1:",

            // [Limb 0: Fused 4-operand MAC across both rows]
            "ldr {s}, [{src}], #4",                      // Load src[0] and advance pointer by 4 bytes
            "ldr {d0}, [{dst}]",                         // Load dst[0]
            "ldr {d1}, [{dst}, #4]",                     // Load dst[1]
            "umaal {d0}, {c0}, {s}, {s0}",               // (c0:d0) = d0 + c0 + (s * s0)
            "umaal {d1}, {c1}, {s}, {s1}",               // (c1:d1) = d1 + c1 + (s * s1)
            "str {d0}, [{dst}], #4",                     // Store finalized dst[0] and advance dst pointer
            "str {d1}, [{dst}]",                         // Store intermediate dst[1]

            // [Limb 1: Fused 4-operand MAC across both rows]
            "ldr {s}, [{src}], #4",                      // Load src[1]
            "ldr {d0}, [{dst}]",                         // Load dst[1] (updated in previous step)
            "ldr {d1}, [{dst}, #4]",                     // Load dst[2]
            "umaal {d0}, {c0}, {s}, {s0}",               // (c0:d0) = d0 + c0 + (s * s0)
            "umaal {d1}, {c1}, {s}, {s1}",               // (c1:d1) = d1 + c1 + (s * s1)
            "str {d0}, [{dst}], #4",                     // Store finalized dst[1] and advance pointer
            "str {d1}, [{dst}]",                         // Store intermediate dst[2]

            "subs {chunks}, {chunks}, #1",               // Decrement chunk counter
            "bne 1b",                                    // Repeat loop while chunks != 0

            // Remainder processing (0 or 1 limb)
            "2:",
            "cmp {rem}, #0",                             // Compare remainder count with 0
            "beq 4f",                                    // If rem == 0, skip to end (4f)

            // 1-limb tail
            "3:",
            "ldr {s}, [{src}], #4",                      // Load single src limb
            "ldr {d0}, [{dst}]",                         // Load single dst limb
            "ldr {d1}, [{dst}, #4]",                     // Load next dst limb
            "umaal {d0}, {c0}, {s}, {s0}",               // Row 0 fused MAC
            "umaal {d1}, {c1}, {s}, {s1}",               // Row 1 fused MAC
            "str {d0}, [{dst}], #4",                     // Store dst limb
            "str {d1}, [{dst}]",                         // Store final dst limb

            // Tail completion
            "4:",

            c0 = inout(reg) c0,
            c1 = inout(reg) c1,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            s = out(reg) _,
            d0 = out(reg) _,
            d1 = out(reg) _,
            options(nostack)
        );
    }
    (c0, c1)
}
