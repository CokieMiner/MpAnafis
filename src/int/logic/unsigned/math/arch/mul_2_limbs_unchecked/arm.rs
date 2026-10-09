//! ARM 32-bit (`ARMv6` / ARMv7-A) write-only dual-row multiplication kernel using `umaal`.
//!
//! Evaluates `dst = src * (s0 + s1 * B)` in a single write-only pass using
//! the four-operand product-and-sum instruction `umaal`.

use core::arch::asm;

use super::Limb;

/// Write `src * (s0 + s1 * B)` into `dst` without reading its old contents.
///
/// Computes:
///
/// ```text
///   dst[0..len+2] = src[0..len] x (s0 + s1 x 2^32)
/// ```
///
/// Utilizes `ARMv6`/v7 `umaal` to accumulate products and carries directly into output registers
/// without condition-flag dependencies. Each block processes two source limbs.
///
/// # Safety
///
/// - For nonzero `len`, aligned `dst` must cover `len + 2` writable limbs,
///   which may be uninitialized; aligned `src` must cover `len` initialized limbs.
/// - `len + 2` must fit `usize`, and both byte spans must fit `isize::MAX`.
/// - Source and destination spans must be disjoint.
/// - The CPU must support `ARMv6` `umaal` in ARM instruction mode.
#[expect(
    clippy::inline_always,
    reason = "inlining joins the two-row initializer to complete basecase multiplication"
)]
#[inline(always)]
pub unsafe fn mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) {
    if len == 0 {
        return;
    }

    let carry0: Limb = 0;
    let carry1: Limb = 0;
    let pending1: Limb = 0;
    let chunks = len >> 1;

    // SAFETY: disjoint aligned spans provide len initialized inputs and len+2
    // writable outputs. Pair blocks and the odd tail consume exactly len
    // inputs before the final two stores. UMAAL's product plus two limbs is
    // <= B^2-1; the final high carry is <= s1. No destination is read before
    // initialization. All clobbers are outputs; the backend gate enables UMAAL.
    unsafe {
        asm!(
            "cmp {chunks}, #0",                          // Compare 2-limb chunk counter with 0
            "beq 2f",                                    // If chunks == 0, skip to remainder (2f)

            // Main 2-way unrolled loop body
            "1:",

            // [Limb 0: Fused MAC across both rows]
            "ldr {s}, [{src}], #4",                      // Load src[0] and advance pointer by 4 bytes
            "umaal {pending1}, {carry0}, {s}, {s0}",     // (carry0:pending1) = pending1 + carry0 + (s * s0)
            "str {pending1}, [{dst}], #4",               // Store finalized dst[0] and advance dst pointer
            "mov {pending1}, #0",                        // Clear pending1 register for row 1
            "umaal {pending1}, {carry1}, {s}, {s1}",     // (carry1:pending1) = 0 + carry1 + (s * s1)

            // [Limb 1: Fused MAC across both rows]
            "ldr {s}, [{src}], #4",                      // Load src[1] and advance pointer by 4 bytes
            "umaal {pending1}, {carry0}, {s}, {s0}",     // (carry0:pending1) = pending1 + carry0 + (s * s0)
            "str {pending1}, [{dst}], #4",               // Store finalized dst[1] and advance dst pointer
            "mov {pending1}, #0",                        // Clear pending1 register for row 1
            "umaal {pending1}, {carry1}, {s}, {s1}",     // (carry1:pending1) = 0 + carry1 + (s * s1)

            "subs {chunks}, {chunks}, #1",               // Decrement chunk counter
            "bne 1b",                                    // Repeat while chunks != 0

            // Remainder processing (0 or 1 limb)
            "2:",
            "tst {len}, #1",                             // Test if len is odd
            "beq 3f",                                    // If even, skip remainder (3f)

            // 1-limb tail
            "ldr {s}, [{src}], #4",                      // Load single src limb
            "umaal {pending1}, {carry0}, {s}, {s0}",     // Row 0 fused MAC
            "str {pending1}, [{dst}], #4",               // Store finalized dst limb
            "mov {pending1}, #0",                        // Clear pending1
            "umaal {pending1}, {carry1}, {s}, {s1}",     // Row 1 fused MAC

            // Epilogue: Flush trailing high row 1 limb + remaining carry
            "3:",
            "adds {pending1}, {pending1}, {carry0}",     // pending1 += carry0, set C flag
            "adc {carry1}, {carry1}, #0",                // carry1 += C flag + 0
            "str {pending1}, [{dst}], #4",               // Store dst[len]
            "str {carry1}, [{dst}]",                     // Store final high limb dst[len+1]

            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = in(reg) len,
            chunks = inout(reg) chunks => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            carry0 = inout(reg) carry0 => _,
            carry1 = inout(reg) carry1 => _,
            pending1 = inout(reg) pending1 => _,
            s = out(reg) _,
            options(nostack)
        );
    }
}
