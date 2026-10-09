//! MIPS 64-bit multiply-add limb kernel using `HI` and `LO` products.
//!
//! Uses hardware 64x64->128-bit integer multiplier (`dmultu` with `mflo`/`mfhi`),
//! non-trapping addition (`daddu`), and branchless carry capture via `sltu`.

use core::arch::asm;

use super::Limb;

/// Accumulates `src * scalar` into `dst` and returns the high carry limb.
///
/// Two limbs are processed per block, followed by an optional scalar tail.
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
    let mut carry_in: Limb = 0;
    let chunks = len >> 1;
    let rem = len & 1;

    // SAFETY: 2*chunks + rem == len bounds every aligned load and store in
    // the initialized spans. Eight-byte pointer advances stay within the live
    // allocations. Stores target only the disjoint writable destination.
    // Zero chunks and remainder skip all pointer accesses.
    unsafe {
        asm!(
            ".set noat",                                  // Disable assembler temporary register ($at) clobber
            "beqz {chunks}, 2f",                          // If chunks == 0, jump to remainder loop (2f)

            // Main 2-way unrolled loop body
            "1:",

            // [Limb 0 Multiply-Accumulate]
            "ld {s0}, 0({src})",                          // Load src[0] (64-bit Load Doubleword)
            "ld {s1}, 8({src})",                          // Load src[1]
            "ld {d0}, 0({dst})",                          // Load dst[0]
            "ld {d1}, 8({dst})",                          // Load dst[1]

            "dmultu {s0}, {scalar}",                      // HI:LO = src[0] * scalar (128-bit unsigned product)
            "mflo {p_lo0}",                               // Move low 64 bits from LO register
            "mfhi {p_hi0}",                               // Move high 64 bits from HI register
            "daddu {t_lo}, {p_lo0}, {carry_in}",          // t_lo = p_lo0 + carry_in (non-trapping add)
            "sltu {ca}, {t_lo}, {p_lo0}",                 // ca = (t_lo < p_lo0) ? 1 : 0 (detect carry)
            "daddu {p_hi0}, {p_hi0}, {ca}",               // p_hi0 += ca (absorb carry into high product)
            "daddu {t0}, {t_lo}, {d0}",                   // t0 = t_lo + dst[0]
            "sltu {cb}, {t0}, {d0}",                      // cb = (t0 < dst[0]) ? 1 : 0 (detect destination carry)
            "daddu {carry_in}, {p_hi0}, {cb}",            // carry_in = p_hi0 + cb (new running carry)
            "sd {t0}, 0({dst})",                          // Store accumulated result to dst[0]

            // [Limb 1 Multiply-Accumulate]
            "dmultu {s1}, {scalar}",                      // HI:LO = src[1] * scalar
            "mflo {p_lo1}",                               // Low 64 bits
            "mfhi {p_hi1}",                               // High 64 bits
            "daddu {t_lo}, {p_lo1}, {carry_in}",          // t_lo = p_lo1 + carry_in
            "sltu {ca}, {t_lo}, {p_lo1}",                 // ca = 1 if carry occurred
            "daddu {p_hi1}, {p_hi1}, {ca}",               // p_hi1 += ca
            "daddu {t0}, {t_lo}, {d1}",                   // t0 = t_lo + dst[1]
            "sltu {cb}, {t0}, {d1}",                      // cb = 1 if destination carry occurred
            "daddu {carry_in}, {p_hi1}, {cb}",            // Update running carry
            "sd {t0}, 8({dst})",                          // Store to dst[1]

            // Advance pointers by 2 limbs (16 bytes)
            "daddiu {src}, {src}, 16",
            "daddiu {dst}, {dst}, 16",
            // Decrement chunk counter and repeat while chunks != 0
            "daddiu {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            // Remainder processing (0 or 1 limb)
            "2:",
            "beqz {rem}, 4f",

            // 1-limb tail
            "3:",
            "ld {s0}, 0({src})",                          // Load single src limb
            "ld {d0}, 0({dst})",                          // Load single dst limb
            "dmultu {s0}, {scalar}",                      // HI:LO = src[0] * scalar
            "mflo {p_lo0}",                               // Low 64 bits
            "mfhi {p_hi0}",                               // High 64 bits
            "daddu {t_lo}, {p_lo0}, {carry_in}",          // Add running carry
            "sltu {ca}, {t_lo}, {p_lo0}",                 // Detect carry
            "daddu {p_hi0}, {p_hi0}, {ca}",               // Propagate carry
            "daddu {t0}, {t_lo}, {d0}",                   // Accumulate into destination limb
            "sltu {cb}, {t0}, {d0}",                      // Detect destination carry
            "daddu {carry_in}, {p_hi0}, {cb}",            // Update running carry
            "sd {t0}, 0({dst})",                          // Store updated limb

            // Tail completion
            "4:",

            carry_in = inout(reg) carry_in,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            scalar = in(reg) scalar,
            s0 = out(reg) _,
            s1 = out(reg) _,
            d0 = out(reg) _,
            d1 = out(reg) _,
            p_lo0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi0 = out(reg) _,
            p_hi1 = out(reg) _,
            t_lo = out(reg) _,
            t0 = out(reg) _,
            ca = out(reg) _,
            cb = out(reg) _,
            options(nostack)
        );
    }
    carry_in
}
