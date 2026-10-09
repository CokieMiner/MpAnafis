//! MIPS 32-bit fused multiply-subtract limb kernel.
//!
//! Uses 32x32->64-bit hardware multipliers (`multu`/`mflo`/`mfhi`), non-trapping addition/subtraction
//! (`addu`/`subu`), and branchless carry/borrow capture via `sltu`.

use core::arch::asm;

use super::Limb;

/// Multiply `len` 32-bit limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// For B = 2^32, `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
/// Two-limb blocks extract product halves from HI:LO and retain borrow separately.
///
/// # Safety
///
/// Both pointers must cover `len` aligned, initialized limbs in disjoint spans;
/// `dst` requires exclusive access. Each span's byte length must fit in `isize`.
/// Zero length permits null pointers.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "short and unrolled assembly paths share one call boundary in the arithmetic hot path"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    if len < 4 {
        if len == 0 {
            return (0, 0);
        }
        let carry: Limb;
        let borrow: Limb;
        // SAFETY: the caller supplies 1..=3 aligned initialized limbs in disjoint
        // spans. Each iteration accesses one limb. HI/LO hold the product;
        // unsigned comparisons preserve carry and borrow independently.
        unsafe {
            asm!(
                ".set push",
                ".set noat",
                "lw {s}, 0({src})",
                "lw {d}, 0({dst})",
                "multu {s}, {scalar}",
                "mflo {lo}",
                "mfhi {carry}",
                "sltu {borrow}, {d}, {lo}",
                "subu {d}, {d}, {lo}",
                "sw {d}, 0({dst})",
                "addiu {count}, {count}, -1",
                "beqz {count}, 3f",
                "2:",
                "addiu {src}, {src}, 4",
                "addiu {dst}, {dst}, 4",
                "lw {s}, 0({src})",
                "lw {d}, 0({dst})",
                "multu {s}, {scalar}",
                "mflo {lo}",
                "mfhi {hi}",
                "addu {lo}, {lo}, {carry}",
                "sltu {flag}, {lo}, {carry}",
                "addu {carry}, {hi}, {flag}",
                "sltu {flag}, {d}, {lo}",
                "subu {d}, {d}, {lo}",
                "sltu {lo}, {d}, {borrow}",
                "subu {d}, {d}, {borrow}",
                "or {borrow}, {flag}, {lo}",
                "sw {d}, 0({dst})",
                "addiu {count}, {count}, -1",
                "bnez {count}, 2b",
                "3:",
                ".set pop",
                src = inout(reg) src => _,
                dst = inout(reg) dst => _,
                count = inout(reg) len => _,
                scalar = in(reg) scalar,
                carry = out(reg) carry,
                borrow = out(reg) borrow,
                s = out(reg) _,
                d = out(reg) _,
                lo = out(reg) _,
                hi = out(reg) _,
                flag = out(reg) _,
                options(nostack),
            );
        }
        return (carry, borrow);
    }
    let mut carry_in: Limb = 0;
    let mut borrow_out: Limb = 0;
    let chunks = len >> 1;
    let rem = len & 1;

    // SAFETY: len >= 4 supplies at least two complete pairs and at most one
    // tail limb in the aligned initialized disjoint spans. Product plus carry
    // is below B^2. HI:LO are read after each MULTU; SLTU captures underflow
    // before either operand is overwritten. Early outputs preserve live
    // inputs, and each pointer advances eight bytes after a complete pair.
    unsafe {
        asm!(
            ".set push",
            ".set noat",

            // Main 2-way unrolled loop body
            "1:",

            // [Load 2 Source and 2 Destination Limbs (32-bit each)]
            "lw {s0}, 0({src})",                         // Load src[0]
            "lw {s1}, 4({src})",                         // Load src[1]
            "lw {d0}, 0({dst})",                         // Load dst[0]
            "lw {d1}, 4({dst})",                         // Load dst[1]

            // [Limb 0 Multiply-Subtract]
            "multu {s0}, {scalar}",                      // HI:LO = src[0] * scalar (64-bit product)
            "mflo {p_lo0}",                              // Extract low 32 bits from LO
            "mfhi {p_hi0}",                              // Extract high 32 bits from HI
            "addu {t_lo}, {p_lo0}, {carry_in}",          // t_lo = p_lo0 + carry_in
            "sltu {ca}, {t_lo}, {p_lo0}",                // ca = 1 if addition wrapped
            "addu {carry_in}, {p_hi0}, {ca}",            // carry_in = p_hi0 + ca
            "subu {t0}, {d0}, {t_lo}",                   // t0 = d0 - t_lo
            "sltu {b0}, {d0}, {t_lo}",                   // b0 = 1 if first subtraction underflowed
            "subu {t1}, {t0}, {borrow_out}",             // t1 = t0 - borrow_out
            "sltu {b1}, {t0}, {borrow_out}",             // b1 = 1 if second subtraction underflowed
            "or {borrow_out}, {b0}, {b1}",               // borrow_out = b0 | b1 (combined borrow)
            "sw {t1}, 0({dst})",                         // Store updated dst[0]

            // [Limb 1 Multiply-Subtract]
            "multu {s1}, {scalar}",                      // HI:LO = src[1] * scalar
            "mflo {p_lo1}",                              // Extract low 32 bits
            "mfhi {p_hi1}",                              // Extract high 32 bits
            "addu {t_lo}, {p_lo1}, {carry_in}",          // t_lo = p_lo1 + carry_in
            "sltu {ca}, {t_lo}, {p_lo1}",                // ca = 1 if addition wrapped
            "addu {carry_in}, {p_hi1}, {ca}",            // carry_in = p_hi1 + ca
            "subu {t0}, {d1}, {t_lo}",                   // t0 = d1 - t_lo
            "sltu {b0}, {d1}, {t_lo}",                   // First borrow
            "subu {t1}, {t0}, {borrow_out}",             // t1 = t0 - borrow
            "sltu {b1}, {t0}, {borrow_out}",             // Second borrow
            "or {borrow_out}, {b0}, {b1}",               // Combined borrow
            "sw {t1}, 4({dst})",                         // Store updated dst[1]

            // Advance pointers by 2 limbs (8 bytes) and loop
            "addiu {src}, {src}, 8",
            "addiu {dst}, {dst}, 8",
            "addiu {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            // Remainder processing (0 or 1 limb)
            "2:",
            "beqz {rem}, 4f",

            // 1-limb tail
            "3:",
            "lw {s0}, 0({src})",                         // Load single src limb
            "lw {d0}, 0({dst})",                         // Load single dst limb
            "multu {s0}, {scalar}",                      // 32x32->64 product
            "mflo {p_lo0}",                              // Extract LO
            "mfhi {p_hi0}",                              // Extract HI
            "addu {t_lo}, {p_lo0}, {carry_in}",          // Add incoming carry
            "sltu {ca}, {t_lo}, {p_lo0}",                // Detect carry out
            "addu {carry_in}, {p_hi0}, {ca}",            // Propagate carry
            "subu {t0}, {d0}, {t_lo}",                   // Subtract product
            "sltu {b0}, {d0}, {t_lo}",                   // First borrow
            "subu {t1}, {t0}, {borrow_out}",             // Subtract previous borrow
            "sltu {b1}, {t0}, {borrow_out}",             // Second borrow
            "or {borrow_out}, {b0}, {b1}",               // Combine borrow bits
            "sw {t1}, 0({dst})",                         // Store updated limb

            // Tail completion
            "4:",
            ".set pop",

            carry_in = inout(reg) carry_in,
            borrow_out = inout(reg) borrow_out,
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
            t1 = out(reg) _,
            ca = out(reg) _,
            b0 = out(reg) _,
            b1 = out(reg) _,
            options(nostack)
        );
    }
    (carry_in, borrow_out)
}
