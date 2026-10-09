//! RISC-V 64-bit fused multiply-subtract limb kernel.
//!
//! Uses 64x64->128-bit unsigned multipliers (`mul`/`mulhu`) and branchless
//! overflow/underflow detection using `sltu` (Set Less Than Unsigned).

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// For B = 2^64, `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
/// Two-limb blocks use unsigned comparisons to capture carry and borrow.
///
/// # Safety
///
/// Both pointers must cover `len` aligned, initialized limbs in disjoint spans;
/// `dst` requires exclusive access. Each span's byte length must fit in `isize`.
/// Zero length permits null pointers. The CPU must support the M extension.
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
        // SAFETY: the disjoint aligned initialized spans contain 1..=3 limbs. Each
        // iteration accesses exactly one limb and advances by eight bytes.
        // The M extension supplies both product halves; unsigned comparisons
        // independently preserve product carry and subtraction borrow.
        unsafe {
            asm!(
                "ld {s}, 0({src})",
                "ld {d}, 0({dst})",
                "mul {lo}, {s}, {scalar}",
                "mulhu {carry}, {s}, {scalar}",
                "sltu {borrow}, {d}, {lo}",
                "sub {d}, {d}, {lo}",
                "sd {d}, 0({dst})",
                "addi {count}, {count}, -1",
                "beqz {count}, 3f",
                "2:",
                "addi {src}, {src}, 8",
                "addi {dst}, {dst}, 8",
                "ld {s}, 0({src})",
                "ld {d}, 0({dst})",
                "mul {lo}, {s}, {scalar}",
                "mulhu {hi}, {s}, {scalar}",
                "add {lo}, {lo}, {carry}",
                "sltu {flag}, {lo}, {carry}",
                "add {carry}, {hi}, {flag}",
                "sltu {flag}, {d}, {lo}",
                "sub {d}, {d}, {lo}",
                "sltu {lo}, {d}, {borrow}",
                "sub {d}, {d}, {borrow}",
                "or {borrow}, {flag}, {lo}",
                "sd {d}, 0({dst})",
                "addi {count}, {count}, -1",
                "bnez {count}, 2b",
                "3:",
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

    // SAFETY: len >= 4 gives at least two pairs and at most one tail limb in
    // the aligned initialized disjoint spans. Product plus carry is below
    // B^2. Each SLTU captures underflow before its operands are overwritten;
    // OR preserves binary borrow. Early outputs cannot overwrite live inputs.
    // The caller supplies the M extension and exclusive destination access.
    unsafe {
        asm!(
            // Main 2-way unrolled loop body
            "1:",

            // [Load 2 Source and 2 Destination Limbs]
            "ld {s0}, 0({src})",                         // Load src[0]
            "ld {s1}, 8({src})",                         // Load src[1]
            "ld {d0}, 0({dst})",                         // Load dst[0]
            "ld {d1}, 8({dst})",                         // Load dst[1]

            // [Limb 0 Multiply-Subtract]
            "mul {p_lo0}, {s0}, {scalar}",               // p_lo0 = low 64 bits of src[0] * scalar
            "mulhu {p_hi0}, {s0}, {scalar}",             // p_hi0 = high 64 bits of src[0] * scalar
            "add {t_lo}, {p_lo0}, {carry_in}",           // t_lo = p_lo0 + carry_in
            "sltu {ca}, {t_lo}, {p_lo0}",                // ca = 1 if addition wrapped, else 0
            "add {carry_in}, {p_hi0}, {ca}",             // carry_in = p_hi0 + ca
            "sub {t0}, {d0}, {t_lo}",                    // t0 = d0 - t_lo
            "sltu {b0}, {d0}, {t_lo}",                   // b0 = 1 if first subtraction underflowed
            "sub {t1}, {t0}, {borrow_out}",              // t1 = t0 - borrow_out
            "sltu {b1}, {t0}, {borrow_out}",             // b1 = 1 if second subtraction underflowed
            "or {borrow_out}, {b0}, {b1}",               // borrow_out = b0 | b1 (combined borrow)
            "sd {t1}, 0({dst})",                         // Store updated dst[0]

            // [Limb 1 Multiply-Subtract]
            "mul {p_lo1}, {s1}, {scalar}",               // Low 64 bits of src[1] * scalar
            "mulhu {p_hi1}, {s1}, {scalar}",             // High 64 bits of src[1] * scalar
            "add {t_lo}, {p_lo1}, {carry_in}",           // t_lo = p_lo1 + carry_in
            "sltu {ca}, {t_lo}, {p_lo1}",                // ca = 1 if addition wrapped
            "add {carry_in}, {p_hi1}, {ca}",             // carry_in = p_hi1 + ca
            "sub {t0}, {d1}, {t_lo}",                    // t0 = d1 - t_lo
            "sltu {b0}, {d1}, {t_lo}",                   // First borrow
            "sub {t1}, {t0}, {borrow_out}",              // t1 = t0 - borrow
            "sltu {b1}, {t0}, {borrow_out}",             // Second borrow
            "or {borrow_out}, {b0}, {b1}",               // Combined borrow
            "sd {t1}, 8({dst})",                         // Store updated dst[1]

            // Advance pointers by 2 limbs (16 bytes) and loop
            "addi {src}, {src}, 16",
            "addi {dst}, {dst}, 16",
            "addi {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            // Remainder processing (0 or 1 limb)
            "2:",
            "beqz {rem}, 4f",

            // 1-limb tail
            "3:",
            "ld {s0}, 0({src})",                         // Load single src limb
            "ld {d0}, 0({dst})",                         // Load single dst limb
            "mul {p_lo0}, {s0}, {scalar}",               // Low 64-bit product
            "mulhu {p_hi0}, {s0}, {scalar}",             // High 64-bit product
            "add {t_lo}, {p_lo0}, {carry_in}",           // Add incoming carry
            "sltu {ca}, {t_lo}, {p_lo0}",                // Detect carry out
            "add {carry_in}, {p_hi0}, {ca}",             // Propagate carry
            "sub {t0}, {d0}, {t_lo}",                    // Subtract product
            "sltu {b0}, {d0}, {t_lo}",                   // First borrow
            "sub {t1}, {t0}, {borrow_out}",              // Subtract previous borrow
            "sltu {b1}, {t0}, {borrow_out}",             // Second borrow
            "or {borrow_out}, {b0}, {b1}",               // Combine borrow bits
            "sd {t1}, 0({dst})",                         // Store updated limb

            // Tail completion
            "4:",

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
            p_hi0 = out(reg) _,
            p_lo1 = out(reg) _,
            p_hi1 = out(reg) _,
            t_lo = out(reg) _,
            ca = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            b0 = out(reg) _,
            b1 = out(reg) _,
            options(nostack)
        );
    }
    (carry_in, borrow_out)
}
