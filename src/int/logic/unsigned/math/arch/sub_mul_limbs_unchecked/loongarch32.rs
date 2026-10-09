//! `LoongArch32` fused multiply-subtract limb kernel.
//!
//! Uses 32x32->64-bit unsigned multipliers (`mul.w`/`mulh.wu`), branchless carry
//! propagation via `sltu`, and dual-stage borrow capture (`sub.w`/`sltu`/`or`).

use core::arch::asm;

use super::Limb;

/// Multiply `len` 32-bit limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// For B = 2^32, `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
/// Two-limb blocks use unsigned comparisons to capture carry and borrow.
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
        // SAFETY: 1 <= len <= 3 bounds each load and store in the caller's
        // aligned initialized disjoint spans. Pointers advance by four bytes per
        // limb; unsigned comparisons retain the independent carry and borrow.
        unsafe {
            asm!(
                "ld.w {s}, {src}, 0",
                "ld.w {d}, {dst}, 0",
                "mul.w {lo}, {s}, {scalar}",
                "mulh.wu {carry}, {s}, {scalar}",
                "sltu {borrow}, {d}, {lo}",
                "sub.w {d}, {d}, {lo}",
                "st.w {d}, {dst}, 0",
                "addi.w {count}, {count}, -1",
                "beqz {count}, 3f",
                "2:",
                "addi.w {src}, {src}, 4",
                "addi.w {dst}, {dst}, 4",
                "ld.w {s}, {src}, 0",
                "ld.w {d}, {dst}, 0",
                "mul.w {lo}, {s}, {scalar}",
                "mulh.wu {hi}, {s}, {scalar}",
                "add.w {lo}, {lo}, {carry}",
                "sltu {flag}, {lo}, {carry}",
                "add.w {carry}, {hi}, {flag}",
                "sltu {flag}, {d}, {lo}",
                "sub.w {d}, {d}, {lo}",
                "sltu {lo}, {d}, {borrow}",
                "sub.w {d}, {d}, {borrow}",
                "or {borrow}, {flag}, {lo}",
                "st.w {d}, {dst}, 0",
                "addi.w {count}, {count}, -1",
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

    // SAFETY: len >= 4 gives at least two complete pairs and at most one tail
    // limb within the aligned initialized disjoint spans. Product plus carry
    // is below B^2. Each SLTU reads its subtraction operands before either is
    // overwritten; OR combines the two binary borrows. Early outputs remain
    // distinct from all live inputs, and each pointer advances by eight bytes.
    unsafe {
        asm!(
            // Main 2-way unrolled loop body
            "1:",

            // [Load 2 Source and 2 Destination Limbs (32-bit each)]
            "ld.w {s0}, {src}, 0",                       // Load src[0]
            "ld.w {s1}, {src}, 4",                       // Load src[1]
            "ld.w {d0}, {dst}, 0",                       // Load dst[0]
            "ld.w {d1}, {dst}, 4",                       // Load dst[1]

            // [Limb 0 Multiply-Subtract]
            "mul.w {p_lo0}, {s0}, {scalar}",             // Low 32 bits of src[0] * scalar
            "mulh.wu {p_hi0}, {s0}, {scalar}",           // High 32 bits of src[0] * scalar
            "add.w {t_lo}, {p_lo0}, {carry_in}",         // t_lo = p_lo0 + carry_in
            "sltu {ca}, {t_lo}, {p_lo0}",                // ca = 1 if carry addition wrapped
            "add.w {carry_in}, {p_hi0}, {ca}",           // carry_in = p_hi0 + ca
            "sub.w {t0}, {d0}, {t_lo}",                  // t0 = dst[0] - t_lo
            "sltu {b0}, {d0}, {t_lo}",                   // b0 = 1 if first subtraction underflowed
            "sub.w {t1}, {t0}, {borrow_out}",            // t1 = t0 - borrow_out
            "sltu {b1}, {t0}, {borrow_out}",             // b1 = 1 if second subtraction underflowed
            "or {borrow_out}, {b0}, {b1}",               // borrow_out = b0 | b1 (combined borrow)
            "st.w {t1}, {dst}, 0",                       // Store updated dst[0]

            // [Limb 1 Multiply-Subtract]
            "mul.w {p_lo1}, {s1}, {scalar}",             // Low 32 bits of src[1] * scalar
            "mulh.wu {p_hi1}, {s1}, {scalar}",           // High 32 bits of src[1] * scalar
            "add.w {t_lo}, {p_lo1}, {carry_in}",         // t_lo = p_lo1 + carry_in
            "sltu {ca}, {t_lo}, {p_lo1}",                // ca = 1 if carry addition wrapped
            "add.w {carry_in}, {p_hi1}, {ca}",           // carry_in = p_hi1 + ca
            "sub.w {t0}, {d1}, {t_lo}",                  // t0 = dst[1] - t_lo
            "sltu {b0}, {d1}, {t_lo}",                   // First borrow
            "sub.w {t1}, {t0}, {borrow_out}",            // t1 = t0 - borrow
            "sltu {b1}, {t0}, {borrow_out}",             // Second borrow
            "or {borrow_out}, {b0}, {b1}",               // Combined borrow
            "st.w {t1}, {dst}, 4",                       // Store updated dst[1]

            // Advance pointers by 2 limbs (8 bytes) and loop
            "addi.w {src}, {src}, 8",
            "addi.w {dst}, {dst}, 8",
            "addi.w {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            // Remainder processing (0 or 1 limb)
            "2:",
            "beqz {rem}, 4f",

            // 1-limb tail
            "3:",
            "ld.w {s0}, {src}, 0",                       // Load single src limb
            "ld.w {d0}, {dst}, 0",                       // Load single dst limb
            "mul.w {p_lo0}, {s0}, {scalar}",             // Low 32-bit product
            "mulh.wu {p_hi0}, {s0}, {scalar}",           // High 32-bit product
            "add.w {t_lo}, {p_lo0}, {carry_in}",         // Add incoming carry
            "sltu {ca}, {t_lo}, {p_lo0}",                // Detect carry out
            "add.w {carry_in}, {p_hi0}, {ca}",           // Propagate carry
            "sub.w {t0}, {d0}, {t_lo}",                  // Subtract product
            "sltu {b0}, {d0}, {t_lo}",                   // First borrow
            "sub.w {t1}, {t0}, {borrow_out}",            // Subtract previous borrow
            "sltu {b1}, {t0}, {borrow_out}",             // Second borrow
            "or {borrow_out}, {b0}, {b1}",               // Combine borrow bits
            "st.w {t1}, {dst}, 0",                       // Store updated limb

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
