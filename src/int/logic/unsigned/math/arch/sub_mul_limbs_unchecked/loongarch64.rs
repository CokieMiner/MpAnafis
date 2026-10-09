//! `LoongArch64` fused multiply-subtract limb kernel.
//!
//! Uses 64x64->128-bit unsigned multipliers (`mul.d`/`mulh.du`), branchless carry
//! propagation via `sltu`, and dual-stage borrow capture (`sub.d`/`sltu`/`or`).

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
        // aligned initialized disjoint spans. Pointers advance by eight bytes per
        // limb; unsigned comparisons retain the independent carry and borrow.
        unsafe {
            asm!(
                "ld.d {s}, {src}, 0",
                "ld.d {d}, {dst}, 0",
                "mul.d {lo}, {s}, {scalar}",
                "mulh.du {carry}, {s}, {scalar}",
                "sltu {borrow}, {d}, {lo}",
                "sub.d {d}, {d}, {lo}",
                "st.d {d}, {dst}, 0",
                "addi.d {count}, {count}, -1",
                "beqz {count}, 3f",
                "2:",
                "addi.d {src}, {src}, 8",
                "addi.d {dst}, {dst}, 8",
                "ld.d {s}, {src}, 0",
                "ld.d {d}, {dst}, 0",
                "mul.d {lo}, {s}, {scalar}",
                "mulh.du {hi}, {s}, {scalar}",
                "add.d {lo}, {lo}, {carry}",
                "sltu {flag}, {lo}, {carry}",
                "add.d {carry}, {hi}, {flag}",
                "sltu {flag}, {d}, {lo}",
                "sub.d {d}, {d}, {lo}",
                "sltu {lo}, {d}, {borrow}",
                "sub.d {d}, {d}, {borrow}",
                "or {borrow}, {flag}, {lo}",
                "st.d {d}, {dst}, 0",
                "addi.d {count}, {count}, -1",
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
    let mut borrow_in: Limb = 0;
    let chunks = len >> 1;
    let rem = len & 1;

    // SAFETY: len >= 4 gives at least two complete pairs and at most one tail
    // limb within the aligned initialized disjoint spans. Product plus carry
    // is below B^2. Each SLTU reads its subtraction operands before either is
    // overwritten; OR combines the two binary borrows. Early outputs remain
    // distinct from all live inputs, and each pointer advances by sixteen bytes.
    unsafe {
        asm!(
            // Main 2-way unrolled loop body
            "1:",

            // [Load 2 Source and 2 Destination Limbs]
            "ld.d {s0}, {src}, 0",                       // Load src[0]
            "ld.d {s1}, {src}, 8",                       // Load src[1]
            "ld.d {d0}, {dst}, 0",                       // Load dst[0]
            "ld.d {d1}, {dst}, 8",                       // Load dst[1]

            // [Limb 0 Multiply-Subtract]
            "mul.d {p_lo0}, {s0}, {scalar}",             // Low 64 bits of src[0] * scalar
            "mulh.du {p_hi0}, {s0}, {scalar}",           // High 64 bits of src[0] * scalar
            "add.d {t_lo}, {p_lo0}, {carry_in}",         // t_lo = p_lo0 + carry_in
            "sltu {ca}, {t_lo}, {p_lo0}",                // ca = 1 if carry addition wrapped
            "add.d {carry_in}, {p_hi0}, {ca}",           // carry_in = p_hi0 + ca
            "sub.d {t0}, {d0}, {t_lo}",                  // t0 = dst[0] - t_lo
            "sltu {b0}, {d0}, {t_lo}",                   // b0 = 1 if first subtraction underflowed
            "sltu {b1}, {t0}, {borrow_in}",             // Capture underflow before overwriting t0
            "sub.d {t0}, {t0}, {borrow_in}",             // t0 = t0 - borrow_in
            "or {borrow_in}, {b0}, {b1}",                // borrow_in = b0 | b1 (combined borrow)
            "st.d {t0}, {dst}, 0",                       // Store updated dst[0]

            // [Limb 1 Multiply-Subtract]
            "mul.d {p_lo1}, {s1}, {scalar}",             // Low 64 bits of src[1] * scalar
            "mulh.du {p_hi1}, {s1}, {scalar}",           // High 64 bits of src[1] * scalar
            "add.d {t_lo}, {p_lo1}, {carry_in}",         // t_lo = p_lo1 + carry_in
            "sltu {ca}, {t_lo}, {p_lo1}",                // ca = 1 if carry addition wrapped
            "add.d {carry_in}, {p_hi1}, {ca}",           // carry_in = p_hi1 + ca
            "sub.d {t0}, {d1}, {t_lo}",                  // t0 = dst[1] - t_lo
            "sltu {b0}, {d1}, {t_lo}",                   // First borrow
            "sltu {b1}, {t0}, {borrow_in}",             // Capture the second borrow from t0
            "sub.d {t0}, {t0}, {borrow_in}",             // Subtract borrow
            "or {borrow_in}, {b0}, {b1}",                // Combined borrow
            "st.d {t0}, {dst}, 8",                       // Store updated dst[1]

            // Advance pointers by 2 limbs (16 bytes) and loop
            "addi.d {src}, {src}, 16",
            "addi.d {dst}, {dst}, 16",
            "addi.d {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            // Remainder processing (0 or 1 limb)
            "2:",
            "beqz {rem}, 4f",

            // 1-limb tail
            "3:",
            "ld.d {s0}, {src}, 0",                       // Load single src limb
            "ld.d {d0}, {dst}, 0",                       // Load single dst limb
            "mul.d {p_lo0}, {s0}, {scalar}",             // Low 64-bit product
            "mulh.du {p_hi0}, {s0}, {scalar}",           // High 64-bit product
            "add.d {t_lo}, {p_lo0}, {carry_in}",         // Add incoming carry
            "sltu {ca}, {t_lo}, {p_lo0}",                // Detect carry out
            "add.d {carry_in}, {p_hi0}, {ca}",           // Propagate carry
            "sub.d {t0}, {d0}, {t_lo}",                  // Subtract product
            "sltu {b0}, {d0}, {t_lo}",                   // First borrow
            "sltu {b1}, {t0}, {borrow_in}",             // Capture the second borrow from t0
            "sub.d {t0}, {t0}, {borrow_in}",             // Subtract previous borrow
            "or {borrow_in}, {b0}, {b1}",                // Combine borrow bits
            "st.d {t0}, {dst}, 0",                       // Store updated limb

            // Tail completion
            "4:",

            carry_in = inout(reg) carry_in,
            borrow_in = inout(reg) borrow_in,
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
            b0 = out(reg) _,
            b1 = out(reg) _,
            options(nostack)
        );
    }
    (carry_in, borrow_in)
}
