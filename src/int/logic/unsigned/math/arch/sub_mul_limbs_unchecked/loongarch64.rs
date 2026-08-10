//! `LoongArch64` multiply-subtract limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// This computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// The subtraction is performed with a two-state accumulator:
///   - `carry_out`:   overflow from the multiplication stage (high word
///     of product + overflow from adding previous carry).
///   - `borrow_out`:  underflow from the subtraction stage.
///
/// The two are returned separately because they propagate independently:
/// `carry_out` advances one limb position in the product, while
/// `borrow_out` represents the combined borrow at the current position.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry_in: Limb = 0; // multiplication carry (hi from previous product)
    let mut borrow_in: Limb = 0; // subtraction borrow (from previous limb)
    let chunks = len >> 1;
    let rem = len & 1;
    // SAFETY: Caller guarantees dst and src hold at least `len` valid u64 values.
    //
    // Unlike addmul, we track two independent accumulators across iterations:
    //
    //   carry_out  =  hi(product) + overflow(lo + carry_in)
    //   borrow_out =  borrow(dst − (lo + carry_in)) | borrow_continue
    //
    // The key insight is that `carry_in` carries the *high word* of the
    // previous product (plus any overflow from adding to lo), while
    // `borrow_in` tracks whether the subtraction itself underflowed.
    // Both are 0 or 1 at any given point.
    unsafe {
        asm!(
            // ── Check for empty input ──────────────────────────────────
            "beqz {chunks}, 2f",     // if chunks == 0, skip loop

            // ── 2-way unrolled loop ────────────────────────────────────
            "1:",
            // Load two source limbs and two destination limbs
            "ld.d {s0}, {src}, 0",   // s0 = src[0]
            "ld.d {s1}, {src}, 8",   // s1 = src[1]
            "ld.d {d0}, {dst}, 0",   // d0 = dst[0]
            "ld.d {d1}, {dst}, 8",   // d1 = dst[1]

            // ── Limb 0: product = s0 × scalar ──────────────────────────
            // Step 1: compute 128-bit product p_hi0:p_lo0 = s0 * scalar
            "mul.d {p_lo0}, {s0}, {scalar}",
            "mulh.du {p_hi0}, {s0}, {scalar}",

            // Step 2: add multiplication carry from previous limb
            //   sum = lo + carry_in
            //   c1  = overflow from that addition (0 or 1)
            //   carry_out = hi + c1
            "add.d {t_lo}, {p_lo0}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo0}",
            "add.d {carry_in}, {p_hi0}, {ca}",

            // Step 3: subtract (product + carry) from destination
            //   diff1 = dst[0] - sum
            //   b0    = borrow from that subtraction (1 if dst < sum)
            "sub.d {t0}, {d0}, {t_lo}",
            "sltu {b0}, {d0}, {t_lo}",

            // Step 4: subtract previous subtraction borrow
            //   Save diff1 (mv) before modifying t0
            //   diff2 = diff1 - borrow_in
            //   b1    = borrow from that subtraction (1 if diff1 < borrow_in)
            "or {t_save}, {t0}, $zero",   // mv t_save, t0  (save before mutation)
            "sub.d {t0}, {t0}, {borrow_in}",
            "sltu {b1}, {t_save}, {borrow_in}",

            // Step 5: combined borrow = b0 | b1
            "or {borrow_in}, {b0}, {b1}",
            // Store result for limb 0
            "st.d {t0}, {dst}, 0",

            // ── Limb 1: product = s1 × scalar ──────────────────────────
            "mul.d {p_lo1}, {s1}, {scalar}",
            "mulh.du {p_hi1}, {s1}, {scalar}",

            // Add carry from limb 0 to low (same pattern as above)
            "add.d {t_lo}, {p_lo1}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo1}",
            "add.d {carry_in}, {p_hi1}, {ca}",

            // Subtract from destination
            "sub.d {t0}, {d1}, {t_lo}",
            "sltu {b0}, {d1}, {t_lo}",

            // Subtract borrow from limb 0
            "or {t_save}, {t0}, $zero",   // mv t_save, t0
            "sub.d {t0}, {t0}, {borrow_in}",
            "sltu {b1}, {t_save}, {borrow_in}",

            // Combined borrow for next block
            "or {borrow_in}, {b0}, {b1}",
            "st.d {t0}, {dst}, 8",

            // Advance pointers by 16 bytes (2 limbs)
            "addi.d {src}, {src}, 16",
            "addi.d {dst}, {dst}, 16",
            "addi.d {chunks}, {chunks}, -1",
            "bnez {chunks}, 1b",

            // ── Single-limb remainder (if len is odd) ──────────────────
            "2:",
            "beqz {rem}, 4f",
            "3:",
            "ld.d {s0}, {src}, 0",
            "ld.d {d0}, {dst}, 0",

            "mul.d {p_lo0}, {s0}, {scalar}",
            "mulh.du {p_hi0}, {s0}, {scalar}",

            "add.d {t_lo}, {p_lo0}, {carry_in}",
            "sltu {ca}, {t_lo}, {p_lo0}",
            "add.d {carry_in}, {p_hi0}, {ca}",

            "sub.d {t0}, {d0}, {t_lo}",
            "sltu {b0}, {d0}, {t_lo}",

            "or {t_save}, {t0}, $zero",   // mv t_save, t0
            "sub.d {t0}, {t0}, {borrow_in}",
            "sltu {b1}, {t_save}, {borrow_in}",

            "or {borrow_in}, {b0}, {b1}",
            "st.d {t0}, {dst}, 0",
            "4:",
            // ── End: carry_in = mul carry, borrow_in = sub borrow ───────

            // ── Operand constraints ────────────────────────────────────
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
            t_save = out(reg) _,
            ca = out(reg) _,
            b0 = out(reg) _,
            b1 = out(reg) _,
            options(nostack)
        );
    }
    (carry_in, borrow_in)
}
