//! `PowerPC64` Montgomery reduction step kernel.

use core::arch::asm;

use super::Limb;

/// Fused Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
/// using `PowerPC64` inline assembly (`mulld`, `mulhdu`, `addc`, `addze`).
///
/// For step `i`, this computes:
/// `(out[0..len] + a_i * b[0..len] + q * m[0..len]) / 2^LIMB_BITS`
/// where `q = ((out[0] + a_i * b[0]) * m_inv) mod 2^LIMB_BITS`.
///
/// Stores the shifted result into `out[0..len-1]`, stores the combined low carry into
/// `out[len-1]`, and returns the top overflow carry (either 0 or 1).
///
/// # Safety
/// - `out` must be valid for reads and writes of `len` elements.
/// - `b` and `m` must be valid for reads of `len` elements.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn monty_redc_step_unchecked(
    out: *mut Limb,
    b: *const Limb,
    m: *const Limb,
    len: usize,
    mut a_i: Limb,
    mut m_inv: Limb,
) -> Limb {
    if len == 0 {
        return 0;
    }

    // SAFETY: caller guarantees out, b, and m have at least len elements.
    unsafe {
        asm!(
            // --- Pass 1: out = out + a_i * b ---
            "li {carry}, 0",
            "li {offset}, 0",
            "mtctr {len}",

            ".p2align 4",
            "1:",
            "ldx {val_m_b}, {b}, {offset}",
            "ldx {val_out}, {out}, {offset}",
            "mulld {p_lo}, {val_m_b}, {a_i}",
            "mulhdu {p_hi}, {val_m_b}, {a_i}",
            "addc {p_lo}, {p_lo}, {carry}",
            "addze {p_hi}, {p_hi}",
            "addc {val_out}, {val_out}, {p_lo}",
            "addze {carry}, {p_hi}",
            "stdx {val_out}, {out}, {offset}",
            "addi {offset}, {offset}, 8",
            "bdnz 1b",

            "mr {a_i}, {carry}",          // return carry_b in a_i reg

            // --- Pass 2: q = out[0] * m_inv ---
            "ld {val_out}, 0({out})",
            "mulld {q}, {val_out}, {m_inv}",

            // --- Pass 3: out = (out + q * m) >> 64 ---
            "li {carry}, 0",
            "li {offset}, 0",

            // Step 0 (j=0): compute q * m[0] + out[0], result is 0 mod 2^64, just keep carry
            "ldx {val_m_b}, {m}, {offset}",
            "ldx {val_out}, {out}, {offset}",
            "mulld {p_lo}, {val_m_b}, {q}",
            "mulhdu {p_hi}, {val_m_b}, {q}",
            "addc {p_lo}, {p_lo}, {carry}",
            "addze {p_hi}, {p_hi}",
            "addc {val_out}, {val_out}, {p_lo}",
            "addze {carry}, {p_hi}",
            "addi {offset}, {offset}, 8",
            "subi {offset_sub}, {offset}, 8",   // offset_sub = 0 (for j=1 store)

            "cmpdi {len}, 1",
            "beq 3f",

            "subi {loops}, {len}, 1",
            "mtctr {loops}",

            ".p2align 4",
            "2:",
            "ldx {val_m_b}, {m}, {offset}",
            "ldx {val_out}, {out}, {offset}",
            "mulld {p_lo}, {val_m_b}, {q}",
            "mulhdu {p_hi}, {val_m_b}, {q}",
            "addc {p_lo}, {p_lo}, {carry}",
            "addze {p_hi}, {p_hi}",
            "addc {val_out}, {val_out}, {p_lo}",
            "addze {carry}, {p_hi}",
            "stdx {val_out}, {out}, {offset_sub}",
            "addi {offset}, {offset}, 8",
            "addi {offset_sub}, {offset_sub}, 8",
            "bdnz 2b",

            "3:",
            "mr {m_inv}, {carry}",        // return carry_m in m_inv reg

            out = in(reg_nonzero) out,
            b = in(reg_nonzero) b,
            m = in(reg_nonzero) m,
            len = in(reg) len,
            a_i = inout(reg) a_i,         // outputs carry_b
            m_inv = inout(reg) m_inv,     // outputs carry_m
            carry = out(reg) _,
            offset = out(reg_nonzero) _,
            offset_sub = out(reg_nonzero) _,
            val_m_b = out(reg) _,
            val_out = out(reg) _,
            p_lo = out(reg) _,
            p_hi = out(reg) _,
            q = out(reg) _,
            loops = out(reg) _,
            out("ctr") _,
            out("xer") _,
            out("cr0") _,
            options(nostack)
        );
        let (final_sum, final_carry) = a_i.overflowing_add(m_inv);
        *out.add(len.wrapping_sub(1)) = final_sum;
        Limb::from(final_carry)
    }
}
