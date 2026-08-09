//! `AArch64` Montgomery reduction step kernel.

use core::arch::asm;

use super::Limb;

/// Fused Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
/// using `AArch64` inline assembly (`mul`, `umulh`, `adds`, `adc`).
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
            "mov {carry}, xzr",
            "mov {offset}, xzr",

            "1:",
            "ldr {val_m_b}, [{b}, {offset}]",
            "ldr {val_out}, [{out}, {offset}]",
            "mul {p_lo}, {val_m_b}, {a_i}",
            "umulh {p_hi}, {val_m_b}, {a_i}",
            "adds {p_lo}, {p_lo}, {carry}",
            "adc {p_hi}, {p_hi}, xzr",
            "adds {val_out}, {val_out}, {p_lo}",
            "adc {carry}, {p_hi}, xzr",
            "str {val_out}, [{out}, {offset}]",
            "add {offset}, {offset}, #8",
            "cmp {offset}, {len}, lsl #3",
            "b.lo 1b",
            "mov {a_i}, {carry}",             // save carry_b from Pass 1 into a_i

            // --- Pass 2: q = out[0] * m_inv ---
            "ldr {val_out}, [{out}]",
            "mul {q}, {val_out}, {m_inv}",

            // --- Pass 3: out = (out + q * m) >> 64 ---
            "mov {carry}, xzr",

            // Step 0 (j=0): compute q * m[0] + out[0], result is 0 mod 2^64, just keep carry
            "ldr {val_m_b}, [{m}]",
            "ldr {val_out}, [{out}]",
            "mul {p_lo}, {val_m_b}, {q}",
            "umulh {p_hi}, {val_m_b}, {q}",
            "adds {p_lo}, {p_lo}, {carry}",
            "adc {p_hi}, {p_hi}, xzr",
            "adds {val_out}, {val_out}, {p_lo}",
            "adc {carry}, {p_hi}, xzr",

            "mov {loops}, {len}",
            "sub {loops}, {loops}, #1",
            "cbz {loops}, 3f",            // if len == 1, skip

            "add {m_ptr}, {m}, #8",       // m_ptr = m + 8 (read m[1] first iter)
            "add {out_read}, {out}, #8",  // out_read = out + 8 (read out[1] first iter)
            "mov {out_write}, {out}",     // out_write = out (write out[0] first iter)

            "2:",                          // for j = 1 to len-1
            "ldr {val_m_b}, [{m_ptr}], #8",
            "ldr {val_out}, [{out_read}], #8",
            "mul {p_lo}, {val_m_b}, {q}",
            "umulh {p_hi}, {val_m_b}, {q}",
            "adds {p_lo}, {p_lo}, {carry}",
            "adc {p_hi}, {p_hi}, xzr",
            "adds {val_out}, {val_out}, {p_lo}",
            "adc {carry}, {p_hi}, xzr",
            "str {val_out}, [{out_write}], #8",
            "subs {loops}, {loops}, #1",
            "b.ne 2b",

            "3:",
            "mov {m_inv}, {carry}",       // return carry_m in m_inv reg

            out = in(reg) out,
            b = in(reg) b,
            m = in(reg) m,
            len = in(reg) len,
            a_i = inout(reg) a_i,         // outputs carry_b
            m_inv = inout(reg) m_inv,     // outputs carry_m
            carry = out(reg) _,
            offset = out(reg) _,
            m_ptr = out(reg) _,
            out_read = out(reg) _,
            out_write = out(reg) _,
            val_m_b = out(reg) _,
            val_out = out(reg) _,
            p_lo = out(reg) _,
            p_hi = out(reg) _,
            q = out(reg) _,
            loops = out(reg) _,
            options(nostack)
        );
        let (final_sum, final_carry) = a_i.overflowing_add(m_inv);
        *out.add(len.wrapping_sub(1)) = final_sum;
        Limb::from(final_carry)
    }
}
