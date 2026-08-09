//! RISC-V 64-bit CIOS Montgomery reduction-step kernel.

use core::arch::asm;

use super::Limb;

/// Compute one fused Coarsely Integrated Operand Scanning reduction step.
///
/// # Safety
///
/// `out`, `b`, and `m` must each be valid for `len` limbs. The modulus must be
/// odd and `m_inv` must equal `-m[0]^-1 mod B`.
#[allow(
    clippy::inline_always,
    reason = "The CIOS reduction step is the inner loop of Montgomery multiplication"
)]
#[inline(always)]
pub unsafe fn monty_redc_step_unchecked(
    out: *mut Limb,
    b: *const Limb,
    m: *const Limb,
    len: usize,
    a_i: Limb,
    m_inv: Limb,
) -> Limb {
    if len == 0 {
        return 0;
    }

    let overflow: Limb;

    // For each product, `mulhu` is its exact high limb and each `sltu` is one
    // carry bit. Thus carry_b and carry_m remain below B. The Montgomery
    // inverse makes the first combined low limb exactly zero, so writing limb
    // j to position j-1 implements the exact division by B without a copy.
    // SAFETY: the caller-provided spans cover all loads. Step zero advances all
    // pointers once; the loop then processes the remaining len-1 limbs, writes
    // out[0..len-1], and the epilogue writes out[len-1].
    unsafe {
        asm!(
            "ld {out_limb}, 0({out})",
            "ld {factor}, 0({b})",
            "mul {low}, {factor}, {a_i}",
            "mulhu {high}, {factor}, {a_i}",
            "add {low}, {low}, {out_limb}",
            "sltu {carry_bit0}, {low}, {out_limb}",
            "add {carry_b}, {high}, {carry_bit0}",

            "mul {quotient}, {low}, {m_inv}",
            "ld {factor}, 0({m})",
            "mul {mod_low}, {factor}, {quotient}",
            "mulhu {mod_high}, {factor}, {quotient}",
            "add {mod_low}, {mod_low}, {low}",
            "sltu {carry_bit0}, {mod_low}, {low}",
            "add {carry_m}, {mod_high}, {carry_bit0}",

            "addi {out}, {out}, 8",
            "addi {b}, {b}, 8",
            "addi {m}, {m}, 8",
            "addi {len}, {len}, -1",
            "beqz {len}, 2f",

            "1:",
            "ld {out_limb}, 0({out})",
            "ld {factor}, 0({b})",
            "mul {low}, {factor}, {a_i}",
            "mulhu {high}, {factor}, {a_i}",
            "add {low}, {low}, {carry_b}",
            "sltu {carry_bit0}, {low}, {carry_b}",
            "add {high}, {high}, {carry_bit0}",
            "add {low}, {low}, {out_limb}",
            "sltu {carry_bit1}, {low}, {out_limb}",
            "add {carry_b}, {high}, {carry_bit1}",

            "ld {factor}, 0({m})",
            "mul {mod_low}, {factor}, {quotient}",
            "mulhu {mod_high}, {factor}, {quotient}",
            "add {mod_low}, {mod_low}, {carry_m}",
            "sltu {carry_bit0}, {mod_low}, {carry_m}",
            "add {mod_high}, {mod_high}, {carry_bit0}",
            "add {mod_low}, {mod_low}, {low}",
            "sltu {carry_bit1}, {mod_low}, {low}",
            "add {carry_m}, {mod_high}, {carry_bit1}",
            "sd {mod_low}, -8({out})",

            "addi {out}, {out}, 8",
            "addi {b}, {b}, 8",
            "addi {m}, {m}, 8",
            "addi {len}, {len}, -1",
            "bnez {len}, 1b",

            "2:",
            "add {final_limb}, {carry_b}, {carry_m}",
            "sltu {overflow}, {final_limb}, {carry_b}",
            "sd {final_limb}, -8({out})",

            out = inout(reg) out => _,
            b = inout(reg) b => _,
            m = inout(reg) m => _,
            len = inout(reg) len => _,
            a_i = in(reg) a_i,
            m_inv = in(reg) m_inv,
            overflow = out(reg) overflow,
            quotient = out(reg) _,
            carry_b = out(reg) _,
            carry_m = out(reg) _,
            out_limb = out(reg) _,
            factor = out(reg) _,
            low = out(reg) _,
            high = out(reg) _,
            mod_low = out(reg) _,
            mod_high = out(reg) _,
            carry_bit0 = out(reg) _,
            carry_bit1 = out(reg) _,
            final_limb = out(reg) _,
            options(nostack)
        );
    }
    overflow
}
