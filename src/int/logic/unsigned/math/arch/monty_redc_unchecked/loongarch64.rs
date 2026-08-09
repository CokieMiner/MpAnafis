//! `LoongArch64` CIOS Montgomery reduction-step kernel.

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

    // `mulh.du` supplies every exact high product limb and `sltu` supplies
    // each single carry bit, keeping both carry chains below B. Because m_inv
    // cancels the first low limb modulo B, the staggered stores are exactly the
    // CIOS division by B and lose no information.
    // SAFETY: the caller-provided spans cover all loads. Step zero advances all
    // pointers once; the loop then processes the remaining len-1 limbs, writes
    // out[0..len-1], and the epilogue writes out[len-1].
    unsafe {
        asm!(
            "ld.d {out_limb}, {out}, 0",
            "ld.d {factor}, {b}, 0",
            "mul.d {low}, {factor}, {a_i}",
            "mulh.du {high}, {factor}, {a_i}",
            "add.d {low}, {low}, {out_limb}",
            "sltu {carry_bit0}, {low}, {out_limb}",
            "add.d {carry_b}, {high}, {carry_bit0}",

            "mul.d {quotient}, {low}, {m_inv}",
            "ld.d {factor}, {m}, 0",
            "mul.d {mod_low}, {factor}, {quotient}",
            "mulh.du {mod_high}, {factor}, {quotient}",
            "add.d {mod_low}, {mod_low}, {low}",
            "sltu {carry_bit0}, {mod_low}, {low}",
            "add.d {carry_m}, {mod_high}, {carry_bit0}",

            "addi.d {out}, {out}, 8",
            "addi.d {b}, {b}, 8",
            "addi.d {m}, {m}, 8",
            "addi.d {len}, {len}, -1",
            "beqz {len}, 2f",

            "1:",
            "ld.d {out_limb}, {out}, 0",
            "ld.d {factor}, {b}, 0",
            "mul.d {low}, {factor}, {a_i}",
            "mulh.du {high}, {factor}, {a_i}",
            "add.d {low}, {low}, {carry_b}",
            "sltu {carry_bit0}, {low}, {carry_b}",
            "add.d {high}, {high}, {carry_bit0}",
            "add.d {low}, {low}, {out_limb}",
            "sltu {carry_bit1}, {low}, {out_limb}",
            "add.d {carry_b}, {high}, {carry_bit1}",

            "ld.d {factor}, {m}, 0",
            "mul.d {mod_low}, {factor}, {quotient}",
            "mulh.du {mod_high}, {factor}, {quotient}",
            "add.d {mod_low}, {mod_low}, {carry_m}",
            "sltu {carry_bit0}, {mod_low}, {carry_m}",
            "add.d {mod_high}, {mod_high}, {carry_bit0}",
            "add.d {mod_low}, {mod_low}, {low}",
            "sltu {carry_bit1}, {mod_low}, {low}",
            "add.d {carry_m}, {mod_high}, {carry_bit1}",
            "st.d {mod_low}, {out}, -8",

            "addi.d {out}, {out}, 8",
            "addi.d {b}, {b}, 8",
            "addi.d {m}, {m}, 8",
            "addi.d {len}, {len}, -1",
            "bnez {len}, 1b",

            "2:",
            "add.d {final_limb}, {carry_b}, {carry_m}",
            "sltu {overflow}, {final_limb}, {carry_b}",
            "st.d {final_limb}, {out}, -8",

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
