//! ADX/BMI2 x86-64 Montgomery reduction step kernel.

use core::arch::asm;

use super::Limb;

/// Fused Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
/// using `x86_64` ADX (`mulx`, `adcx`, `adox`) instructions.
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
/// - `out` must not overlap `b` or `m`, even partially: the loop reads `b`
///   and `m` while it writes `out`, so any overlap is a data race. `b` and
///   `m` are read-only and may alias each other.
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
    let mut j: usize = 0;

    // SAFETY: caller guarantees out, b, and m have at least len elements.
    unsafe {
        asm!(
            // --- Pass 1: out = out + a_i * b ---
            "xorl %r10d, %r10d",
            "xorl %eax, %eax",
            "movq {a_i}, %rdx",

            ".p2align 4",
            "1:",
            "mulxq ({b}, {j}, 8), %r8, %r9",
            "movq ({out}, {j}, 8), %r11",
            "adcxq %r8, %r11",
            "adoxq %r10, %r11",
            "movq %r11, ({out}, {j}, 8)",
            "movq %r9, %r10",
            "adoxq %rax, %r10",           // absorb OF into r10 before decq
            "leaq 1({j}), {j}",           // j++ (does not touch ANY flags)
            "decq {len}",                 // len-- (does not touch CF)
            "jnz 1b",

            "movq $0, %rax",
            "adcxq %rax, %r10",
            "movq %r10, {a_i}",           // return carry_b in a_i reg

            // --- Step between passes: compute q = out[0] * m_inv ---
            // Note: at this point, j == original len.
            "movq ({out}), %r11",
            "imulq {m_inv}, %r11",        // r11 = q
            "movq %r11, %rdx",            // rdx = q for mulx in Pass 2

            // --- Pass 2 step 0: compute q * m[0] + out[0] ---
            "leaq -1({j}), {len}",        // len = original len - 1
            "testq {len}, {len}",         // sets ZF if len == 1, and clears CF and OF to 0!

            "mulxq ({m}), %r8, %r10",     // r8 = lo_m0, r10 = hi_m0 (modifies zero flags)
            "movq ({out}), %r11",         // modifies zero flags
            "adcxq %r8, %r11",            // r11 += lo_m0 + CF(0). Sets CF into column 1! (modifies only CF)

            "jz 3f",                      // if original len == 1, skip loop (uses ZF from testq)

            "movq $1, {j}",               // j = 1

            // --- Pass 2 loop: j from 1 to len - 1, storing to out[j - 1] ---
            ".p2align 4",
            "2:",
            "mulxq ({m}, {j}, 8), %r8, %r9",
            "movq ({out}, {j}, 8), %r11",
            "adcxq %r8, %r11",            // add lo_m + CF_prev
            "adoxq %r10, %r11",           // add hi_m_prev + OF_prev(0)
            "movq %r9, %r10",             // hi_m_prev = hi_m
            "movq %r11, -8({out}, {j}, 8)", // store shifted to out[j - 1]
            "adoxq %rax, %r10",           // absorb OF into r10 before decq
            "leaq 1({j}), {j}",           // j++ (does not touch ANY flags)
            "decq {len}",                 // len-- (does not touch CF)
            "jnz 2b",

            "3:",
            "movq $0, %rax",
            "adcxq %rax, %r10",
            "movq %r10, {m_inv}",         // return carry_m in m_inv reg

            out = in(reg) out,
            b = in(reg) b,
            m = in(reg) m,
            len = inout(reg) len => _,
            j = inout(reg) j,
            a_i = inout(reg) a_i,         // outputs carry_b
            m_inv = inout(reg) m_inv,     // outputs carry_m
            out("rax") _,
            out("rdx") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
        let (final_sum, final_carry) = a_i.overflowing_add(m_inv);
        *out.add(j.wrapping_sub(1)) = final_sum;
        Limb::from(final_carry)
    }
}
