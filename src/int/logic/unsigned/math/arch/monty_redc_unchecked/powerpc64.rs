//! `PowerPC64` Montgomery reduction step kernel.
//!
//! Implements Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
//! using `PowerPC64` inline assembly (`mulld`, `mulhdu`, `addc`, `addze`).

use core::arch::asm;

use super::Limb;

/// Fused Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
/// using `PowerPC64` inline assembly (`mulld`, `mulhdu`, `addc`, `addze`).
///
/// For step `i`, this computes:
///
/// ```text
///   (out[0..len] + a_i * b[0..len] + q * m[0..len]) / 2^64
/// ```
///
/// where `q = ((out[0] + a_i * b[0]) * m_inv) mod 2^64`.
///
/// Stores the shifted result into `out[0..len-1]`, stores the combined low carry into
/// `out[len-1]`, and returns the top overflow carry (either 0 or 1).
///
/// `PowerPC64` executes 64x64->128-bit multiplications with `mulld`/`mulhdu`, chains carries
/// via `addc`/`addze` through `XER[CA]`. `bdnz` decrements CTR and branches on a nonzero count.
///
/// # Safety
///
/// - For nonzero `len`, all pointers must cover `len` aligned, initialized
///   limbs within `isize::MAX` bytes; `out` must be writable.
/// - `out` must be disjoint from `b` and `m`; the two inputs may overlap.
/// - `m` must be odd and `m_inv * m[0] = -1 mod 2^64`.
#[expect(
    clippy::inline_always,
    reason = "Inlining keeps the reduction step inside Montgomery multiplication"
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

    // SAFETY: aligned initialized spans cover len limbs, and writable out is
    // disjoint from both inputs. CTR bounds the first pass by len and the
    // shifted pass by len-1; each shifted store follows its source load.
    // Products plus two limbs are <= B^2-1, so high carries fit one limb.
    // The inverse cancels the discarded low word. The byte-span bound makes
    // offsets exact; modified general registers, CTR, XER, and CR0 are outputs.
    unsafe {
        asm!(
            // --- Pass 1: out = out + a_i * b ---
            "li {carry}, 0",                             // carry = 0
            "li {offset}, 0",                            // offset = 0
            "mtctr {len}",                               // Load len into hardware CTR register

            ".p2align 4",
            "1:",
            "ldx {val_m_b}, {b}, {offset}",              // Load b[j]
            "ldx {val_out}, {out}, {offset}",            // Load out[j]
            "mulld {p_lo}, {val_m_b}, {a_i}",            // Low 64 bits of b[j] * a_i
            "mulhdu {p_hi}, {val_m_b}, {a_i}",           // High 64 bits of b[j] * a_i
            "addc {p_lo}, {p_lo}, {carry}",              // p_lo += carry, set CA bit in XER
            "addze {p_hi}, {p_hi}",                      // p_hi += CA bit
            "addc {val_out}, {val_out}, {p_lo}",         // out[j] += p_lo, set CA bit
            "addze {carry}, {p_hi}",                     // carry = p_hi + CA bit
            "stdx {val_out}, {out}, {offset}",           // Store updated out[j]
            "addi {offset}, {offset}, 8",                // Advance offset by 8 bytes
            "bdnz 1b",                                   // Decrement CTR and branch if != 0

            "mr {a_i}, {carry}",                         // Return carry_b in a_i register

            // --- Pass 2: q = out[0] * m_inv ---
            "ld {val_out}, 0({out})",                    // Load out[0]
            "mulld {q}, {val_out}, {m_inv}",             // q = (out[0] * m_inv) mod 2^64

            // --- Pass 3: out = (out + q * m) >> 64 ---
            "li {carry}, 0",                             // Reset carry for reduction pass
            "li {offset}, 0",                            // Reset offset

            // Step 0 (j=0): compute q * m[0] + out[0], result is 0 mod 2^64, capture carry
            "ldx {val_m_b}, {m}, {offset}",              // Load m[0]
            "ldx {val_out}, {out}, {offset}",            // Load out[0]
            "mulld {p_lo}, {val_m_b}, {q}",              // Low 64 bits of m[0] * q
            "mulhdu {p_hi}, {val_m_b}, {q}",             // High 64 bits of m[0] * q
            "addc {p_lo}, {p_lo}, {carry}",              // p_lo += carry, set CA bit
            "addze {p_hi}, {p_hi}",                      // p_hi += CA bit
            "addc {val_out}, {val_out}, {p_lo}",         // Low word is 0 (discarded by shift)
            "addze {carry}, {p_hi}",                     // carry = p_hi + CA bit
            "addi {offset}, {offset}, 8",                // offset = 8
            "subi {offset_sub}, {offset}, 8",            // offset_sub = 0 (for writing out[0])

            "cmpdi {len}, 1",                            // Check if len == 1
            "beq 3f",                                    // If len == 1, skip reduction loop (3f)

            "subi {loops}, {len}, 1",                    // loops = len - 1
            "mtctr {loops}",                             // Load loops into CTR

            ".p2align 4",
            "2:",                                        // Loop for j = 1 to len-1
            "ldx {val_m_b}, {m}, {offset}",              // Load m[j]
            "ldx {val_out}, {out}, {offset}",            // Load out[j]
            "mulld {p_lo}, {val_m_b}, {q}",              // Low 64 bits of m[j] * q
            "mulhdu {p_hi}, {val_m_b}, {q}",             // High 64 bits of m[j] * q
            "addc {p_lo}, {p_lo}, {carry}",              // p_lo += carry, set CA bit
            "addze {p_hi}, {p_hi}",                      // p_hi += CA bit
            "addc {val_out}, {val_out}, {p_lo}",         // out[j] += p_lo, set CA bit
            "addze {carry}, {p_hi}",                     // carry = p_hi + CA bit
            "stdx {val_out}, {out}, {offset_sub}",       // Store shifted limb into out[j-1]
            "addi {offset}, {offset}, 8",                // Advance read offset
            "addi {offset_sub}, {offset_sub}, 8",        // Advance write offset
            "bdnz 2b",                                   // Decrement CTR and branch if != 0

            "3:",
            "mr {m_inv}, {carry}",                       // Return carry_m in m_inv register

            out = in(reg_nonzero) out,
            b = in(reg_nonzero) b,
            m = in(reg_nonzero) m,
            len = in(reg) len,
            a_i = inout(reg) a_i,                        // Outputs carry_b
            m_inv = inout(reg) m_inv,                    // Outputs carry_m
            carry = out(reg) _,
            offset = out(reg) _,
            offset_sub = out(reg) _,
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
    }

    let carry_b = a_i;
    let carry_m = m_inv;
    // SAFETY: len > 0 makes len-1 exact and selects the writable top limb.
    unsafe {
        let (final_sum, final_carry) = carry_b.overflowing_add(carry_m);
        *out.add(len.unchecked_sub(1)) = final_sum;
        Limb::from(final_carry)
    }
}
