//! RISC-V 64-bit CIOS Montgomery reduction-step kernel.
//!
//! Implements Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
//! using 64×64→128-bit multipliers (`mul`/`mulhu`) and branchless `sltu` carry capture.

use core::arch::asm;

use super::Limb;

/// Compute one fused Coarsely Integrated Operand Scanning (CIOS) reduction step.
///
/// Computes:
///
/// ```text
///   (out[0..len] + a_i * b[0..len] + q * m[0..len]) / 2^64
/// ```
///
/// # Microarchitectural Strategy
///
/// CIOS interleaves multiplicand and modular reduction accumulation in a single loop,
/// tracking `carry_b` and `carry_m` branchlessly using `sltu` without flags.
///
/// # Safety
///
/// - `out` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `b` and `m` must point to readable buffers of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of all buffers.
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

    // SAFETY:
    // 1. `out` is valid for reads and writes of `len` 64-bit `Limb` elements.
    // 2. `b` and `m` are valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets remain within `len * 8` bytes.
    unsafe {
        asm!(
            // Step 0: Prime the reduction pipeline with limb 0
            "ld {out_limb}, 0({out})",                   // Load out[0]
            "ld {factor}, 0({b})",                       // Load b[0]
            "mul {low}, {factor}, {a_i}",                // Low 64 bits of b[0] * a_i
            "mulhu {high}, {factor}, {a_i}",             // High 64 bits of b[0] * a_i
            "add {low}, {low}, {out_limb}",              // low += out[0]
            "sltu {carry_bit0}, {low}, {out_limb}",      // Detect wrap
            "add {carry_b}, {high}, {carry_bit0}",       // carry_b = high + carry_bit0

            // Derive quotient multiplier q
            "mul {quotient}, {low}, {m_inv}",            // quotient = (low * m_inv) mod 2^64
            "ld {factor}, 0({m})",                       // Load m[0]
            "mul {mod_low}, {factor}, {quotient}",       // Low 64 bits of m[0] * q
            "mulhu {mod_high}, {factor}, {quotient}",    // High 64 bits of m[0] * q
            "add {mod_low}, {mod_low}, {low}",           // Low word cancelled to 0 mod 2^64
            "sltu {carry_bit0}, {mod_low}, {low}",       // Detect wrap
            "add {carry_m}, {mod_high}, {carry_bit0}",   // carry_m = mod_high + carry_bit0

            // Advance pointers to limb 1
            "addi {out}, {out}, 8",
            "addi {b}, {b}, 8",
            "addi {m}, {m}, 8",
            "addi {len}, {len}, -1",
            "beqz {len}, 2f",                            // If len == 1, skip main loop (2f)

            // Main reduction loop for j = 1 to len-1
            "1:",
            "ld {out_limb}, 0({out})",                   // Load out[j]
            "ld {factor}, 0({b})",                       // Load b[j]
            "mul {low}, {factor}, {a_i}",                // Low 64 bits of b[j] * a_i
            "mulhu {high}, {factor}, {a_i}",             // High 64 bits of b[j] * a_i
            "add {low}, {low}, {carry_b}",               // low += carry_b
            "sltu {carry_bit0}, {low}, {carry_b}",       // Detect wrap
            "add {high}, {high}, {carry_bit0}",          // high += carry_bit0
            "add {low}, {low}, {out_limb}",              // low += out[j]
            "sltu {carry_bit1}, {low}, {out_limb}",      // Detect wrap
            "add {carry_b}, {high}, {carry_bit1}",       // Update carry_b

            "ld {factor}, 0({m})",                       // Load m[j]
            "mul {mod_low}, {factor}, {quotient}",       // Low 64 bits of m[j] * q
            "mulhu {mod_high}, {factor}, {quotient}",    // High 64 bits of m[j] * q
            "add {mod_low}, {mod_low}, {carry_m}",       // mod_low += carry_m
            "sltu {carry_bit0}, {mod_low}, {carry_m}",   // Detect wrap
            "add {mod_high}, {mod_high}, {carry_bit0}",  // mod_high += carry_bit0
            "add {mod_low}, {mod_low}, {low}",           // mod_low += low
            "sltu {carry_bit1}, {mod_low}, {low}",       // Detect wrap
            "add {carry_m}, {mod_high}, {carry_bit1}",   // Update carry_m
            "sd {mod_low}, -8({out})",                   // Store shifted limb into out[j-1]

            "addi {out}, {out}, 8",                      // Advance pointers
            "addi {b}, {b}, 8",
            "addi {m}, {m}, 8",
            "addi {len}, {len}, -1",
            "bnez {len}, 1b",                            // Repeat while len != 0

            // Epilogue: Flush combined carries to out[len-1] and return top overflow
            "2:",
            "add {final_limb}, {carry_b}, {carry_m}",    // final_limb = carry_b + carry_m
            "sltu {overflow}, {final_limb}, {carry_b}",  // overflow = 1 if final addition wrapped
            "sd {final_limb}, -8({out})",                // Store out[len-1]

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
            carry_bit0 = out(reg) _,
            carry_bit1 = out(reg) _,
            mod_low = out(reg) _,
            mod_high = out(reg) _,
            final_limb = out(reg) _,
            options(nostack)
        );
    }

    overflow
}
