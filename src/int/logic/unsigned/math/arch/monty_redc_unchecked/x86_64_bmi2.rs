//! BMI2 x86-64 Montgomery reduction step kernel.
//!
//! Implements Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
//! using `x86_64` BMI2 (`mulx`) instructions without ADX.

use core::arch::asm;

use super::Limb;

/// Fused Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
/// using `x86_64` BMI2 (`mulx`) instructions without ADX.
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
/// # Microarchitectural Strategy
///
/// `mulx` generates flag-free $64 \times 64 \to 128$-bit products into register pairs,
/// which are accumulated with standard `addq`/`adcq` carry propagation.
///
/// # Safety
///
/// - `out` must be valid for reads and writes of `len` 64-bit limbs.
/// - `b` and `m` must be valid for reads of `len` 64-bit limbs.
/// - `out` must not overlap `b` or `m` in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of all buffers.
#[allow(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "Critical for peak assembly performance; unrolled 4-way loops require contiguous blocks"
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

    // SAFETY:
    // 1. `out` is valid for reads and writes of `len` 64-bit `Limb` elements.
    // 2. `b` and `m` are valid for reads of `len` 64-bit `Limb` elements.
    // 3. Memory spans are non-overlapping for output writes.
    unsafe {
        asm!(
            // --- Pass 1: out = out + a_i * b ---
            "movq {len}, %r12",                          // %r12 = len
            "shrq $2, %r12",                             // %r12 = chunks = len / 4
            "movq {len}, %r13",                          // %r13 = len
            "andq $3, %r13",                             // %r13 = rem = len % 4

            "xorl %r10d, %r10d",                         // Zero %r10 (carry_b = 0)
            "movq {a_i}, %rdx",                          // %rdx = a_i (operand for mulx)

            // Main 4-way unrolled loop for Pass 1
            "decq %r12",                                 // Decrement chunk counter
            "js 11f",                                    // If chunks == 0, skip to remainder (11f)

            "10:",
            // [Limb 0]
            "mulxq 0({b}, {j}, 8), %r8, %r9",            // %r9:%r8 = b[j] * a_i
            "movq 0({out}, {j}, 8), %r11",               // Load out[j]
            "addq %r10, %r11",                           // %r11 += carry_b
            "adcq $0, %r9",                              // %r9 += CF
            "addq %r8, %r11",                            // %r11 += %r8
            "adcq $0, %r9",                              // %r9 += CF
            "movq %r11, 0({out}, {j}, 8)",               // Store updated out[j]

            // [Limb 1]
            "mulxq 8({b}, {j}, 8), %r8, %r10",           // %r10:%r8 = b[j+1] * a_i
            "movq 8({out}, {j}, 8), %r11",               // Load out[j+1]
            "addq %r9, %r11",                            // %r11 += carry from limb 0
            "adcq $0, %r10",                             // %r10 += CF
            "addq %r8, %r11",                            // %r11 += %r8
            "adcq $0, %r10",                             // %r10 += CF
            "movq %r11, 8({out}, {j}, 8)",               // Store updated out[j+1]

            // [Limb 2]
            "mulxq 16({b}, {j}, 8), %r8, %r9",           // %r9:%r8 = b[j+2] * a_i
            "movq 16({out}, {j}, 8), %r11",              // Load out[j+2]
            "addq %r10, %r11",                           // %r11 += carry from limb 1
            "adcq $0, %r9",                              // %r9 += CF
            "addq %r8, %r11",                            // %r11 += %r8
            "adcq $0, %r9",                              // %r9 += CF
            "movq %r11, 16({out}, {j}, 8)",              // Store updated out[j+2]

            // [Limb 3]
            "mulxq 24({b}, {j}, 8), %r8, %r10",          // %r10:%r8 = b[j+3] * a_i
            "movq 24({out}, {j}, 8), %r11",              // Load out[j+3]
            "addq %r9, %r11",                            // %r11 += carry from limb 2
            "adcq $0, %r10",                             // %r10 += CF
            "addq %r8, %r11",                            // %r11 += %r8
            "adcq $0, %r10",                             // %r10 += CF
            "movq %r11, 24({out}, {j}, 8)",              // Store updated out[j+3]

            "leaq 4({j}), {j}",                          // j += 4
            "decq %r12",                                 // Decrement chunk counter
            "jns 10b",                                   // Repeat while chunks >= 0

            // Remainder limbs for Pass 1 (0 to 3 limbs)
            "11:",
            "decq %r13",                                 // Decrement remainder counter
            "js 13f",

            "12:",
            "mulxq ({b}, {j}, 8), %r8, %r9",             // %r9:%r8 = b[j] * a_i
            "movq ({out}, {j}, 8), %r11",                // Load out[j]
            "addq %r10, %r11",                           // %r11 += carry
            "adcq $0, %r9",                              // %r9 += CF
            "addq %r8, %r11",                            // %r11 += %r8
            "adcq $0, %r9",                              // %r9 += CF
            "movq %r11, ({out}, {j}, 8)",                // Store updated out[j]
            "movq %r9, %r10",                            // %r10 = carry forward
            "leaq 1({j}), {j}",                          // j++
            "decq %r13",                                 // rem--
            "jns 12b",

            "13:",
            "movq %r10, {a_i}",                          // Return carry_b in a_i register

            // --- Pass 2: Compute q = out[0] * m_inv ---
            "movq ({out}), %r11",                        // Load updated out[0]
            "imulq {m_inv}, %r11",                       // %r11 = q = (out[0] * m_inv) mod 2^64
            "movq %r11, %rdx",                           // %rdx = q (multiplier operand for Pass 3)

            // Prepare Pass 3 counts
            "leaq -1({j}), %r13",                        // %r13 = original len - 1
            "movq %r13, %r12",                           // %r12 = len - 1
            "shrq $2, %r12",                             // chunks for Pass 3
            "andq $3, %r13",                             // rem for Pass 3

            // --- Pass 3 Step 0: Compute q * m[0] + out[0] ---
            "mulxq ({m}), %r8, %r10",                    // %r10:%r8 = m[0] * q
            "movq ({out}), %r11",                        // Load out[0]
            "addq %r8, %r11",                            // Cancel low word to 0 mod 2^64
            "adcq $0, %r10",                             // %r10 = hi_m0 + CF (carry into step 1)

            "movq $1, {j}",                              // j = 1

            // --- Pass 3 Loop: j from 1 to len - 1, unrolled 4-way ---
            "decq %r12",                                 // Decrement chunk counter
            "js 21f",

            "20:",
            // [Limb 0]
            "mulxq 0({m}, {j}, 8), %r8, %r9",            // %r9:%r8 = m[j] * q
            "movq 0({out}, {j}, 8), %r11",               // Load out[j]
            "addq %r10, %r11",                           // %r11 += carry_prev
            "adcq $0, %r9",                              // %r9 += CF
            "addq %r8, %r11",                            // %r11 += %r8
            "adcq $0, %r9",                              // %r9 += CF
            "movq %r11, -8({out}, {j}, 8)",              // Store shifted limb into out[j-1]

            // [Limb 1]
            "mulxq 8({m}, {j}, 8), %r8, %r10",           // %r10:%r8 = m[j+1] * q
            "movq 8({out}, {j}, 8), %r11",               // Load out[j+1]
            "addq %r9, %r11",                            // %r11 += carry from limb 0
            "adcq $0, %r10",                             // %r10 += CF
            "addq %r8, %r11",                            // %r11 += %r8
            "adcq $0, %r10",                             // %r10 += CF
            "movq %r11, 0({out}, {j}, 8)",               // Store shifted limb into out[j]

            // [Limb 2]
            "mulxq 16({m}, {j}, 8), %r8, %r9",           // %r9:%r8 = m[j+2] * q
            "movq 16({out}, {j}, 8), %r11",              // Load out[j+2]
            "addq %r10, %r11",                           // %r11 += carry from limb 1
            "adcq $0, %r9",                              // %r9 += CF
            "addq %r8, %r11",                            // %r11 += %r8
            "adcq $0, %r9",                              // %r9 += CF
            "movq %r11, 8({out}, {j}, 8)",               // Store shifted limb into out[j+1]

            // [Limb 3]
            "mulxq 24({m}, {j}, 8), %r8, %r10",          // %r10:%r8 = m[j+3] * q
            "movq 24({out}, {j}, 8), %r11",              // Load out[j+3]
            "addq %r9, %r11",                            // %r11 += carry from limb 2
            "adcq $0, %r10",                             // %r10 += CF
            "addq %r8, %r11",                            // %r11 += %r8
            "adcq $0, %r10",                             // %r10 += CF
            "movq %r11, 16({out}, {j}, 8)",              // Store shifted limb into out[j+2]

            "leaq 4({j}), {j}",                          // j += 4
            "decq %r12",                                 // Decrement chunk counter
            "jns 20b",

            // Remainder limbs for Pass 3 (0 to 3 limbs)
            "21:",
            "decq %r13",                                 // Decrement remainder counter
            "js 23f",

            "22:",
            "mulxq ({m}, {j}, 8), %r8, %r9",             // %r9:%r8 = m[j] * q
            "movq ({out}, {j}, 8), %r11",                // Load out[j]
            "addq %r10, %r11",                           // %r11 += carry_prev
            "adcq $0, %r9",                              // %r9 += CF
            "addq %r8, %r11",                            // %r11 += %r8
            "adcq $0, %r9",                              // %r9 += CF
            "movq %r11, -8({out}, {j}, 8)",              // Store shifted limb into out[j-1]
            "movq %r9, %r10",                            // %r10 = carry_prev
            "leaq 1({j}), {j}",                          // j++
            "decq %r13",                                 // rem--
            "jns 22b",

            "23:",
            "movq %r10, {m_inv}",                        // Return carry_m in m_inv register

            out = in(reg) out,
            b = in(reg) b,
            m = in(reg) m,
            len = in(reg) len,
            j = inout(reg) j,
            a_i = inout(reg) a_i,                        // Outputs carry_b
            m_inv = inout(reg) m_inv,                    // Outputs carry_m
            out("rax") _,
            out("rdx") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            out("r12") _,
            out("r13") _,
            options(nostack, att_syntax)
        );
        let (final_sum, final_carry) = a_i.overflowing_add(m_inv);
        *out.add(j.wrapping_sub(1)) = final_sum;
        Limb::from(final_carry)
    }
}
