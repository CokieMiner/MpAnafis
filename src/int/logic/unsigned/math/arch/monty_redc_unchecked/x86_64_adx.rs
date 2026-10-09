//! ADX/BMI2 x86-64 Montgomery reduction step kernel.
//!
//! Implements Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
//! using `x86_64` ADX dual-carry pipelines (`mulx`, `adcx`, `adox`).

use core::arch::asm;

use super::Limb;

/// Fused Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction step
/// using `x86_64` ADX (`mulx`, `adcx`, `adox`) instructions.
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
/// ADX enables simultaneous progression of two independent carry chains:
/// `adcx` updates CF for low-part product sums, while `adox` updates OF for high-part product carries.
/// Each block processes four limbs before advancing the index.
///
/// # Safety
///
/// - For nonzero `len`, all pointers must cover `len` aligned, initialized
///   limbs within `isize::MAX` bytes; `out` must be writable.
/// - `out` must be disjoint from `b` and `m`; the two inputs may overlap.
/// - `m` must be odd and `m_inv * m[0] = -1 mod 2^64`.
/// - The CPU must support ADX and BMI2.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "Inlining keeps the reduction step at its caller; four-limb assembly blocks share live carry flags"
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

    // SAFETY: the caller supplies aligned initialized spans, writable out
    // disjoint from the inputs, and ADX/BMI2. The first pass covers len limbs;
    // the reduction pass loads j before storing j-1, then returns j=len.
    // Four-limb blocks and tails stay within those ranges. Products plus two
    // limbs are <= B^2-1. OF is absorbed before counter updates; the signed
    // counts cannot reach their overflow boundary under the byte-span bound.
    // The inverse cancels limb zero, and every modified register is an output.
    unsafe {
        asm!(
            // --- Pass 1: out = out + a_i * b ---
            "movq {len}, %r12",                          // %r12 = len
            "shrq $2, %r12",                             // %r12 = chunks = len / 4
            "andq $3, {len}",                            // {len} = rem = len % 4

            "xorl %r10d, %r10d",                         // Zero %r10 (previous high carry)
            "xorl %eax, %eax",                           // Zero %rax (clears CF and OF)
            "movq {a_i}, %rdx",                          // %rdx = a_i (multiplier operand for mulx)

            // Main 4-way unrolled loop for Pass 1
            "decq %r12",                                 // Decrement chunk counter
            "js 11f",                                    // If chunks == 0, skip to remainder (11f)

            "10:",
            // [Limb 0]
            "movq 0({out}, {j}, 8), %r11",               // Load out[j]
            "mulxq 0({b}, {j}, 8), %r8, %r9",            // %r9:%r8 = b[j] * a_i (flag-free product)
            "adcxq %r8, %r11",                           // %r11 += %r8 + CF (updates CF)
            "adoxq %r10, %r11",                          // %r11 += %r10 + OF (updates OF)
            "movq %r11, 0({out}, {j}, 8)",               // Store updated out[j]

            // [Limb 1]
            "movq 8({out}, {j}, 8), %r11",               // Load out[j+1]
            "mulxq 8({b}, {j}, 8), %r8, %r10",           // %r10:%r8 = b[j+1] * a_i
            "adcxq %r8, %r11",                           // %r11 += %r8 + CF
            "adoxq %r9, %r11",                           // %r11 += %r9 + OF
            "movq %r11, 8({out}, {j}, 8)",               // Store updated out[j+1]

            // [Limb 2]
            "movq 16({out}, {j}, 8), %r11",              // Load out[j+2]
            "mulxq 16({b}, {j}, 8), %r8, %r9",           // %r9:%r8 = b[j+2] * a_i
            "adcxq %r8, %r11",                           // %r11 += %r8 + CF
            "adoxq %r10, %r11",                          // %r11 += %r10 + OF
            "movq %r11, 16({out}, {j}, 8)",              // Store updated out[j+2]

            // [Limb 3]
            "movq 24({out}, {j}, 8), %r11",              // Load out[j+3]
            "mulxq 24({b}, {j}, 8), %r8, %r10",          // %r10:%r8 = b[j+3] * a_i
            "adcxq %r8, %r11",                           // %r11 += %r8 + CF
            "adoxq %r9, %r11",                           // %r11 += %r9 + OF
            "adoxq %rax, %r10",                          // Absorb OF into %r10
            "decq %r12",                                 // Decrement chunk counter (preserves CF)
            "movq %r11, 24({out}, {j}, 8)",              // Store updated out[j+3]
            "leaq 4({j}), {j}",                          // j += 4 without modifying flags
            "jns 10b",                                   // Repeat while chunks >= 0

            // Remainder limbs for Pass 1 (0 to 3 limbs)
            "11:",
            "decq {len}",                                // Decrement remainder counter
            "js 13f",                                    // If rem == 0, skip (13f)

            "12:",
            "mulxq ({b}, {j}, 8), %r8, %r9",             // %r9:%r8 = b[j] * a_i
            "movq ({out}, {j}, 8), %r11",                // Load out[j]
            "adcxq %r8, %r11",                           // %r11 += %r8 + CF
            "adoxq %r10, %r11",                          // %r11 += %r10 + OF
            "movq %r11, ({out}, {j}, 8)",                // Store updated out[j]
            "movq %r9, %r10",                            // %r10 = %r9 (carry forward high product)
            "adoxq %rax, %r10",                          // Absorb OF into %r10
            "leaq 1({j}), {j}",                          // j++ (preserves all flags)
            "decq {len}",                                // rem-- (preserves CF)
            "jns 12b",

            "13:",
            "movq $0, %rax",                             // Clear %rax
            "adcxq %rax, %r10",                          // Absorb remaining CF into %r10
            "movq %r10, {a_i}",                          // Return carry_b in a_i register

            // --- Pass 2: Compute q = out[0] * m_inv ---
            "movq ({out}), %r11",                        // Load updated out[0]
            "imulq {m_inv}, %r11",                       // %r11 = q = (out[0] * m_inv) mod 2^64
            "movq %r11, %rdx",                           // %rdx = q (multiplier operand for Pass 3 mulx)

            // Prepare Pass 3 loop counts
            "leaq -1({j}), {len}",                       // {len} = original len - 1
            "movq {len}, %r12",                          // %r12 = len - 1
            "shrq $2, %r12",                             // chunks for Pass 3
            "andq $3, {len}",                            // rem for Pass 3

            // --- Pass 3 Step 0: Compute q * m[0] + out[0] ---
            "xorl %eax, %eax",                           // Clears CF and OF
            "mulxq ({m}), %r8, %r10",                    // %r10:%r8 = m[0] * q
            "movq ({out}), %r11",                        // Load out[0]
            "adcxq %r8, %r11",                           // Cancel low word to 0 mod 2^64, sets CF

            "movq $1, {j}",                              // j = 1

            // --- Pass 3 Loop: j from 1 to len - 1, unrolled 4-way ---
            "decq %r12",                                 // Decrement chunk counter (preserves CF)
            "js 21f",

            "20:",
            // [Limb 0]
            "mulxq 0({m}, {j}, 8), %r8, %r9",            // %r9:%r8 = m[j] * q
            "movq 0({out}, {j}, 8), %r11",               // Load out[j]
            "adcxq %r8, %r11",                           // %r11 += %r8 + CF
            "adoxq %r10, %r11",                          // %r11 += %r10 + OF
            "movq %r11, -8({out}, {j}, 8)",              // Store shifted limb into out[j-1]

            // [Limb 1]
            "mulxq 8({m}, {j}, 8), %r8, %r10",           // %r10:%r8 = m[j+1] * q
            "movq 8({out}, {j}, 8), %r11",               // Load out[j+1]
            "adcxq %r8, %r11",                           // %r11 += %r8 + CF
            "adoxq %r9, %r11",                           // %r11 += %r9 + OF
            "movq %r11, 0({out}, {j}, 8)",               // Store shifted limb into out[j]

            // [Limb 2]
            "mulxq 16({m}, {j}, 8), %r8, %r9",           // %r9:%r8 = m[j+2] * q
            "movq 16({out}, {j}, 8), %r11",              // Load out[j+2]
            "adcxq %r8, %r11",                           // %r11 += %r8 + CF
            "adoxq %r10, %r11",                          // %r11 += %r10 + OF
            "movq %r11, 8({out}, {j}, 8)",               // Store shifted limb into out[j+1]

            // [Limb 3]
            "mulxq 24({m}, {j}, 8), %r8, %r10",          // %r10:%r8 = m[j+3] * q
            "movq 24({out}, {j}, 8), %r11",              // Load out[j+3]
            "adcxq %r8, %r11",                           // %r11 += %r8 + CF
            "adoxq %r9, %r11",                           // %r11 += %r9 + OF
            "movq %r11, 16({out}, {j}, 8)",              // Store shifted limb into out[j+2]

            "leaq 4({j}), {j}",                          // j += 4 (preserves all flags)
            "adoxq %rax, %r10",                          // Absorb OF into %r10
            "decq %r12",                                 // Decrement chunk counter
            "jns 20b",

            // Remainder limbs for Pass 3 (0 to 3 limbs)
            "21:",
            "decq {len}",                                // Decrement remainder counter
            "js 23f",

            "22:",
            "mulxq ({m}, {j}, 8), %r8, %r9",             // %r9:%r8 = m[j] * q
            "movq ({out}, {j}, 8), %r11",                // Load out[j]
            "adcxq %r8, %r11",                           // %r11 += %r8 + CF
            "adoxq %r10, %r11",                          // %r11 += %r10 + OF
            "movq %r9, %r10",                            // %r10 = %r9
            "movq %r11, -8({out}, {j}, 8)",              // Store shifted limb into out[j-1]
            "adoxq %rax, %r10",                          // Absorb OF into %r10
            "leaq 1({j}), {j}",                          // j++ (preserves all flags)
            "decq {len}",                                // rem-- (preserves CF)
            "jns 22b",

            "23:",
            "movq $0, %rax",                             // Clear %rax
            "adcxq %rax, %r10",                          // Absorb remaining CF into %r10
            "movq %r10, {m_inv}",                        // Return carry_m in m_inv register

            out = in(reg) out,
            b = in(reg) b,
            m = in(reg) m,
            len = inout(reg) len => _,
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
            options(nostack, att_syntax)
        );
        let (final_sum, final_carry) = a_i.overflowing_add(m_inv);
        *out.add(j.unchecked_sub(1)) = final_sum;
        Limb::from(final_carry)
    }
}
