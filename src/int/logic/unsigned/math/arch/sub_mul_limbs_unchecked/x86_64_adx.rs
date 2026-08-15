//! ADX/BMI2 x86-64 fused multiply-subtract limb kernel.
//!
//! Uses `mulxq` (BMI2) for flag-free multiplication, `adoxq` (ADX) along `OF` for
//! product row assembly, and `sbbq` along `CF` for destination subtraction.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// Computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// Multi-precision fused multiply-subtract requires tracking two simultaneous invariants:
/// 1. The multiplication carry chain (`sum(a_i * b) * 2^(64*i)`), accumulated in registers using `adoxq` over `OF`.
/// 2. The destination subtraction borrow chain (`dst[i] - term_i - borrow`), tracked using `sbbq` over `CF`.
///
/// The 4-way unrolled body first generates the 4 multi-precision product limbs in registers via `mulxq`
/// and `adoxq`, then subtracts the assembled 4 limbs from destination memory via consecutive `sbbq`
/// instructions, preserving the borrow bit across chunk iterations.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized 64-bit limbs.
/// - `src` must point to a readable buffer of at least `len` initialized 64-bit limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in multi-precision hot paths"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let carry_hi: Limb;
    let borrow_out: Limb;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` 64-bit `Limb` elements.
    // 2. `src` is valid for reads of `len` 64-bit `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`, `24`, `32`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "xorl %ecx, %ecx",                           // rcx = 0, clears CF and OF
            "xorl %eax, %eax",                           // rax = 0 (zero register for OF absorption)
            "decq {chunks}",                             // Pre-decrement chunk counter
            "js 1f",                                     // If chunks < 0, skip to remainder (1f)

            // Main 4-way unrolled loop body
            "2:",                                        // Loop head label
            // [Limb 0 Product & High Carry Assembly]
            "mulxq 0({src}), %r8, %r9",                  // %rdx * src[0] -> (%r9:%r8)
            "adoxq %rcx, %r8",                           // %r8 += rcx (running high carry) via OF

            // [Limb 1 Product & High Carry Assembly]
            "mulxq 8({src}), %r10, %r11",                // %rdx * src[1] -> (%r11:%r10)
            "adoxq %r9, %r10",                           // %r10 += r9 (limb 0 high product) via OF

            // [Limb 2 Product & High Carry Assembly]
            "mulxq 16({src}), %r9, %r12",                // %rdx * src[2] -> (%r12:%r9)
            "adoxq %r11, %r9",                           // %r9 += r11 (limb 1 high product) via OF

            // [Limb 3 Product & High Carry Assembly]
            "mulxq 24({src}), %r11, %rcx",               // %rdx * src[3] -> (%rcx:%r11)
            "adoxq %r12, %r11",                          // %r11 += r12 (limb 2 high product) via OF

            "adoxq %rax, %rcx",                          // Absorb remaining OF into rcx

            // [4-Limb Sequential Subtraction via CF Borrow Chain]
            "movq 0({dst}), %r12",                       // Load dst[0]
            "sbbq %r8, %r12",                            // dst[0] - r8 - CF -> updates CF (borrow)
            "movq %r12, 0({dst})",                       // Store updated dst[0]

            "movq 8({dst}), %r12",                       // Load dst[1]
            "sbbq %r10, %r12",                           // dst[1] - r10 - CF -> updates CF
            "movq %r12, 8({dst})",                       // Store updated dst[1]

            "movq 16({dst}), %r12",                      // Load dst[2]
            "sbbq %r9, %r12",                            // dst[2] - r9 - CF -> updates CF
            "movq %r12, 16({dst})",                      // Store updated dst[2]

            "movq 24({dst}), %r12",                      // Load dst[3]
            "sbbq %r11, %r12",                           // dst[3] - r11 - CF -> updates CF
            "movq %r12, 24({dst})",                      // Store updated dst[3]

            "leaq 32({src}), {src}",                     // Advance src pointer by 32 bytes
            "leaq 32({dst}), {dst}",                     // Advance dst pointer by 32 bytes
            "decq {chunks}",                             // Decrement chunk counter (preserves CF)
            "jns 2b",                                    // Repeat while chunks >= 0

            // Remainder processing entry point (0 to 3 limbs)
            "1:",                                        // Remainder entry label
            "decq {rem}",                                // Pre-decrement remainder counter
            "js 4f",                                     // If rem < 0, skip to finish (4f)

            // 1-limb unrolled tail loop
            "3:",                                        // Tail loop label
            "mulxq 0({src}), %r8, %r9",                  // %rdx * src[0] -> (%r9:%r8)
            "adoxq %rcx, %r8",                           // %r8 += rcx via OF
            "adoxq %rax, %r9",                           // %r9 += 0 + OF
            "movq 0({dst}), %r12",                       // Load dst[0]
            "sbbq %r8, %r12",                            // dst[0] - r8 - CF
            "movq %r12, 0({dst})",                       // Store updated dst[0]
            "movq %r9, %rcx",                            // Update running high carry
            "leaq 8({src}), {src}",                      // Advance src pointer (+8)
            "leaq 8({dst}), {dst}",                      // Advance dst pointer (+8)
            "decq {rem}",                                // Decrement remainder counter
            "jns 3b",                                    // Repeat while rem >= 0

            // Final carry and borrow extraction
            "4:",                                        // Finish label
            "sbbq %r12, %r12",                           // %r12 = -1 (if CF=1) or 0 (if CF=0)
            "negq %r12",                                 // %r12 = 1 (if borrow) or 0 (if no borrow)
            "movq %rcx, {carry_hi}",                     // Output high multiplication carry
            "movq %r12, {borrow_out}",                   // Output boolean borrow bit

            carry_hi = out(reg) carry_hi,
            borrow_out = out(reg) borrow_out,
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            in("rdx") scalar,
            out("rax") _,
            out("rcx") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            out("r12") _,
            options(nostack, att_syntax)
        );
    }
    (carry_hi, borrow_out)
}
