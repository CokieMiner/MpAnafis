//! ADX/BMI2 x86-64 fused multiply-add limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// Computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// # Microarchitectural Strategy
///
/// This kernel exploits Intel/AMD ADX and BMI2 hardware extensions:
/// - `mulxq` generates 64×64 → 128-bit unsigned products without modifying any CPU flags.
/// - Two independent carry chains operate concurrently in superscalar execution:
///   - `CF` (Carry Flag): Driven by `adcxq` to accumulate destination additions `dst[i] + lo_prod`.
///   - `OF` (Overflow Flag): Driven by `adoxq` to thread high-half carries `hi_prod` into the next limb.
///
/// The loop is 4-way unrolled (32 bytes per iteration), saturating out-of-order execution ports.
///
/// # Safety
///
/// - `dst` must point to a readable and writable buffer of at least `len` initialized limbs.
/// - `src` must point to a readable buffer of at least `len` initialized limbs.
/// - `src` and `dst` buffers must not overlap in memory (non-aliasing invariant).
/// - `len` must reflect the allocated capacity of both buffers.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance in multi-precision hot paths"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry_hi: Limb;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY:
    // 1. `dst` is valid for writes of `len` `Limb` elements.
    // 2. `src` is valid for reads of `len` `Limb` elements.
    // 3. Pointer offsets (`0`, `8`, `16`, `24`, `32`) remain within `len * 8` bytes.
    // 4. Memory spans are non-overlapping.
    unsafe {
        asm!(
            "xorl %ecx, %ecx",                           // rcx = 0, clears CF and OF
            "xorl %eax, %eax",                           // rax = 0 (zero register for carry absorption)
            "decq {chunks}",                             // Pre-decrement chunk counter
            "js 1f",                                     // If chunks < 0, jump to remainder (1f)

            // Main 4-way unrolled loop body
            "2:",                                        // Loop head label
            "mulxq 0({src}), %r8, %r9",                  // (%r9:%r8) = scalar * src[0]
            "mulxq 8({src}), %r10, %r11",                // (%r11:%r10) = scalar * src[1]
            "adcxq 0({dst}), %r8",                       // %r8 = dst[0] + lo0 + CF
            "adoxq %rcx, %r8",                           // %r8 += prev_hi + OF
            "movq %r8, 0({dst})",                        // Store updated dst[0]
            "adcxq 8({dst}), %r10",                      // %r10 = dst[1] + lo1 + CF
            "adoxq %r9, %r10",                           // %r10 += hi0 + OF
            "movq %r10, 8({dst})",                       // Store updated dst[1]

            "mulxq 16({src}), %r8, %r9",                 // (%r9:%r8) = scalar * src[2]
            "mulxq 24({src}), %r10, %rcx",               // (%rcx:%r10) = scalar * src[3]
            "adcxq 16({dst}), %r8",                      // %r8 = dst[2] + lo2 + CF
            "adoxq %r11, %r8",                           // %r8 += hi1 + OF
            "movq %r8, 16({dst})",                       // Store updated dst[2]
            "adcxq 24({dst}), %r10",                     // %r10 = dst[3] + lo3 + CF
            "adoxq %r9, %r10",                           // %r10 += hi2 + OF
            "movq %r10, 24({dst})",                      // Store updated dst[3]

            "adoxq %rax, %rcx",                          // Absorb pending OF into rcx (hi3)
            "leaq 32({src}), {src}",                     // Advance src pointer by 32 bytes
            "leaq 32({dst}), {dst}",                     // Advance dst pointer by 32 bytes
            "decq {chunks}",                             // Decrement chunks (preserves CF)
            "jns 2b",                                    // Repeat while chunks >= 0

            // Tail processing entry point (0 to 3 limbs remaining)
            "1:",                                        // Tail entry label
            "decq {rem}",                                // Pre-decrement remainder counter
            "js 4f",                                     // If rem < 0, skip to finish (4f)

            // 1-limb unrolled tail loop
            "3:",                                        // Tail loop label
            "mulxq 0({src}), %r8, %r9",                  // (%r9:%r8) = scalar * src[0]
            "adcxq 0({dst}), %r8",                       // %r8 = dst[0] + lo + CF
            "adoxq %rcx, %r8",                           // %r8 += running_hi + OF
            "movq %r8, 0({dst})",                        // Store updated dst[0]
            "movq %r9, %rcx",                            // Move current high product into rcx
            "adoxq %rax, %rcx",                          // Absorb pending OF into rcx
            "leaq 8({src}), {src}",                      // Advance src pointer (+8)
            "leaq 8({dst}), {dst}",                      // Advance dst pointer (+8)
            "decq {rem}",                                // Decrement remainder counter
            "jns 3b",                                    // Repeat while rem >= 0

            // Final carry consolidation
            "4:",                                        // Finish label
            "movq $0, %r8",                              // Scratch zero register
            "adcxq %r8, %rcx",                           // Flush pending CF into rcx
            "adoxq %r8, %rcx",                           // Flush pending OF into rcx
            "movq %rcx, {carry_hi}",                     // Store final 64-bit carry out

            carry_hi = out(reg) carry_hi,
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
            options(nostack, att_syntax)
        );
    }
    carry_hi
}
