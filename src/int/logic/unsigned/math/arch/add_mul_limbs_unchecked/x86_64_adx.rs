//! ADX/BMI2 x86-64 fused multiply-add limb kernel.

use core::arch::asm;

use super::Limb;

/// Accumulates `src * scalar` into `dst` and returns the high carry limb.
///
/// `adcxq` uses CF for destination additions; `adoxq` uses OF for high-product
/// additions. Eight limbs are processed per block, followed by a scalar tail.
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// Nonempty spans must cover `len` aligned, initialized limbs in disjoint live
/// allocations of at most `isize::MAX` bytes. The destination must be writable.
/// The executing CPU must support ADX and BMI2.
#[expect(
    clippy::inline_always,
    reason = "Keep both hardware flag recurrences in the selected multiplication caller"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let carry_hi: Limb;
    let chunks = len >> 3;
    let rem = len & 7;

    // SAFETY:
    // 1. `dst` covers len aligned, initialized, writable limbs.
    // 2. `src` is valid for reads of `len` `Limb` elements.
    // 3. Main-loop pointer offsets through `56` address its eight-limb chunk;
    //    advancing by `64` reaches the next chunk. The tail advances within
    //    the remaining `len & 7` limbs. All accesses therefore stay within
    //    the caller-provided `len * 8` byte spans.
    // 4. Memory spans are non-overlapping.
    // 5. Architecture dispatch establishes ADX and BMI2 availability. MULX
    //    preserves both flags; ADCX changes only CF and ADOX changes only OF.
    //    Product high limbs are at most Limb::MAX-1, so absorbing one OF bit
    //    cannot overflow. Each bounded DEC clears OF while preserving CF.
    // 6. RAX holds zero throughout carry propagation and receives the return
    //    value only after all source, destination, and counter inputs are consumed.
    unsafe {
        asm!(
            "xorl %ecx, %ecx",                           // rcx = 0, clears CF and OF
            "xorl %eax, %eax",                           // rax = 0 (zero register for carry absorption)
            "decq {chunks}",                             // Pre-decrement chunk counter
            "js 1f",                                     // If chunks < 0, jump to remainder (1f)

            // Main 8-way unrolled loop body
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

            "mulxq 32({src}), %r8, %r9",                 // (%r9:%r8) = scalar * src[4]
            "mulxq 40({src}), %r10, %r11",               // (%r11:%r10) = scalar * src[5]
            "adcxq 32({dst}), %r8",                      // %r8 = dst[4] + lo4 + CF
            "adoxq %rcx, %r8",                           // Consume hi3 and pending OF directly
            "movq %r8, 32({dst})",                       // Store updated dst[4]
            "adcxq 40({dst}), %r10",                     // %r10 = dst[5] + lo5 + CF
            "adoxq %r9, %r10",                           // %r10 += hi4 + OF
            "movq %r10, 40({dst})",                      // Store updated dst[5]

            "mulxq 48({src}), %r8, %r9",                 // (%r9:%r8) = scalar * src[6]
            "mulxq 56({src}), %r10, %rcx",               // (%rcx:%r10) = scalar * src[7]
            "adcxq 48({dst}), %r8",                      // %r8 = dst[6] + lo6 + CF
            "adoxq %r11, %r8",                           // %r8 += hi5 + OF
            "movq %r8, 48({dst})",                       // Store updated dst[6]
            "adcxq 56({dst}), %r10",                     // %r10 = dst[7] + lo7 + CF
            "adoxq %r9, %r10",                           // %r10 += hi6 + OF
            "movq %r10, 56({dst})",                      // Store updated dst[7]

            "adoxq %rax, %rcx",                          // Absorb pending OF into rcx (hi7)
            "leaq 64({src}), {src}",                     // Advance src pointer by 64 bytes
            "leaq 64({dst}), {dst}",                     // Advance dst pointer by 64 bytes
            "decq {chunks}",                             // Decrement chunks (preserves CF)
            "jns 2b",                                    // Repeat while chunks >= 0

            // Tail processing entry point (0 to 7 limbs remaining)
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
            // The final DEC takes a counter from zero to -1, clearing OF.
            // Only CF remains; RAX is still the zero register.
            "adcxq %rax, %rcx",                          // Flush pending CF into rcx
            "movq %rcx, %rax",                           // Return directly in the ABI result register

            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            in("rdx") scalar,
            out("rax") carry_hi,
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
