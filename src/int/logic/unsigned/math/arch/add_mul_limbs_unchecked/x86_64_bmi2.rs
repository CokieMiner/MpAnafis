//! BMI2-only `mulx` fused multiply-add limb kernel for `x86_64` (without ADX).
//!
//! Uses `mulxq` (BMI2) for flag-free multiplication and standard `addq`/`adcq`
//! for single-chain carry propagation. Architecture dispatch selects this backend
//! when BMI2 is available and the ADX backend is not selected.

use core::arch::asm;

use super::Limb;

/// Accumulates `src * scalar` into `dst` and returns the high carry limb.
///
/// `mulxq` preserves flags while `addq`/`adcq` propagate the carry.
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// Nonempty spans must cover `len` aligned, initialized limbs in disjoint live
/// allocations of at most `isize::MAX` bytes. The destination must be writable.
/// The executing CPU must support BMI2.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "Both assembly paths remain inline in one entry point to preserve short-row register allocation"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let carry: Limb;
    if len < 4 {
        // SAFETY: the caller supplies initialized disjoint spans and BMI2
        // availability. The zero-length branch precedes every access; each
        // iteration consumes one limb before advancing within the supplied spans.
        // With B = 2^64 and carry <= scalar, the full limb sum is at most B^2 - 1,
        // so its high word fits in R8. R9 holds the bounded remaining count.
        unsafe {
            asm!(
                "xorl %ecx, %ecx",
                "testq %r9, %r9",
                "jz 3f",
                "2:",
                "mulxq ({src}), %rax, %r8",
                "addq %rcx, %rax",
                "adcq $0, %r8",
                "addq ({dst}), %rax",
                "adcq $0, %r8",
                "movq %rax, ({dst})",
                "movq %r8, %rcx",
                "leaq 8({src}), {src}",
                "leaq 8({dst}), {dst}",
                "decq %r9",
                "jnz 2b",
                "3:",
                "movq %rcx, %rax",
                dst = inout(reg) dst => _,
                src = inout(reg) src => _,
                inout("r9") len => _,
                in("rdx") scalar,
                out("rax") carry,
                out("rcx") _,
                out("r8") _,
                options(nostack, att_syntax),
            );
        }
        return carry;
    }
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: the caller supplies initialized disjoint spans of len limbs and
    // BMI2 availability. Since len >= 4, the first four-limb chunk exists;
    // every main iteration consumes four limbs and the tail consumes len & 3.
    // Pointer advances reach at most the ends of the supplied spans. R11 holds
    // the preceding pair's carry and is consumed before MULX overwrites it.
    // For B = 2^64 and incoming carry <= scalar, each limb sum satisfies
    // src * scalar + dst + carry <= B * scalar + B - 1 <= B^2 - 1.
    // Thus both carry additions fit the product high word. RAX receives the
    // return carry only after its last low product has been stored.
    unsafe {
        asm!(
            "xorl %r11d, %r11d",                         // Zero incoming carry
            "decq {chunks}",                             // Pre-decrement chunk counter

            // Main 4-way unrolled loop body
            "2:",                                        // Loop head label
            "movq 0({dst}), %rcx",                       // rcx = dst[0]
            "mulxq 0({src}), %rax, %r8",                 // (%r8:%rax) = scalar * src[0]
            "addq %r11, %rax",                           // Consume incoming carry before the next MULX
            "adcq $0, %r8",                              // r8 += CF
            "mulxq 8({src}), %r10, %r11",                // (%r11:%r10) = scalar * src[1]
            "addq %rax, %rcx",                           // rcx += rax, set CF
            "adcq $0, %r8",                              // r8 += CF
            "addq %r8, %r10",                            // r10 += r8 (hi0), set CF
            "movq %rcx, 0({dst})",                       // Store updated dst[0]
            "adcq $0, %r11",                             // r11 += CF
            "movq 8({dst}), %rcx",                       // rcx = dst[1]
            "addq %r10, %rcx",                           // rcx += r10, set CF
            "adcq $0, %r11",                             // r11 += CF
            "movq %rcx, 8({dst})",                       // Store updated dst[1]

            "movq 16({dst}), %rcx",                      // rcx = dst[2]
            "mulxq 16({src}), %rax, %r8",                // (%r8:%rax) = scalar * src[2]
            "addq %r11, %rax",                           // Consume incoming carry before the next MULX
            "mulxq 24({src}), %r10, %r11",               // (%r11:%r10) = scalar * src[3]
            "adcq $0, %r8",                              // r8 += CF
            "addq %rax, %rcx",                           // rcx += rax, set CF
            "adcq $0, %r8",                              // r8 += CF
            "movq %rcx, 16({dst})",                      // Store updated dst[2]
            "movq 24({dst}), %rcx",                      // rcx = dst[3]
            "addq %r8, %r10",                            // r10 += r8 (hi2), set CF
            "adcq $0, %r11",                             // r11 += CF
            "addq %r10, %rcx",                           // rcx += r10, set CF
            "adcq $0, %r11",                             // r11 += CF
            "movq %rcx, 24({dst})",                      // Store updated dst[3]
            "leaq 32({dst}), {dst}",                     // Advance dst pointer by 32 bytes

            "decq {chunks}",                             // Decrement chunks
            "leaq 32({src}), {src}",                     // Advance src pointer by 32 bytes
            "jns 2b",                                    // Repeat while chunks >= 0

            // Tail processing entry point (0 to 3 limbs remaining)
            "1:",                                        // Tail entry label
            "decq {rem}",                                // Pre-decrement remainder counter
            "js 4f",                                     // If rem < 0, skip to finish (4f)

            // 1-limb unrolled tail loop
            "3:",                                        // Tail loop label
            "mulxq 0({src}), %rax, %r8",                 // (%r8:%rax) = scalar * src[0]
            "addq %r11, %rax",                           // Add preceding chunk or tail carry
            "adcq $0, %r8",                              // r8 += CF
            "movq 0({dst}), %rcx",                       // rcx = dst[0]
            "addq %rax, %rcx",                           // rcx += rax, set CF
            "movq %rcx, 0({dst})",                       // Store updated dst[0]
            "adcq $0, %r8",                              // r8 += CF
            "movq %r8, %r11",                            // Update running carry
            "leaq 8({src}), {src}",                      // Advance src pointer (+8)
            "leaq 8({dst}), {dst}",                      // Advance dst pointer (+8)
            "decq {rem}",                                // Decrement remainder counter
            "jns 3b",                                    // Repeat while rem >= 0

            // Tail completion
            "4:",                                        // Completion label
            "movq %r11, %rax",                           // Return in the ABI result register

            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            in("rdx") scalar,
            out("rax") carry,
            out("rcx") _,
            out("r8") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
    carry
}
