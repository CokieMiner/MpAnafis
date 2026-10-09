//! ADX/BMI2 x86-64 fused multiply-subtract limb kernel.
//!
//! Uses `mulxq` (BMI2) for flag-free multiplication, `adoxq` (ADX) along `OF` for
//! product row assembly, and `sbbq` along `CF` for destination subtraction.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// For B = 2^64, `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
/// Each four-limb block assembles product digits along OF, then subtracts
/// those digits along CF.
///
/// # Safety
///
/// Both pointers must cover `len` aligned, initialized limbs in disjoint spans;
/// `dst` requires exclusive access. Each span's byte length must fit in `isize`.
/// Zero length permits null pointers. The CPU must support ADX and BMI2.
#[expect(
    clippy::inline_always,
    clippy::too_many_lines,
    reason = "short and unrolled assembly paths share one call boundary in the arithmetic hot path"
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    if len < 4 {
        if len == 0 {
            return (0, 0);
        }
        let carry: Limb;
        let borrow: Limb;
        // SAFETY: 1 <= len <= 3 bounds every load and store within the disjoint
        // aligned initialized spans. All registers are caller-saved, so short rows need
        // no stack frame. CF carries subtraction borrow and OF product carry.
        // dec only sees 1..=3, hence preserves CF and leaves OF clear after each
        // product's high half has absorbed its carry. The selected CPU has ADX/BMI2.
        unsafe {
            asm!(
                "mulxq (%rsi), %r9, %rax",
                "xorl %r8d, %r8d",
                "subq %r9, (%rdi)",
                "decq %r11",
                "jz 4f",
                "mulxq 8(%rsi), %r9, %r10",
                "adoxq %rax, %r9",
                "adoxq %r8, %r10",
                "sbbq %r9, 8(%rdi)",
                "decq %r11",
                "jnz 3f",
                "movq %r10, %rax",
                "jmp 4f",
                "3:",
                "mulxq 16(%rsi), %r9, %rax",
                "adoxq %r10, %r9",
                "adoxq %r8, %rax",
                "sbbq %r9, 16(%rdi)",
                "4:",
                "sbbq %rdx, %rdx",
                "negq %rdx",
                inout("rdi") dst => _,
                inout("rsi") src => _,
                inout("r11") len => _,
                inout("rdx") scalar => borrow,
                out("rax") carry,
                out("r8") _,
                out("r9") _,
                out("r10") _,
                options(nostack, att_syntax),
            );
        }
        return (carry, borrow);
    }
    let carry_hi: Limb;
    let borrow_out: Limb;
    let chunks = len >> 2;
    let rem = len & 3;

    // SAFETY: len >= 4 supplies at least one complete four-limb block; its
    // shorter tail remains in the aligned initialized disjoint spans. Product
    // plus carry is below B^2, so its high digit absorbs OF. SBB preserves the
    // binary borrow in CF. DEC preserves CF and clears OF: the byte bound
    // excludes the signed minimum counter. The caller proves ADX/BMI2, and
    // RDX/RAX hold return values only after their scalar/zero uses end.
    unsafe {
        asm!(
            "xorl %ecx, %ecx",                           // rcx = 0, clears CF and OF
            "xorl %eax, %eax",                           // rax = 0 (zero register for OF absorption)
            "decq {chunks}",                             // len >= 4 proves at least one chunk

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

            // [4-Limb Sequential Subtraction via CF Borrow Chain directly to memory]
            "sbbq %r8, 0({dst})",                        // dst[0] -= r8 + CF
            "sbbq %r10, 8({dst})",                       // dst[1] -= r10 + CF
            "sbbq %r9, 16({dst})",                       // dst[2] -= r9 + CF
            "sbbq %r11, 24({dst})",                      // dst[3] -= r11 + CF

            "leaq 32({src}), {src}",                     // Advance src pointer by 32 bytes
            "leaq 32({dst}), {dst}",                     // Advance dst pointer by 32 bytes
            "decq {chunks}",                             // Decrement chunk counter (preserves CF)
            "jns 2b",                                    // Repeat while chunks >= 0

            // Remainder processing entry point (0 to 3 limbs)
            "decq {rem}",                                // Pre-decrement remainder counter
            "js 4f",                                     // If rem < 0, skip to finish (4f)

            // Select the initialized tail before its carry chain. DEC keeps
            // CF and clears OF for these bounded counters. Overlapping the
            // products preserves the full block's pipeline for 2-3 limbs.
            "decq {rem}",
            "js 6f",
            "decq {rem}",
            "js 5f",
            "mulxq 0({src}), %r8, %r9",
            "adoxq %rcx, %r8",
            "mulxq 8({src}), %r10, %r11",
            "adoxq %r9, %r10",
            "mulxq 16({src}), %r9, %rcx",
            "adoxq %r11, %r9",
            "adoxq %rax, %rcx",
            "sbbq %r8, 0({dst})",
            "sbbq %r10, 8({dst})",
            "sbbq %r9, 16({dst})",
            "jmp 4f",
            "5:",
            "mulxq 0({src}), %r8, %r9",
            "adoxq %rcx, %r8",
            "mulxq 8({src}), %r10, %rcx",
            "adoxq %r9, %r10",
            "adoxq %rax, %rcx",
            "sbbq %r8, 0({dst})",
            "sbbq %r10, 8({dst})",
            "jmp 4f",
            "6:",
            "mulxq 0({src}), %r8, %r9",
            "adoxq %rcx, %r8",
            "adoxq %rax, %r9",
            "sbbq %r8, 0({dst})",
            "movq %r9, %rcx",

            // Final carry and borrow extraction
            "4:",                                        // Finish label
            "sbbq %rdx, %rdx",                           // Scalar is dead; extract -borrow
            "negq %rdx",                                 // Return boolean borrow in rdx
            "movq %rcx, %rax",                           // Zero constant is dead; return carry

            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            chunks = inout(reg) chunks => _,
            rem = inout(reg) rem => _,
            inout("rdx") scalar => borrow_out,
            out("rax") carry_hi,
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
