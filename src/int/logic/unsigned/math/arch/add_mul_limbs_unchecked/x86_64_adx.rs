//! ADX/BMI2 x86-64 multiply-add limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// This computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// This implementation uses the `x86_64` ADX and BMI2 instruction sets, with
/// `mulxq` supplying flag-free multiplication and two independent carry chains
/// riding the OF and CF flags.
///
/// The loop is unrolled by a factor of 4. `adoxq` threads each product's high
/// half into the next limb's low half over OF, assembling one whole row in
/// registers; the row is then accumulated with memory-destination `adcq`, whose
/// CF chain carries the destination addition. Folding the read-modify-write into
/// `adcq` costs one instruction per limb where a separate load, register add and
/// store would cost two.
///
/// This works only because of the loop control. `adcq` writes CF *and* OF, so it
/// destroys the OF chain — but the chain is closed into `%rcx` by `adoxq %rax,
/// %rcx` beforehand, and `decq` then rewrites OF from the signed overflow of the
/// decrement, which is always zero for a limb count. The counter therefore hands
/// the next iteration a cleared OF, while leaving CF untouched so the
/// destination chain survives, including across the transition into the tail
/// loop. Replacing `decq` with flag-preserving loop control would leave the
/// `adcq` residue in OF and silently corrupt the row.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `src` and `dst` spans must not overlap, even partially: the loop
///   reads `src` while it writes `dst`, so any overlap is a data race.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
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

    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`.
    unsafe {
        asm!(
            // rcx = running high half, rax = 0; `xorl` also clears CF and OF.
            "xorl %ecx, %ecx",
            "xorl %eax, %eax",

            "decq {chunks}",
            "js 1f",
            ".p2align 4",
            "2:",
            "mulxq 0({src}), %r8, %r9",
            "adoxq %rcx, %r8",

            "mulxq 8({src}), %r10, %r11",
            "adoxq %r9, %r10",

            "mulxq 16({src}), %r9, %r12",
            "adoxq %r11, %r9",

            "mulxq 24({src}), %r11, %rcx",
            "adoxq %r12, %r11",

            "adoxq %rax, %rcx",

            "adcq %r8, 0({dst})",
            "adcq %r10, 8({dst})",
            "adcq %r9, 16({dst})",
            "adcq %r11, 24({dst})",

            "leaq 32({src}), {src}",
            "leaq 32({dst}), {dst}",
            "decq {chunks}",
            "jns 2b",

            "1:",
            "decq {rem}",
            "js 4f",
            ".p2align 4",
            "3:",
            "mulxq 0({src}), %r8, %r9",
            "adoxq %rcx, %r8",
            "adoxq %rax, %r9",

            "adcq %r8, 0({dst})",
            "movq %r9, %rcx",

            "leaq 8({src}), {src}",
            "leaq 8({dst}), {dst}",
            "decq {rem}",
            "jns 3b",

            // Fold the pending destination carry into the returned high half.
            "4:",
            "movq $0, %r8",
            "adcq %r8, %rcx",
            "movq %rcx, {carry_hi}",

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
            out("r12") _,
            options(nostack, att_syntax)
        );
    }
    carry_hi
}
