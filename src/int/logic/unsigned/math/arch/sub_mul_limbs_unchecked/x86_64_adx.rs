//! ADX/BMI2 x86-64 multiply-subtract limb kernel.

use core::arch::asm;

use super::Limb;

/// Multiply `len` limbs from `src` by `scalar`, subtract the result from
/// `dst`, and return the final `(carry, borrow)` pair.
///
/// This computes:
///
/// ```text
///   (borrow, carry, dst[0..len]) = dst[0..len] - (src[0..len] × scalar)
/// ```
///
/// This implementation utilizes `mulx` (BMI2) for flag-free multiplication with
/// dual carry tracking via the ADX flag pair. Since subtraction cannot easily use
/// dual-flag `adcx`/`adox` parallelism efficiently, we track both chains in
/// general-purpose registers where appropriate.
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

    // SAFETY: Assembly block accesses `len` elements from `dst` and `src`.
    // Uses mulx (BMI2) for flag-free multiplication with dual carry tracking:
    // - r10: multiplication carry chain (hi words from mulx)
    // - r11: subtraction borrow chain (underflow tracking)
    // Unlike add_mul, subtraction cannot use dual-flag adcx/adox parallelism,
    // so we track both chains in general-purpose registers.
    unsafe {
        asm!(
            "xorl %ecx, %ecx",
            "xorl %eax, %eax",

            "decq {chunks}",
            "js 1f",
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

            "sbbq %r8, 0({dst})",
            "sbbq %r10, 8({dst})",
            "sbbq %r9, 16({dst})",
            "sbbq %r11, 24({dst})",

            "leaq 32({src}), {src}",
            "leaq 32({dst}), {dst}",
            "decq {chunks}",
            "jns 2b",

            "1:",
            "decq {rem}",
            "js 4f",
            "3:",
            "mulxq 0({src}), %r8, %r9",
            "adoxq %rcx, %r8",
            "adoxq %rax, %r9",

            "sbbq %r8, 0({dst})",
            "movq %r9, %rcx",

            "leaq 8({src}), {src}",
            "leaq 8({dst}), {dst}",
            "decq {rem}",
            "jns 3b",

            "4:",
            "movq $0, %r8",
            "adcq %r8, %r8",
            "movq %r8, {borrow_out}",
            "movq %rcx, {carry_hi}",

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
