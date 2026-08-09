//! 32-bit x86 three-span addition kernel.

use core::arch::asm;

use super::Limb;

/// Write `src1 + src2` into `dst` and return the final carry.
///
/// # Safety
///
/// All three pointers must cover `len` limbs and the destination must not
/// overlap either input.
#[allow(
    clippy::inline_always,
    reason = "Three-span addition is a hot interpolation loop and preserving CF removes per-limb carry conversion"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 1 {
        // SAFETY: the caller guarantees both source pointers cover one limb.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: the caller guarantees one writable destination limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }

    let carry: Limb;
    let chunks = len >> 2;
    let remainder = len & 3;
    // ADC carries flow directly between limbs because every intervening MOV,
    // LEA, DEC, and conditional jump preserves CF. At most 2^30 limbs fit in
    // the 32-bit address space, proving the signed chunk sentinel cannot wrap.
    // SAFETY: all pointers cover len limbs. The body and tail access exactly
    // that many positions and write only the disjoint destination span.
    unsafe {
        asm!(
            "xorl {carry}, {carry}",
            "decl {chunks}",
            "js 2f",
            ".p2align 4",
            "1:",
            "movl 0({src1}), %eax",
            "adcl 0({src2}), %eax",
            "movl %eax, 0({dst})",
            "movl 4({src1}), %eax",
            "adcl 4({src2}), %eax",
            "movl %eax, 4({dst})",
            "movl 8({src1}), %eax",
            "adcl 8({src2}), %eax",
            "movl %eax, 8({dst})",
            "movl 12({src1}), %eax",
            "adcl 12({src2}), %eax",
            "movl %eax, 12({dst})",
            "leal 16({src1}), {src1}",
            "leal 16({src2}), {src2}",
            "leal 16({dst}), {dst}",
            "decl {chunks}",
            "jns 1b",
            "2:",
            "decl {remainder}",
            "js 4f",
            "3:",
            "movl 0({src1}), %eax",
            "adcl 0({src2}), %eax",
            "movl %eax, 0({dst})",
            "leal 4({src1}), {src1}",
            "leal 4({src2}), {src2}",
            "leal 4({dst}), {dst}",
            "decl {remainder}",
            "jns 3b",
            "4:",
            "adcl {carry}, {carry}",
            dst = inout(reg) dst => _,
            src1 = inout(reg) src1 => _,
            src2 = inout(reg) src2 => _,
            chunks = inout(reg) chunks => _,
            remainder = inout(reg) remainder => _,
            carry = out(reg) carry,
            out("eax") _,
            options(nostack, att_syntax)
        );
    }
    carry
}
