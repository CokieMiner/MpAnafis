//! 32-bit x86 in-place addition kernel.

use core::arch::asm;

use super::Limb;

/// Add `src[0..len]` into `dst[0..len]` and return the final carry.
///
/// # Safety
///
/// Both pointers must cover `len` limbs and the spans must not overlap.
#[allow(
    clippy::inline_always,
    reason = "Addition is a foundational limb loop and preserving CF avoids per-limb carry materialization"
)]
#[inline(always)]
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    if len == 1 {
        // SAFETY: the caller guarantees both pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*dst).overflowing_add(*src) };
        // SAFETY: the caller guarantees the destination limb is writable.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }

    let carry: Limb;
    let prefix = (len >> 2) & 1;
    let chunks = len >> 3;
    let remainder = len & 3;
    // Every ADC consumes the CF produced by the preceding limb. MOV and LEA do
    // not change flags, while DEC changes every status flag except CF; hence
    // the loop control cannot break the carry chain. A 32-bit address space
    // holds fewer than 2^30 four-byte limbs, so the signed DEC/JS sentinel is
    // valid for every possible allocation.
    // SAFETY: the caller supplies both len-limb spans. The four-limb body and
    // tail together access exactly len elements and only advance valid pointers.
    unsafe {
        asm!(
            "xorl {carry}, {carry}",
            "decl {prefix}",
            "js 1f",
            "movl 0({src}), %eax",
            "adcl %eax, 0({dst})",
            "movl 4({src}), %eax",
            "adcl %eax, 4({dst})",
            "movl 8({src}), %eax",
            "adcl %eax, 8({dst})",
            "movl 12({src}), %eax",
            "adcl %eax, 12({dst})",
            "leal 16({src}), {src}",
            "leal 16({dst}), {dst}",
            "1:",
            "decl {chunks}",
            "js 3f",
            ".p2align 4",
            "2:",
            "movl 0({src}), %eax",
            "adcl %eax, 0({dst})",
            "movl 4({src}), %eax",
            "adcl %eax, 4({dst})",
            "movl 8({src}), %eax",
            "adcl %eax, 8({dst})",
            "movl 12({src}), %eax",
            "adcl %eax, 12({dst})",
            "movl 16({src}), %eax",
            "adcl %eax, 16({dst})",
            "movl 20({src}), %eax",
            "adcl %eax, 20({dst})",
            "movl 24({src}), %eax",
            "adcl %eax, 24({dst})",
            "movl 28({src}), %eax",
            "adcl %eax, 28({dst})",
            "leal 32({src}), {src}",
            "leal 32({dst}), {dst}",
            "decl {chunks}",
            "jns 2b",
            "3:",
            "decl {remainder}",
            "js 5f",
            ".p2align 4",
            "4:",
            "movl 0({src}), %eax",
            "adcl %eax, 0({dst})",
            "leal 4({src}), {src}",
            "leal 4({dst}), {dst}",
            "decl {remainder}",
            "jns 4b",
            "5:",
            "adcl {carry}, {carry}",
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            prefix = inout(reg) prefix => _,
            chunks = inout(reg) chunks => _,
            remainder = inout(reg) remainder => _,
            carry = out(reg) carry,
            out("eax") _,
            options(nostack, att_syntax)
        );
    }
    carry
}
