//! 32-bit x86 in-place subtraction kernel.

use core::arch::asm;

use super::Limb;

/// Subtract `src[0..len]` from `dst[0..len]` and return the final borrow.
///
/// # Safety
///
/// Both pointers must cover `len` limbs and the spans must not overlap.
#[allow(
    clippy::inline_always,
    reason = "Subtraction is a foundational limb loop and preserving CF avoids per-limb borrow materialization"
)]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    if len == 1 {
        // SAFETY: the caller guarantees both pointers cover the sole limb.
        let (difference, underflow) = unsafe { (*dst).overflowing_sub(*src) };
        // SAFETY: the caller guarantees the destination limb is writable.
        unsafe {
            *dst = difference;
        }
        return Limb::from(underflow);
    }

    let borrow: Limb;
    let prefix = (len >> 2) & 1;
    let chunks = len >> 3;
    let remainder = len & 3;
    // Each SBB consumes the prior limb's CF. MOV/LEA preserve all flags and DEC
    // preserves CF, so loop control cannot disturb the borrow chain. The signed
    // DEC/JS sentinel is valid because at most 2^30 four-byte limbs fit in a
    // 32-bit address space.
    // SAFETY: the caller supplies both len-limb spans. The unrolled body and
    // tail access exactly len elements and restore no external machine state.
    unsafe {
        asm!(
            "xorl {borrow}, {borrow}",
            "decl {prefix}",
            "js 1f",
            "movl 0({src}), %eax",
            "sbbl %eax, 0({dst})",
            "movl 4({src}), %eax",
            "sbbl %eax, 4({dst})",
            "movl 8({src}), %eax",
            "sbbl %eax, 8({dst})",
            "movl 12({src}), %eax",
            "sbbl %eax, 12({dst})",
            "leal 16({src}), {src}",
            "leal 16({dst}), {dst}",
            "1:",
            "decl {chunks}",
            "js 3f",
            ".p2align 4",
            "2:",
            "movl 0({src}), %eax",
            "sbbl %eax, 0({dst})",
            "movl 4({src}), %eax",
            "sbbl %eax, 4({dst})",
            "movl 8({src}), %eax",
            "sbbl %eax, 8({dst})",
            "movl 12({src}), %eax",
            "sbbl %eax, 12({dst})",
            "movl 16({src}), %eax",
            "sbbl %eax, 16({dst})",
            "movl 20({src}), %eax",
            "sbbl %eax, 20({dst})",
            "movl 24({src}), %eax",
            "sbbl %eax, 24({dst})",
            "movl 28({src}), %eax",
            "sbbl %eax, 28({dst})",
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
            "sbbl %eax, 0({dst})",
            "leal 4({src}), {src}",
            "leal 4({dst}), {dst}",
            "decl {remainder}",
            "jns 4b",
            "5:",
            "adcl {borrow}, {borrow}",
            dst = inout(reg) dst => _,
            src = inout(reg) src => _,
            prefix = inout(reg) prefix => _,
            chunks = inout(reg) chunks => _,
            remainder = inout(reg) remainder => _,
            borrow = out(reg) borrow,
            out("eax") _,
            options(nostack, att_syntax)
        );
    }
    borrow
}
