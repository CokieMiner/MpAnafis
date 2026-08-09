//! x86-64 BMI2 shifted-high subtraction.

use core::arch::asm;

use super::Limb;

/// Subtract a cross-limb shifted source span from `dst`, including `borrow`.
///
/// `SHLX`/`SHRX`, `LEA`, and the pointer updates preserve the carry flag, so
/// one `SBB` chain runs across the complete span. The last source limb is
/// handled separately because the mathematical limb above it is zero, not an
/// addressable padding or Fermat guard limb.
///
/// # Safety
///
/// - `dst` and `src` must each cover `len` limbs.
/// - Their active spans must not overlap.
/// - `0 < shift < Limb::BITS`.
/// - `borrow <= 1`.
/// - The caller must ensure the CPU supports BMI2.
#[allow(
    clippy::as_conversions,
    reason = "shift is strictly below 64 and therefore fits the x86-64 shift-count register"
)]
pub unsafe fn sub_shifted_high_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
    borrow: Limb,
) -> Limb {
    debug_assert!(
        shift > 0 && shift < Limb::BITS,
        "the cross-limb shift must be strictly inside one limb"
    );
    debug_assert!(borrow <= 1, "a subtraction borrow is one bit");
    if len == 0 {
        return borrow;
    }

    let dst_ptr = dst;
    let src_ptr = src;
    // Leave one final limb outside the paired loop. An even length leaves one
    // additional ordinary limb before that final zero-extended limb.
    let pair_count = len.wrapping_sub(1).wrapping_shr(1);
    let even_extra = usize::from(len.is_multiple_of(2));
    let left_shift = shift as usize;
    let right_shift = (Limb::BITS as usize).wrapping_sub(left_shift);
    let borrow_out: u8;

    // SAFETY: the caller proves both `len`-limb spans and BMI2 availability.
    // Each paired iteration reads three source limbs to produce two results;
    // `pair_count = (len - 1) / 2` keeps that third read inside `src[..len]`.
    // The optional even-length limb and the final zero-extended limb complete
    // the span. Every instruction between SBB operations is flag-neutral.
    unsafe {
        asm!(
            "movq ({src_ptr}), %r8",
            "btq $0, {borrow}",
            "jrcxz 2f",
            ".p2align 4",
            "1:",
            "movq 8({src_ptr}), %r9",
            "movq 16({src_ptr}), %r10",
            "shrxq {right_shift}, %r8, %r11",
            "shlxq {left_shift}, %r9, %r12",
            "leaq (%r11,%r12), %r11",
            "movq ({dst_ptr}), %r13",
            "sbbq %r11, %r13",
            "movq %r13, ({dst_ptr})",
            "shrxq {right_shift}, %r9, %r11",
            "shlxq {left_shift}, %r10, %r12",
            "leaq (%r11,%r12), %r11",
            "movq 8({dst_ptr}), %r13",
            "sbbq %r11, %r13",
            "movq %r13, 8({dst_ptr})",
            "movq %r10, %r8",
            "leaq 16({src_ptr}), {src_ptr}",
            "leaq 16({dst_ptr}), {dst_ptr}",
            "leaq -1(%rcx), %rcx",
            "jrcxz 2f",
            "jmp 1b",
            "2:",
            "movq {even_extra}, %rcx",
            "jrcxz 3f",
            "movq 8({src_ptr}), %r9",
            "shrxq {right_shift}, %r8, %r11",
            "shlxq {left_shift}, %r9, %r12",
            "leaq (%r11,%r12), %r11",
            "movq ({dst_ptr}), %r13",
            "sbbq %r11, %r13",
            "movq %r13, ({dst_ptr})",
            "movq %r9, %r8",
            "leaq 8({src_ptr}), {src_ptr}",
            "leaq 8({dst_ptr}), {dst_ptr}",
            "3:",
            "shrxq {right_shift}, %r8, %r11",
            "movq ({dst_ptr}), %r13",
            "sbbq %r11, %r13",
            "movq %r13, ({dst_ptr})",
            "setc {borrow_out}",
            src_ptr = inout(reg) src_ptr => _,
            dst_ptr = inout(reg) dst_ptr => _,
            inout("rcx") pair_count => _,
            even_extra = in(reg) even_extra,
            left_shift = in(reg) left_shift,
            right_shift = in(reg) right_shift,
            borrow = in(reg) borrow,
            borrow_out = lateout(reg_byte) borrow_out,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            out("r12") _,
            out("r13") _,
            options(nostack, att_syntax)
        );
    }
    Limb::from(borrow_out)
}
