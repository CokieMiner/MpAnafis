//! Baseline x86-64 write-only `mul_2` kernel.

use core::arch::asm;

use super::Limb;

/// Write `src * (s0 + s1 * B)` into `dst` without reading its old contents.
///
/// # Safety
///
/// `src` must be valid for `len` limbs and `dst` for `len + 2` limbs.  The
/// input and output regions must not overlap.  A zero length returns without
/// dereferencing either pointer.
#[allow(
    clippy::inline_always,
    reason = "Critical basecase initialization kernel; the call is selected once and is tiny for small products"
)]
#[inline(always)]
pub unsafe fn mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) {
    if len == 0 {
        return;
    }

    // SAFETY: the caller guarantees the source and destination spans; the
    // assembly touches exactly the first `len` source and `len+2` output limbs.
    unsafe {
        asm!(
            "movq {s0}, %rax",
            "mulq ({src})",
            "movq %rax, ({dst})",
            "movq %rdx, %r8",
            "movq {s1}, %rax",
            "mulq ({src})",
            "movq %rax, 8({dst})",
            "movq %rdx, %r9",
            "leaq 8({src}), {src}",
            "leaq 8({dst}), {dst}",
            "decq {len}",
            "jz 2f",

            ".p2align 4",
            "1:",
            "movq {s0}, %rax",
            "mulq ({src})",
            "addq %r8, %rax",
            "adcq $0, %rdx",
            "addq ({dst}), %rax",
            "adcq $0, %rdx",
            "movq %rax, ({dst})",
            "movq %rdx, %r8",
            "movq {s1}, %rax",
            "mulq ({src})",
            "addq %r9, %rax",
            "adcq $0, %rdx",
            "movq %rax, 8({dst})",
            "movq %rdx, %r9",
            "leaq 8({src}), {src}",
            "leaq 8({dst}), {dst}",
            "decq {len}",
            "jnz 1b",

            "2:",
            "addq %r8, ({dst})",
            "adcq $0, %r9",
            "movq %r9, 8({dst})",

            len = inout(reg) len => _,
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            out("rax") _,
            out("rdx") _,
            out("r8") _,
            out("r9") _,
            options(nostack, att_syntax)
        );
    }
}
