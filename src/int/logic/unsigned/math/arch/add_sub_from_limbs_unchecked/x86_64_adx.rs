//! x86-64 ADX shared-source simultaneous addition and subtraction.

use core::arch::asm;

use super::Limb;

/// Replace `sum` with `sum_original + source` and write
/// `sum_original - source` to `difference`, returning carry and borrow.
///
/// # Safety
///
/// - All three pointers must cover `len` limbs.
/// - `difference` may equal `source` exactly, but otherwise must not overlap an input span.
/// - `sum` and `source` must not overlap.
/// - The caller must ensure the CPU supports ADX.
#[allow(
    clippy::too_many_lines,
    reason = "Eight-limb ADX unrolling retained a repeatable 1-1.5% Fermat-transform win"
)]
pub unsafe fn add_sub_from_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    source: *const Limb,
    len: usize,
) -> (Limb, Limb) {
    if len == 0 {
        return (0, 0);
    }
    let sum_ptr = sum;
    let difference_ptr = difference;
    let source_ptr = source;
    let block_count = len >> 3;
    let tail_count = len & 7;
    let sum_carry: u8;
    let difference_carry: u8;

    // SAFETY: the caller provides the three valid spans documented above.
    // The loop consumes `len / 8` eight-limb groups, then `len % 8` single limbs.
    // Flag-neutral pointer/counter instructions keep both carry chains live.
    // Dispatch proves ADX support before entering this block.
    unsafe {
        asm!(
            "movabsq $0x7fffffffffffffff, %r8",
            "addq $1, %r8",
            "jrcxz 1f",
            "jmp 2f",
            "1: jmp 3f",
            "2:",
            "movq ({sum_ptr}), %r8",
            "movq ({source_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, ({sum_ptr})",
            "movq %r8, ({difference_ptr})",
            "movq 8({sum_ptr}), %r8",
            "movq 8({source_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, 8({sum_ptr})",
            "movq %r8, 8({difference_ptr})",
            "movq 16({sum_ptr}), %r8",
            "movq 16({source_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, 16({sum_ptr})",
            "movq %r8, 16({difference_ptr})",
            "movq 24({sum_ptr}), %r8",
            "movq 24({source_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, 24({sum_ptr})",
            "movq %r8, 24({difference_ptr})",

            "movq 32({sum_ptr}), %r8",
            "movq 32({source_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, 32({sum_ptr})",
            "movq %r8, 32({difference_ptr})",
            "movq 40({sum_ptr}), %r8",
            "movq 40({source_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, 40({sum_ptr})",
            "movq %r8, 40({difference_ptr})",
            "movq 48({sum_ptr}), %r8",
            "movq 48({source_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, 48({sum_ptr})",
            "movq %r8, 48({difference_ptr})",
            "movq 56({sum_ptr}), %r8",
            "movq 56({source_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, 56({sum_ptr})",
            "movq %r8, 56({difference_ptr})",
            "leaq 64({sum_ptr}), {sum_ptr}",
            "leaq 64({difference_ptr}), {difference_ptr}",
            "leaq 64({source_ptr}), {source_ptr}",
            "leaq -1(%rcx), %rcx",
            "jrcxz 3f",
            "jmp 2b",
            "3:",
            "movq {tail_count}, %rcx",
            "jrcxz 5f",
            "4:",
            "movq ({sum_ptr}), %r8",
            "movq ({source_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, ({sum_ptr})",
            "movq %r8, ({difference_ptr})",
            "leaq 8({sum_ptr}), {sum_ptr}",
            "leaq 8({difference_ptr}), {difference_ptr}",
            "leaq 8({source_ptr}), {source_ptr}",
            "leaq -1(%rcx), %rcx",
            "jrcxz 5f",
            "jmp 4b",
            "5:",
            "setc {sum_carry}",
            "seto {difference_carry}",
            sum_ptr = inout(reg) sum_ptr => _,
            difference_ptr = inout(reg) difference_ptr => _,
            source_ptr = inout(reg) source_ptr => _,
            inout("rcx") block_count => _,
            tail_count = in(reg) tail_count,
            sum_carry = lateout(reg_byte) sum_carry,
            difference_carry = lateout(reg_byte) difference_carry,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            options(nostack, att_syntax)
        );
    }
    (Limb::from(sum_carry), Limb::from(difference_carry == 0))
}
