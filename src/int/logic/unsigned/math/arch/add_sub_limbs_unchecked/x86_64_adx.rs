//! x86-64 ADX simultaneous addition and subtraction.

use core::arch::asm;

use super::Limb;

/// Replace `sum` with `sum + difference` and `difference` with
/// `sum_original - difference_original`, returning carry and borrow.
///
/// `ADCX` propagates the sum through CF while `ADOX` computes
/// `a + !b + 1` through OF. Pointer and counter updates use only flag-neutral
/// instructions, so both chains remain live across the complete loop.
///
/// # Safety
///
/// - Both pointers must be valid for reads and writes of `len` limbs.
/// - The two spans must not overlap.
/// - The caller must ensure the CPU supports ADX.
pub unsafe fn add_sub_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    len: usize,
) -> (Limb, Limb) {
    if len == 0 {
        return (0, 0);
    }
    let sum_ptr = sum;
    let difference_ptr = difference;
    let block_count = len >> 2;
    let tail_count = len & 3;
    let sum_carry: u8;
    let difference_carry: u8;

    // SAFETY: the caller provides two disjoint readable/writable spans of
    // `len` limbs. The loop consumes `len / 4` four-limb groups, then `len % 4`
    // single limbs. Flag-neutral pointer/counter instructions keep both carry
    // chains live. Dispatch proves ADX support before this block runs.
    unsafe {
        asm!(
            "movabsq $0x7fffffffffffffff, %r8",
            "addq $1, %r8",
            "jrcxz 1f",
            "jmp 2f",
            "1: jmp 3f",
            "2:",
            "movq ({sum_ptr}), %r8",
            "movq ({difference_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, ({sum_ptr})",
            "movq %r8, ({difference_ptr})",
            "movq 8({sum_ptr}), %r8",
            "movq 8({difference_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, 8({sum_ptr})",
            "movq %r8, 8({difference_ptr})",
            "movq 16({sum_ptr}), %r8",
            "movq 16({difference_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, 16({sum_ptr})",
            "movq %r8, 16({difference_ptr})",
            "movq 24({sum_ptr}), %r8",
            "movq 24({difference_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, 24({sum_ptr})",
            "movq %r8, 24({difference_ptr})",
            "leaq 32({sum_ptr}), {sum_ptr}",
            "leaq 32({difference_ptr}), {difference_ptr}",
            "leaq -1(%rcx), %rcx",
            "jrcxz 3f",
            "jmp 2b",
            "3:",
            "movq {tail_count}, %rcx",
            "jrcxz 5f",
            "4:",
            "movq ({sum_ptr}), %r8",
            "movq ({difference_ptr}), %r9",
            "movq %r8, %r10",
            "adcxq %r9, %r10",
            "notq %r9",
            "adoxq %r9, %r8",
            "movq %r10, ({sum_ptr})",
            "movq %r8, ({difference_ptr})",
            "leaq 8({sum_ptr}), {sum_ptr}",
            "leaq 8({difference_ptr}), {difference_ptr}",
            "leaq -1(%rcx), %rcx",
            "jrcxz 5f",
            "jmp 4b",
            "5:",
            "setc {sum_carry}",
            "seto {difference_carry}",
            sum_ptr = inout(reg) sum_ptr => _,
            difference_ptr = inout(reg) difference_ptr => _,
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
