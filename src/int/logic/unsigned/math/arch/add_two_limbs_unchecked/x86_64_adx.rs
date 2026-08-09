//! x86-64 dual addition using the independent ADX carry chains.

use core::arch::asm;

use super::Limb;

/// Add two independent source spans into two destination spans.
///
/// `ADCX` carries the first sum through CF while `ADOX` carries the second
/// through OF. Pointer and counter updates preserve both flags, so one loop
/// retires the two otherwise independent reconstruction passes.
///
/// # Safety
///
/// - Every pointer must cover `len` readable limbs.
/// - Both destination pointers must cover `len` writable limbs.
/// - No destination span may overlap any other span.
/// - The caller must ensure the CPU supports ADX.
pub unsafe fn add_two_limbs_unchecked(
    dst_a: *mut Limb,
    src_a: *const Limb,
    dst_b: *mut Limb,
    src_b: *const Limb,
    len: usize,
) -> (Limb, Limb) {
    if len == 0 {
        return (0, 0);
    }
    let block_count = len >> 2;
    let tail_count = len & 3;
    let carry_a: u8;
    let carry_b: u8;

    // SAFETY: the caller proves all four spans valid and disjoint. The block
    // loop consumes exactly `len / 4` four-limb groups, then the tail loop
    // consumes `len % 4` limbs. Flag-neutral pointer/counter instructions keep
    // both carry chains live. Dispatch proves ADX support before this block.
    unsafe {
        asm!(
            "xorl %eax, %eax",
            "jrcxz 1f",
            "jmp 2f",
            "1:",
            "jmp 3f",
            "2:",
            "movq ({dst_a}), %r8",
            "movq ({src_a}), %r9",
            "adcxq %r9, %r8",
            "movq %r8, ({dst_a})",
            "movq ({dst_b}), %r10",
            "movq ({src_b}), %r11",
            "adoxq %r11, %r10",
            "movq %r10, ({dst_b})",
            "movq 8({dst_a}), %r8",
            "movq 8({src_a}), %r9",
            "adcxq %r9, %r8",
            "movq %r8, 8({dst_a})",
            "movq 8({dst_b}), %r10",
            "movq 8({src_b}), %r11",
            "adoxq %r11, %r10",
            "movq %r10, 8({dst_b})",
            "movq 16({dst_a}), %r8",
            "movq 16({src_a}), %r9",
            "adcxq %r9, %r8",
            "movq %r8, 16({dst_a})",
            "movq 16({dst_b}), %r10",
            "movq 16({src_b}), %r11",
            "adoxq %r11, %r10",
            "movq %r10, 16({dst_b})",
            "movq 24({dst_a}), %r8",
            "movq 24({src_a}), %r9",
            "adcxq %r9, %r8",
            "movq %r8, 24({dst_a})",
            "movq 24({dst_b}), %r10",
            "movq 24({src_b}), %r11",
            "adoxq %r11, %r10",
            "movq %r10, 24({dst_b})",
            "leaq 32({dst_a}), {dst_a}",
            "leaq 32({src_a}), {src_a}",
            "leaq 32({dst_b}), {dst_b}",
            "leaq 32({src_b}), {src_b}",
            "leaq -1(%rcx), %rcx",
            "jrcxz 3f",
            "jmp 2b",
            "3:",
            "movq {tail_count}, %rcx",
            "jrcxz 5f",
            "4:",
            "movq ({dst_a}), %r8",
            "movq ({src_a}), %r9",
            "adcxq %r9, %r8",
            "movq %r8, ({dst_a})",
            "movq ({dst_b}), %r10",
            "movq ({src_b}), %r11",
            "adoxq %r11, %r10",
            "movq %r10, ({dst_b})",
            "leaq 8({dst_a}), {dst_a}",
            "leaq 8({src_a}), {src_a}",
            "leaq 8({dst_b}), {dst_b}",
            "leaq 8({src_b}), {src_b}",
            "leaq -1(%rcx), %rcx",
            "jrcxz 5f",
            "jmp 4b",
            "5:",
            "setc {carry_a}",
            "seto {carry_b}",
            dst_a = inout(reg) dst_a => _,
            src_a = inout(reg) src_a => _,
            dst_b = inout(reg) dst_b => _,
            src_b = inout(reg) src_b => _,
            inout("rcx") block_count => _,
            tail_count = in(reg) tail_count,
            carry_a = lateout(reg_byte) carry_a,
            carry_b = lateout(reg_byte) carry_b,
            out("rax") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            options(nostack, att_syntax)
        );
    }
    (Limb::from(carry_a), Limb::from(carry_b))
}
