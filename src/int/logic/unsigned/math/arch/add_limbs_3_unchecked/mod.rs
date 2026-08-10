//! Architecture-specific three-buffer addition kernel.

#![allow(
    unsafe_code,
    reason = "hardware inline assembly natively requires unsafe code"
)]

use super::Limb;

select_arch_kernel! {
    function: add_limbs_3_unchecked;
    surface: direct;
    backends: [
        x86 => all(not(miri), target_arch = "x86", target_pointer_width = "32"),
        aarch64 => all(not(miri), target_arch = "aarch64", target_pointer_width = "64"),
        arm => all(not(miri), target_arch = "arm"),
        powerpc => all(not(miri), target_arch = "powerpc"),
        s390x => all(not(miri), target_arch = "s390x"),
        riscv64 => all(not(miri), target_arch = "riscv64"),
        riscv32 => all(not(miri), target_arch = "riscv32"),
        loongarch64 => all(not(miri), target_arch = "loongarch64"),
        loongarch32 => all(not(miri), target_arch = "loongarch32"),
        mips64 => all(not(miri), target_arch = "mips64"),
        mips => all(not(miri), target_arch = "mips"),
    ];
    x86_64: [baseline];
    powerpc64: [baseline];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
        target_arch = "powerpc64",
    ];
    fallback_imports: [];
}
