//! Architecture-selected single-limb multiply-add kernel.

#![allow(
    unsafe_code,
    reason = "hardware inline assembly natively requires unsafe code"
)]

use super::Limb;

select_arch_kernel! {
    function: add_mul_limbs_unchecked;
    kernel: AddMulKernel;
    surface: provider;
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
    x86_64: [bmi2, adx_bmi2];
    powerpc64: [power8, power9];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
        target_arch = "powerpc64",
    ];
    fallback_imports: [DoubleLimb, LIMB_BITS];
    runtime_backends: [
        add_mul_limbs_vanilla_backend => x86_64,
        add_mul_limbs_adx_backend => x86_64_adx,
        add_mul_limbs_bmi2_backend => x86_64_bmi2,
    ];
    test_backends: [
        add_mul_limbs_vanilla_test => x86_64,
        add_mul_limbs_adx_test => x86_64_adx,
        add_mul_limbs_bmi2_test => x86_64_bmi2,
    ];
}
