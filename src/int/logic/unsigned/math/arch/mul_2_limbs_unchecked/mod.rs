//! Write-only dual-row multiplication kernels for basecase initialization.

#![allow(
    unsafe_code,
    reason = "The kernels use raw pointers and inline assembly after callers validate their buffers"
)]

use super::Limb;

select_arch_kernel! {
    function: mul_2_limbs_unchecked;
    kernel: Mul2Kernel;
    surface: provider;
    backends: [
        x86 => all(not(miri), target_arch = "x86", target_pointer_width = "32"),
        aarch64 => all(not(miri), target_arch = "aarch64", target_pointer_width = "64"),
        riscv64 => all(not(miri), target_arch = "riscv64", target_pointer_width = "64"),
        s390x => all(not(miri), target_arch = "s390x", target_pointer_width = "64"),
    ];
    x86_64: [bmi2];
    powerpc64: [baseline];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
        all(target_arch = "powerpc64", target_pointer_width = "64"),
    ];
    fallback_imports: [DoubleLimb, LIMB_BITS];
    runtime_backends: [
        mul_2_limbs_vanilla_backend => x86_64,
        mul_2_limbs_bmi2_backend => x86_64_bmi2,
    ];
    test_backends: [
        mul_2_limbs_vanilla_test => x86_64,
        mul_2_limbs_bmi2_test => x86_64_bmi2,
    ];
}
