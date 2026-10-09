//! Interleaved accumulation of two scalar-product rows.
//!
//! With base `B = 2^LIMB_BITS`, the result satisfies
//! `dst_new + c0 * B^len + c1 * B^(len + 1) = dst_old + src * (s0 + s1 * B)`.
//! Each row sum is at most `(B - 1)^2 + 2 * (B - 1) = B^2 - 1`.

#![expect(
    unsafe_code,
    reason = "Raw pointer access and inline assembly implement the limb-span contract"
)]

use super::Limb;

select_arch_kernel! {
    function: add_mul_2_limbs_unchecked;
    kernel: AddMul2Kernel;
    surface: provider;
    backends: [
        x86 => all(not(miri), target_arch = "x86", target_pointer_width = "32"),
        aarch64 => all(not(miri), target_arch = "aarch64", target_pointer_width = "64"),
        arm => all(not(miri), target_arch = "arm", target_feature = "v6", not(target_feature = "thumb-mode")),
        powerpc => all(not(miri), target_arch = "powerpc"),
        s390x => all(not(miri), target_arch = "s390x"),
        riscv64 => all(not(miri), target_arch = "riscv64", target_pointer_width = "64", target_feature = "m"),
        riscv32 => all(not(miri), target_arch = "riscv32", target_feature = "m"),
        loongarch64 => all(not(miri), target_arch = "loongarch64", target_pointer_width = "64"),
        loongarch32 => all(not(miri), target_arch = "loongarch32"),
        mips64 => all(not(miri), target_arch = "mips64"),
        mips => all(not(miri), target_arch = "mips"),
    ];
    x86_64: [bmi2];
    powerpc64: [power8, power9];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
        target_arch = "powerpc64",
    ];
    fallback_imports: [DoubleLimb, LIMB_BITS];
    runtime_backends: [
        add_mul_2_limbs_vanilla_backend => x86_64,
        add_mul_2_limbs_bmi2_backend => x86_64_bmi2,
    ];
}

#[cfg(test)]
mod tests;
