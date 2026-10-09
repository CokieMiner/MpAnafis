//! Architecture-selected single-limb multiply-subtract kernel.

#![expect(
    unsafe_code,
    reason = "raw limb spans and architecture assembly implement the multiply-subtract contract"
)]

use super::Limb;

select_arch_kernel! {
    function: sub_mul_limbs_unchecked;
    kernel: SubMulKernel;
    surface: provider;
    backends: [
        x86 => all(not(miri), target_arch = "x86", target_pointer_width = "32"),
        aarch64 => all(not(miri), target_arch = "aarch64", target_pointer_width = "64"),
        arm => all(not(miri), target_arch = "arm", not(target_feature = "thumb-mode")),
        powerpc => all(not(miri), target_arch = "powerpc"),
        s390x => all(not(miri), target_arch = "s390x"),
        riscv64 => all(not(miri), target_arch = "riscv64", target_feature = "m"),
        riscv32 => all(not(miri), target_arch = "riscv32", target_feature = "m"),
        loongarch64 => all(not(miri), target_arch = "loongarch64"),
        loongarch32 => all(not(miri), target_arch = "loongarch32"),
        mips64 => all(not(miri), target_arch = "mips64"),
        mips => all(not(miri), target_arch = "mips"),
    ];
    x86_64: [bmi2, adx_bmi2];
    powerpc64: [baseline];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
        target_arch = "powerpc64",
    ];
    fallback_imports: [DoubleLimb, LIMB_BITS];
    runtime_backends: [];
}

#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(target_feature = "adx", target_feature = "bmi2"))
))]
pub use self::{
    x86_64::sub_mul_limbs_unchecked as fallback_kernel,
    x86_64_adx::sub_mul_limbs_unchecked as adx_kernel,
    x86_64_bmi2::sub_mul_limbs_unchecked as bmi2_kernel,
};

#[cfg(test)]
mod tests;
