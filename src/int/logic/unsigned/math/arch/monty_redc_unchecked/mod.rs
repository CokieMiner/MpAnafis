//! Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction kernels.

#![expect(
    unsafe_code,
    reason = "Hardware inline assembly and raw pointer manipulation require unsafe code"
)]

select_arch_kernel! {
    function: monty_redc_step_unchecked;
    kernel: MontyKernel;
    surface: selector;
    backends: [
        aarch64 => all(not(miri), target_arch = "aarch64", target_pointer_width = "64"),
        loongarch64 => all(not(miri), target_arch = "loongarch64", target_pointer_width = "64"),
        riscv64 => all(not(miri), target_arch = "riscv64", target_pointer_width = "64", target_feature = "m"),
        s390x => all(not(miri), target_arch = "s390x"),
    ];
    x86_64: [fallback, bmi2, adx_bmi2];
    powerpc64: [baseline];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
        target_arch = "powerpc64",
    ];
    fallback_imports: [DoubleLimb, LIMB_BITS];
}

#[cfg(test)]
mod tests;
