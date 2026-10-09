//! Cross-limb shifted-high subtraction kernel.
//!
//! BMI2, `AArch64`, s390x and PowerPC backends preserve borrow through shifts,
//! bit merges and loop control. Other targets use the portable limb recurrence.

#![expect(
    unsafe_code,
    reason = "raw limb spans and target assembly implement shifted-source subtraction"
)]

select_arch_kernel! {
    function: sub_shifted_high_limbs_unchecked;
    kernel: SubShiftedHighKernel;
    surface: selector;
    backends: [
        aarch64 => all(not(miri), target_arch = "aarch64", target_pointer_width = "64"),
        s390x => all(not(miri), target_arch = "s390x"),
        powerpc => all(not(miri), target_arch = "powerpc", target_pointer_width = "32"),
    ];
    x86_64: [fallback, bmi2];
    powerpc64: [baseline];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
        all(target_arch = "powerpc64", target_pointer_width = "64"),
    ];
    fallback_imports: [];
}

#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "bmi2")
))]
pub use self::{
    fallback::sub_shifted_high_limbs_unchecked as fallback_kernel,
    x86_64_bmi2::sub_shifted_high_limbs_unchecked as bmi2_kernel,
};

#[cfg(test)]
mod tests;
