//! Architecture-selected complete schoolbook squaring kernel.
//!
//! Runtime-selected x86-64 builds dispatch once for the whole squaring
//! operation: the dual-chain ADX kernel on ADX+BMI2 hosts, the `mulx`+`adcq`
//! BMI2 kernel on BMI2-only hosts, and the portable unrolled driver elsewhere.

#![expect(
    unsafe_code,
    reason = "The kernel operates on caller-validated raw square spans"
)]

select_arch_kernel! {
    function: sqr_basecase_unchecked;
    surface: composite;
    x86_64: [adx_bmi2, bmi2];
}

#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "bmi2")
))]
pub use direct::sqr_basecase_unchecked as portable_square;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "bmi2")
))]
pub use x86_64_adx::sqr_basecase_unchecked as adx_square;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "bmi2")
))]
pub use x86_64_bmi2::sqr_basecase_unchecked as bmi2_square;

#[cfg(test)]
mod tests;
