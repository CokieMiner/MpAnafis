//! Architecture-selected complete schoolbook multiplication kernel.
//!
//! Runtime-selected x86-64 builds dispatch once for the whole quadratic
//! product, keeping both initialization and accumulation as direct calls
//! inside the selected backend.

#![expect(
    unsafe_code,
    reason = "The kernel operates on caller-validated raw product spans"
)]

select_arch_kernel! {
    function: mul_basecase_unchecked;
    surface: composite;
    x86_64: [bmi2, adx_bmi2];
}

#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(target_feature = "adx", target_feature = "bmi2"))
))]
pub use self::{
    portable::{mul_4x4_portable_unchecked, mul_8x8_portable_unchecked},
    x86_64_adx::{
        add_mul_4_limbs_unchecked as add_mul_4_adx, add_mul_5_limbs_unchecked as add_mul_5_adx,
        add_mul_6_limbs_unchecked as add_mul_6_adx, add_mul_7_limbs_unchecked as add_mul_7_adx,
        add_mul_8_limbs_unchecked as add_mul_8_adx, add_mul_9_limbs_unchecked as add_mul_9_adx,
        add_mul_10_limbs_unchecked as add_mul_10_adx, add_mul_11_limbs_unchecked as add_mul_11_adx,
        add_mul_12_limbs_unchecked as add_mul_12_adx, add_mul_13_limbs_unchecked as add_mul_13_adx,
        mul_2x4_limbs_unchecked as mul_2x4_adx, mul_2x5_limbs_unchecked as mul_2x5_adx,
        mul_2x6_limbs_unchecked as mul_2x6_adx, mul_2x7_limbs_unchecked as mul_2x7_adx,
        mul_2x8_limbs_unchecked as mul_2x8_adx, mul_2x9_limbs_unchecked as mul_2x9_adx,
        mul_2x10_limbs_unchecked as mul_2x10_adx, mul_2x11_limbs_unchecked as mul_2x11_adx,
        mul_2x12_limbs_unchecked as mul_2x12_adx, mul_2x13_limbs_unchecked as mul_2x13_adx,
        mul_4x4_adx_unchecked as mul_4x4_adx, mul_8x8_adx_unchecked as mul_8x8_adx,
    },
    x86_64_adx_tail::{
        add_mul_14_limbs_unchecked as add_mul_14_adx, add_mul_15_limbs_unchecked as add_mul_15_adx,
        add_mul_16_limbs_unchecked as add_mul_16_adx, add_mul_17_limbs_unchecked as add_mul_17_adx,
    },
};

#[cfg(test)]
mod tests;
