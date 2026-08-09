//! Architecture-selected complete schoolbook multiplication kernel.
//!
//! Runtime-selected x86-64 builds dispatch once for the whole quadratic
//! product, keeping both initialization and accumulation as direct calls
//! inside the selected backend.

#![allow(
    unsafe_code,
    reason = "The kernel operates on caller-validated raw product spans"
)]

select_arch_kernel! {
    function: mul_basecase_unchecked;
    surface: composite;
    x86_64: [bmi2, adx_bmi2];
}
