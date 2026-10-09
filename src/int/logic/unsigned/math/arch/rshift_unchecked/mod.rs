//! In-place right shifts with low-bit carry extraction.
//!
//! x86-64 selects between a 256-bit AVX2 loop and a scalar `shrdq` baseline;
//! `aarch64` uses an `lsl`+`lsr`+`orr` sequence (`extr` requires an
//! immediate, but the shift counts are runtime values). All other platforms
//! use the pure Rust fallback.

#![expect(
    unsafe_code,
    reason = "Hardware inline assembly natively requires unsafe code"
)]

select_arch_kernel! {
    function: rshift_unchecked;
    kernel: RshiftKernel;
    surface: selector;
    backends: [
        aarch64 => all(not(miri), target_arch = "aarch64", target_pointer_width = "64"),
    ];
    x86_64: [sse2, avx2];
    powerpc64: [];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
    ];
    fallback_imports: [LIMB_BITS];
}

#[cfg(test)]
mod tests;
