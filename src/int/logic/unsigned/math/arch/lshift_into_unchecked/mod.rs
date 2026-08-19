//! Architecture-specific optimised shift kernels.
//!
//! x86-64 selects between a 256-bit AVX2 loop and the mandatory SSE2 baseline
//! (`psllq`/`psrlq`/`por`), with the SSE2 tier also covering compile-time
//! builds without AVX2. `AArch64` uses a hand-tuned `lsl`+`lsr`+`orr`
//! sequence
//! (`extr` requires an immediate, but our shift counts are runtime values).
//! All other platforms use the pure Rust fallback. The in-place kernels shift
//! one writable span; the `into` variants write a separate destination in a
//! single pass, which is what keeps allocating shifts at GMP speed instead of
//! copying and shifting in place.

#![allow(
    unsafe_code,
    reason = "Hardware inline assembly natively requires unsafe code"
)]

select_arch_kernel! {
    function: lshift_into_unchecked;
    kernel: LshiftIntoKernel;
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
    test_backends: [
        lshift_into_sse2_test => x86_64,
        lshift_into_avx2_test => x86_64_avx2,
    ];
}
