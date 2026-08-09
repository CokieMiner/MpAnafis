//! Architecture-specific optimised out-of-place right-shift kernels.
//!
//! x86-64 selects between a 256-bit AVX2 loop and the mandatory SSE2 baseline
//! (`psllq`/`psrlq`/`por`); aarch64 uses a hand-tuned `lsl`+`lsr`+`orr`
//! sequence (`extr` requires an immediate, but our shift counts are runtime
//! values). All other platforms use the pure Rust fallback. The kernels write
//! `dst[0..len] = src[0..len] >> shift` in a single pass.

#![allow(
    unsafe_code,
    reason = "Hardware inline assembly natively requires unsafe code"
)]

select_arch_kernel! {
    function: rshift_into_unchecked;
    kernel: RshiftIntoKernel;
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
        rshift_into_sse2_test => x86_64,
        rshift_into_avx2_test => x86_64_avx2,
    ];
}
