//! Overlap-safe left shift from a limb prefix into the same or a higher suffix.
//!
//! This is the one-pass primitive used when an SSA coefficient's whole-limb
//! displacement makes its source and destination overlap. Backends traverse
//! from high limbs to low limbs, so every source is consumed before a higher
//! destination store can overwrite it.

#![expect(
    unsafe_code,
    reason = "Raw overlapping spans and SIMD intrinsics require unsafe operations"
)]

select_arch_kernel! {
    function: lshift_overlapping_unchecked;
    kernel: LshiftOverlappingKernel;
    surface: selector;
    backends: [];
    x86_64: [sse2, avx2, avx512];
    powerpc64: [];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
    ];
    fallback_imports: [];
    test_backends: [
        lshift_overlapping_sse2_test => x86_64,
        lshift_overlapping_avx2_test => x86_64_avx2,
        lshift_overlapping_avx512_test => x86_64_avx512,
    ];
}
