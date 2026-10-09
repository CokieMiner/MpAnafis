//! Right shifts into a separate write-only destination.
//!
//! x86-64 selects between a 512-bit AVX-512 loop, a 256-bit AVX2 loop, and
//! the mandatory SSE2 baseline (`psllq`/`psrlq`/`por`); aarch64 uses two-limb
//! NEON shifts and adjacent-limb merges. Other platforms use the pure
//! Rust fallback. The kernels write `dst[0..len] = src[0..len] >> shift` in a
//! single pass.

#![expect(
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
    x86_64: [sse2, avx2, avx512, small];
    powerpc64: [];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
    ];
    fallback_imports: [LIMB_BITS];
}

#[cfg(test)]
mod tests;
