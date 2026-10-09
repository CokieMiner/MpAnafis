//! Left shifts into a separate write-only destination.
//!
//! x86-64 selects between a 512-bit AVX-512 loop, a 256-bit AVX2 loop, and
//! the mandatory SSE2 baseline (`psllq`/`psrlq`/`por`), with the SSE2 tier
//! also covering compile-time builds without AVX2. `AArch64` uses two-limb
//! NEON shifts and adjacent-limb merges. Other platforms use the Rust fallback.
//! Each output limb is written once; no intermediate copy is required.

#![expect(
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
    x86_64: [sse2, avx2, avx512, small];
    powerpc64: [];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
    ];
    fallback_imports: [LIMB_BITS];
}

#[cfg(test)]
mod tests;
