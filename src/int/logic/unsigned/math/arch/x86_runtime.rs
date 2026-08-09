//! Process-stable `x86_64` backend selection shared by arithmetic kernels.
//!
//! Two independent feature levels are detected once and cached: the
//! add/multiply backend (`adx`/`bmi2`) and the SIMD tier (`avx2` vs the
//! mandatory `sse2` baseline) used by vector kernels such as limb shifts.
//!
//! Debug builds accept `MP_ANAFIS_TEST_BACKEND=adx|bmi2|vanilla|avx2|sse2`. A
//! requested instruction set is selected only when the host supports it;
//! unsupported or unknown values fall back to the baseline level.

#[cfg(debug_assertions)]
use std::env::var;
use std::{arch::is_x86_feature_detected, sync::OnceLock};

/// CPU feature level available to every runtime-dispatched x86 kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum X86Backend {
    /// ADX and BMI2 are both available.
    AdxBmi2,
    /// ADX is available without BMI2.
    Adx,
    /// BMI2 is available without selecting ADX kernels.
    Bmi2,
    /// Baseline x86-64 instruction set only.
    Baseline,
}

/// SIMD tier available to runtime-dispatched x86-64 vector kernels.
#[cfg(not(target_feature = "avx2"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum X86SimdTier {
    /// AVX2 256-bit vector operations are available.
    Avx2,
    /// SSE2 128-bit vector operations only; mandatory on all x86-64 CPUs.
    Sse2,
}

static BACKEND: OnceLock<X86Backend> = OnceLock::new();

/// SIMD tier selected from CPU features, cached once per process.
#[cfg(not(target_feature = "avx2"))]
static SIMD_TIER: OnceLock<X86SimdTier> = OnceLock::new();

/// Return the process-stable x86 backend selected from CPU features.
#[inline]
pub fn selected_x86_backend() -> X86Backend {
    *BACKEND.get_or_init(detect_x86_backend)
}

/// Return the process-stable SIMD tier selected from CPU features.
#[cfg(not(target_feature = "avx2"))]
#[inline]
pub fn selected_x86_simd_tier() -> X86SimdTier {
    *SIMD_TIER.get_or_init(detect_x86_simd_tier)
}

fn detect_x86_backend() -> X86Backend {
    let has_adx = is_x86_feature_detected!("adx");
    let has_bmi2 = is_x86_feature_detected!("bmi2");

    // Debug builds support deterministic backend testing. Unsupported or
    // unrecognized requests deliberately fall back to the baseline kernel, so
    // the override can never execute an instruction absent from the host CPU.
    #[cfg(debug_assertions)]
    if let Ok(requested) = var("MP_ANAFIS_TEST_BACKEND") {
        return match requested.as_str() {
            "adx" if has_adx && has_bmi2 => X86Backend::AdxBmi2,
            "adx" if has_adx => X86Backend::Adx,
            "bmi2" if has_bmi2 => X86Backend::Bmi2,
            _ => X86Backend::Baseline,
        };
    }

    if has_adx && has_bmi2 {
        X86Backend::AdxBmi2
    } else if has_adx {
        X86Backend::Adx
    } else if has_bmi2 {
        X86Backend::Bmi2
    } else {
        X86Backend::Baseline
    }
}

/// Detect the SIMD tier, honoring the debug-testing override for deterministic
/// vector-kernel selection.
#[cfg(not(target_feature = "avx2"))]
fn detect_x86_simd_tier() -> X86SimdTier {
    let has_avx2 = is_x86_feature_detected!("avx2");

    #[cfg(debug_assertions)]
    if let Ok(requested) = var("MP_ANAFIS_TEST_BACKEND") {
        return match requested.as_str() {
            "avx2" if has_avx2 => X86SimdTier::Avx2,
            _ => X86SimdTier::Sse2,
        };
    }

    if has_avx2 {
        X86SimdTier::Avx2
    } else {
        X86SimdTier::Sse2
    }
}
