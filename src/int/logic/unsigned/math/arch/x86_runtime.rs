//! Cached arithmetic and SIMD feature selection for `x86_64`.
//!
//! Debug builds accept `MP_ANAFIS_TEST_BACKEND=adx|bmi2|vanilla|avx512|avx2|sse2`. A
//! requested instruction set is selected only when the host supports it;
//! unsupported or unknown values fall back to the baseline level.

#[cfg(debug_assertions)]
use std::env::var;
use std::{arch::is_x86_feature_detected, sync::OnceLock};

#[cfg(not(debug_assertions))]
use alloc::string::String;

/// CPU feature level available to every runtime-dispatched x86 kernel.
#[cfg(not(all(target_feature = "adx", target_feature = "bmi2")))]
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum X86SimdTier {
    /// AVX-512 512-bit vector operations are available.
    Avx512,
    /// AVX2 256-bit vector operations are available.
    Avx2,
    /// SSE2 128-bit vector operations only; mandatory on all x86-64 CPUs.
    Sse2,
}

#[cfg(not(all(target_feature = "adx", target_feature = "bmi2")))]
impl X86Backend {
    /// Selects an arithmetic tier from detected features and an optional debug request.
    pub fn from_features(has_adx: bool, has_bmi2: bool, requested: Option<&str>) -> Self {
        if let Some(request) = requested {
            return match request {
                "adx" if has_adx && has_bmi2 => Self::AdxBmi2,
                "adx" if has_adx => Self::Adx,
                "bmi2" if has_bmi2 => Self::Bmi2,
                _ => Self::Baseline,
            };
        }
        match (has_adx, has_bmi2) {
            (true, true) => Self::AdxBmi2,
            (true, false) => Self::Adx,
            (false, true) => Self::Bmi2,
            (false, false) => Self::Baseline,
        }
    }
}

impl X86SimdTier {
    /// Selects a SIMD tier from detected features and an optional debug request.
    ///
    /// The shared AVX-512 tier also requires AVX2 for in-place shift providers.
    pub fn from_features(has_avx2: bool, has_avx512: bool, requested: Option<&str>) -> Self {
        if let Some(request) = requested {
            return match request {
                "avx512" if has_avx2 && has_avx512 => Self::Avx512,
                "avx2" if has_avx2 => Self::Avx2,
                _ => Self::Sse2,
            };
        }
        match (has_avx2, has_avx512) {
            (true, true) => Self::Avx512,
            (true, false) => Self::Avx2,
            (false, _) => Self::Sse2,
        }
    }
}

#[cfg(not(all(target_feature = "adx", target_feature = "bmi2")))]
static BACKEND: OnceLock<X86Backend> = OnceLock::new();

/// SIMD tier selected from CPU features, cached once per process.
static SIMD_TIER: OnceLock<X86SimdTier> = OnceLock::new();

/// Returns the arithmetic tier, detected and cached once per process.
#[cfg(not(all(target_feature = "adx", target_feature = "bmi2")))]
#[inline]
pub fn selected_x86_backend() -> X86Backend {
    *BACKEND.get_or_init(|| {
        let has_adx = is_x86_feature_detected!("adx");
        let has_bmi2 = is_x86_feature_detected!("bmi2");
        #[cfg(debug_assertions)]
        let requested = var("MP_ANAFIS_TEST_BACKEND").ok();
        #[cfg(not(debug_assertions))]
        let requested: Option<String> = None;
        X86Backend::from_features(has_adx, has_bmi2, requested.as_deref())
    })
}

/// Returns the SIMD tier, detected and cached once per process.
#[inline]
pub fn selected_x86_simd_tier() -> X86SimdTier {
    *SIMD_TIER.get_or_init(|| {
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_avx512 = is_x86_feature_detected!("avx512f");
        #[cfg(debug_assertions)]
        let requested = var("MP_ANAFIS_TEST_BACKEND").ok();
        #[cfg(not(debug_assertions))]
        let requested: Option<String> = None;
        X86SimdTier::from_features(has_avx2, has_avx512, requested.as_deref())
    })
}
