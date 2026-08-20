# Archived: 50-bit Floating-Point Harvey NTT Engine

This folder preserves the complete 3-prime floating-point Harvey NTT multiplication engine (`ntt_float_f64` architecture kernels and `mul/ntt` digit pipeline), archived in favor of the unified and higher-performing Fermat-ring Schönhage-Strassen Algorithm (SSA).

## Architecture Preserved
- `arch/ntt_float_f64/`: AVX2+FMA (x86-64) and NEON (AArch64) vectorized 50-bit floating-point butterfly kernels with zero on-the-fly twiddle squaring.
- `mul/ntt/`: 3-prime Harvey NTT convolution pipeline ($P_1, P_2, P_3 < 2^{50}$) with truncated scaling (TFT) and Garner Chinese Remainder Theorem reconstruction.
