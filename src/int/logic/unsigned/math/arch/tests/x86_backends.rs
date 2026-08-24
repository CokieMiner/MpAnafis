//! Cross-backend agreement properties for x86-64 runtime-dispatched kernels.

#![expect(
    unsafe_code,
    clippy::indexing_slicing,
    clippy::unreachable,
    reason = "The properties call unsafe x86 backends with owned buffers and slice exact-length generated seeds; proptest generators restrict match arms"
)]

use std::arch::is_x86_feature_detected;

use proptest::prelude::*;

use super::{
    equal_length_limb_vecs, equal_length_odd_limb_vecs, exact_limb_vec, montgomery_inverse,
    reference_multiply_two,
};
use crate::int::{
    logic::math::arch::{
        add_mul_2_limbs_unchecked::{
            add_mul_2_limbs_bmi2_test as add_mul_2_bmi2,
            add_mul_2_limbs_vanilla_test as add_mul_2_vanilla,
        },
        add_mul_limbs_unchecked::{
            add_mul_limbs_adx_test as add_mul_adx, add_mul_limbs_bmi2_test as add_mul_bmi2,
            add_mul_limbs_vanilla_test as add_mul_vanilla,
        },
        lshift_into_unchecked::{
            lshift_into_avx2_test as lshift_into_avx2,
            lshift_into_avx512_test as lshift_into_avx512,
            lshift_into_sse2_test as lshift_into_sse2,
        },
        lshift_overlapping_unchecked::{
            lshift_overlapping_avx2_test as lshift_overlapping_avx2,
            lshift_overlapping_avx512_test as lshift_overlapping_avx512,
            lshift_overlapping_sse2_test as lshift_overlapping_sse2,
        },
        lshift_unchecked::{lshift_avx2_test as lshift_avx2, lshift_sse2_test as lshift_sse2},
        monty_redc_unchecked::{
            monty_redc_step_adx_test as monty_adx, monty_redc_step_bmi2_test as monty_bmi2,
            monty_redc_step_fallback_test as monty_fallback,
        },
        mul_2_limbs_unchecked::{
            mul_2_limbs_bmi2_test as mul_2_bmi2, mul_2_limbs_vanilla_test as mul_2_vanilla,
        },
        mul_basecase_unchecked::{
            x86_64_adx::{
                add_mul_4_limbs_unchecked as add_mul_4_adx,
                add_mul_5_limbs_unchecked as add_mul_5_adx,
                add_mul_6_limbs_unchecked as add_mul_6_adx,
                add_mul_7_limbs_unchecked as add_mul_7_adx,
                add_mul_8_limbs_unchecked as add_mul_8_adx,
                add_mul_9_limbs_unchecked as add_mul_9_adx,
                add_mul_10_limbs_unchecked as add_mul_10_adx,
                add_mul_11_limbs_unchecked as add_mul_11_adx,
                add_mul_12_limbs_unchecked as add_mul_12_adx,
                add_mul_13_limbs_unchecked as add_mul_13_adx,
                mul_2x4_limbs_unchecked as mul_2x4_adx, mul_2x5_limbs_unchecked as mul_2x5_adx,
                mul_2x6_limbs_unchecked as mul_2x6_adx, mul_2x7_limbs_unchecked as mul_2x7_adx,
                mul_2x8_limbs_unchecked as mul_2x8_adx, mul_2x9_limbs_unchecked as mul_2x9_adx,
                mul_2x10_limbs_unchecked as mul_2x10_adx, mul_2x11_limbs_unchecked as mul_2x11_adx,
                mul_2x12_limbs_unchecked as mul_2x12_adx, mul_2x13_limbs_unchecked as mul_2x13_adx,
            },
            x86_64_adx_tail::{
                add_mul_14_limbs_unchecked as add_mul_14_adx,
                add_mul_15_limbs_unchecked as add_mul_15_adx,
                add_mul_16_limbs_unchecked as add_mul_16_adx,
                add_mul_17_limbs_unchecked as add_mul_17_adx,
            },
        },
        rshift_into_unchecked::{
            rshift_into_avx2_test as rshift_into_avx2,
            rshift_into_avx512_test as rshift_into_avx512,
            rshift_into_sse2_test as rshift_into_sse2,
        },
        rshift_unchecked::{rshift_avx2_test as rshift_avx2, rshift_sse2_test as rshift_sse2},
        sub_mul_limbs_unchecked::{
            sub_mul_limbs_adx_test as sub_mul_adx, sub_mul_limbs_bmi2_test as sub_mul_bmi2,
            sub_mul_limbs_vanilla_test as sub_mul_vanilla,
        },
    },
    types::Limb,
};

/// Whether this host can execute the kernels these properties call.
///
/// The properties bypass runtime dispatch deliberately -- comparing the
/// backends against each other is the point -- so on a CPU lacking these
/// features the calls would execute an unsupported instruction rather than
/// fall back to a legal one. The module is selected by `target_arch`, which
/// says nothing about ADX or BMI2, and neither is a default target feature of
/// `x86_64-unknown-linux-gnu`, so pre-Broadwell hosts reach this code.
///
/// Deliberately one combined check rather than one per feature. The few
/// properties that exercise only BMI2 are also skipped on a BMI2-without-ADX
/// host such as Haswell, which costs a little coverage on hardware a decade
/// old and keeps eight call sites honest about a single precondition.
fn host_executes_dispatch_backends() -> bool {
    is_x86_feature_detected!("adx") && is_x86_feature_detected!("bmi2")
}

/// Whether this host can execute the SIMD shift kernels.
///
/// Same reasoning as `host_executes_dispatch_backends`: the shift properties
/// bypass runtime dispatch to compare the SSE2 and AVX2 tiers directly, so
/// they must not run on a host whose CPUID does not report AVX2 (e.g. a VM
/// with the feature masked).
fn host_executes_simd_backends() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Whether this host can execute the AVX-512 shift tiers.
///
/// The AVX-512 properties bypass runtime dispatch to compare the 512-bit
/// backends against the AVX2 and SSE2 tiers directly, so they must not run on
/// a host whose CPUID does not report `avx512f` — including AVX2-only VMs and
/// Zen 3 or older silicon.
fn host_executes_avx512_backends() -> bool {
    is_x86_feature_detected!("avx512f")
}

proptest! {
    #[test]
    fn prop_all_x86_addmul_backends_agree(
        case in equal_length_limb_vecs(0..=16),
        scalar in any::<Limb>(),
    ) {
        if !host_executes_dispatch_backends() {
            return Ok(());
        }

        let (src, dst) = case;
        let len = src.len();
        let mut dst_adx = dst.clone();
        let mut dst_bmi2 = dst.clone();
        let mut dst_vanilla = dst;

        // SAFETY: every destination and `src` contains exactly `len` initialized limbs.
        let carry_adx = unsafe { add_mul_adx(dst_adx.as_mut_ptr(), src.as_ptr(), len, scalar) };
        // SAFETY: every destination and `src` contains exactly `len` initialized limbs.
        let carry_bmi2 = unsafe { add_mul_bmi2(dst_bmi2.as_mut_ptr(), src.as_ptr(), len, scalar) };
        // SAFETY: every destination and `src` contains exactly `len` initialized limbs.
        let carry_vanilla = unsafe {
            add_mul_vanilla(dst_vanilla.as_mut_ptr(), src.as_ptr(), len, scalar)
        };

        prop_assert_eq!((carry_adx, &dst_adx), (carry_bmi2, &dst_bmi2), "addmul_1: ADX != BMI2");
        prop_assert_eq!((carry_adx, &dst_adx), (carry_vanilla, &dst_vanilla), "addmul_1: ADX != vanilla");
    }

    #[test]
    fn prop_x86_fixed_8_addmul_matches_generic_adx(
        case in equal_length_limb_vecs(8..=8),
        scalar in any::<Limb>(),
    ) {
        if !host_executes_dispatch_backends() {
            return Ok(());
        }

        let (src, dst) = case;
        let mut dst_fixed = dst.clone();
        let mut dst_generic = dst;

        // SAFETY: both destinations and `src` contain exactly eight initialized limbs.
        let carry_fixed = unsafe { add_mul_8_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar) };
        // SAFETY: both destinations and `src` contain exactly eight initialized limbs.
        let carry_generic = unsafe { add_mul_adx(dst_generic.as_mut_ptr(), src.as_ptr(), 8, scalar) };

        prop_assert_eq!(
            (carry_fixed, &dst_fixed),
            (carry_generic, &dst_generic),
            "fixed eight-limb ADX row != generic ADX row",
        );
    }

    #[test]
    fn prop_x86_fixed_small_addmul_rows_match_generic_adx(
        len in 4_usize..=17,
        src_seed in exact_limb_vec(17),
        dst_seed in exact_limb_vec(17),
        scalar in any::<Limb>(),
    ) {
        if !host_executes_dispatch_backends() {
            return Ok(());
        }

        let src = &src_seed[..len];
        let mut dst_fixed = dst_seed[..len].to_vec();
        let mut dst_generic = dst_fixed.clone();

        // SAFETY: both destinations and `src` contain exactly `len` limbs;
        // every match arm encodes that same fixed width.
        let carry_fixed = unsafe {
            match len {
                4 => add_mul_4_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                5 => add_mul_5_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                6 => add_mul_6_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                7 => add_mul_7_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                8 => add_mul_8_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                9 => add_mul_9_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                10 => add_mul_10_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                11 => add_mul_11_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                12 => add_mul_12_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                13 => add_mul_13_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                14 => add_mul_14_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                15 => add_mul_15_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                16 => add_mul_16_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                17 => add_mul_17_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), scalar),
                _ => unreachable!("generator restricts fixed ADX rows to 4..=17"),
            }
        };
        // SAFETY: both destination and source contain exactly `len` initialized limbs.
        let carry_generic = unsafe {
            add_mul_adx(dst_generic.as_mut_ptr(), src.as_ptr(), len, scalar)
        };

        prop_assert_eq!(
            (carry_fixed, &dst_fixed),
            (carry_generic, &dst_generic),
            "fixed ADX row != generic ADX row at len={}",
            len,
        );
    }

    #[test]
    fn prop_x86_fixed_two_row_initializers_match_generic_bmi2(
        len in 4_usize..=13,
        src_seed in exact_limb_vec(13),
        dirty_seed in exact_limb_vec(15),
        scalars in (any::<Limb>(), any::<Limb>()),
    ) {
        if !host_executes_dispatch_backends() {
            return Ok(());
        }

        let src = &src_seed[..len];
        let dst_len = len.wrapping_add(2);
        let mut dst_fixed = dirty_seed[..dst_len].to_vec();
        let mut dst_generic = dst_fixed.clone();
        let (low_scalar, high_scalar) = scalars;

        // SAFETY: `src` has exactly `len` limbs and each destination has
        // `len + 2`; every match arm encodes that same fixed width.
        unsafe {
            match len {
                4 => mul_2x4_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), low_scalar, high_scalar),
                5 => mul_2x5_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), low_scalar, high_scalar),
                6 => mul_2x6_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), low_scalar, high_scalar),
                7 => mul_2x7_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), low_scalar, high_scalar),
                8 => mul_2x8_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), low_scalar, high_scalar),
                9 => mul_2x9_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), low_scalar, high_scalar),
                10 => mul_2x10_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), low_scalar, high_scalar),
                11 => mul_2x11_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), low_scalar, high_scalar),
                12 => mul_2x12_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), low_scalar, high_scalar),
                13 => mul_2x13_adx(dst_fixed.as_mut_ptr(), src.as_ptr(), low_scalar, high_scalar),
                _ => unreachable!("generator restricts fixed two-row kernels to 4..=13"),
            }
            mul_2_bmi2(
                dst_generic.as_mut_ptr(),
                src.as_ptr(),
                len,
                low_scalar,
                high_scalar,
            );
        }

        prop_assert_eq!(
            dst_fixed,
            dst_generic,
            "fixed two-row initializer != generic BMI2 at len={}",
            len,
        );
    }

    #[test]
    fn prop_all_x86_submul_backends_agree(
        case in equal_length_limb_vecs(0..=16),
        scalar in any::<Limb>(),
    ) {
        if !host_executes_dispatch_backends() {
            return Ok(());
        }

        let (src, dst) = case;
        let len = src.len();
        let mut dst_adx = dst.clone();
        let mut dst_bmi2 = dst.clone();
        let mut dst_vanilla = dst;

        // SAFETY: every destination and `src` contains exactly `len` initialized limbs.
        let result_adx = unsafe { sub_mul_adx(dst_adx.as_mut_ptr(), src.as_ptr(), len, scalar) };
        // SAFETY: every destination and `src` contains exactly `len` initialized limbs.
        let result_bmi2 = unsafe { sub_mul_bmi2(dst_bmi2.as_mut_ptr(), src.as_ptr(), len, scalar) };
        // SAFETY: every destination and `src` contains exactly `len` initialized limbs.
        let result_vanilla = unsafe {
            sub_mul_vanilla(dst_vanilla.as_mut_ptr(), src.as_ptr(), len, scalar)
        };

        prop_assert_eq!((result_adx, &dst_adx), (result_bmi2, &dst_bmi2), "submul_1: ADX != BMI2");
        prop_assert_eq!((result_adx, &dst_adx), (result_vanilla, &dst_vanilla), "submul_1: ADX != vanilla");
    }

    #[test]
    fn prop_x86_addmul_2_backends_agree(
        len in 0_usize..=16,
        src_seed in exact_limb_vec(16),
        dst_seed in exact_limb_vec(17),
        scalars in (any::<Limb>(), any::<Limb>()),
    ) {
        if !host_executes_dispatch_backends() {
            return Ok(());
        }

        let src = src_seed[..len].to_vec();
        let dst_len = len.wrapping_add(1);
        let dst = dst_seed[..dst_len].to_vec();
        let (low_scalar, high_scalar) = scalars;
        let mut dst_bmi2 = dst.clone();
        let mut dst_vanilla = dst;

        // SAFETY: each destination has `len + 1` limbs and `src` has `len` limbs.
        let carry_bmi2 = unsafe {
            add_mul_2_bmi2(
                dst_bmi2.as_mut_ptr(),
                src.as_ptr(),
                len,
                low_scalar,
                high_scalar,
            )
        };
        // SAFETY: each destination has `len + 1` limbs and `src` has `len` limbs.
        let carry_vanilla = unsafe {
            add_mul_2_vanilla(
                dst_vanilla.as_mut_ptr(),
                src.as_ptr(),
                len,
                low_scalar,
                high_scalar,
            )
        };

        prop_assert_eq!((carry_bmi2, &dst_bmi2), (carry_vanilla, &dst_vanilla), "addmul_2: BMI2 != vanilla");
    }

    #[test]
    fn prop_x86_mul_2_backends_agree(
        len in 1_usize..=16,
        src_seed in exact_limb_vec(16),
        dst_seed in exact_limb_vec(18),
        scalars in (any::<Limb>(), any::<Limb>()),
    ) {
        if !host_executes_dispatch_backends() {
            return Ok(());
        }

        let src = src_seed[..len].to_vec();
        let (low_scalar, high_scalar) = scalars;
        let expected = reference_multiply_two(&src, low_scalar, high_scalar);
        let dst_len = len.wrapping_add(2);
        let mut dst_bmi2 = dst_seed[..dst_len].to_vec();
        let mut dst_vanilla = dst_bmi2.clone();

        // SAFETY: each destination has `len + 2` limbs and `src` has `len` limbs.
        unsafe {
            mul_2_bmi2(
                dst_bmi2.as_mut_ptr(),
                src.as_ptr(),
                len,
                low_scalar,
                high_scalar,
            );
        }
        // SAFETY: each destination has `len + 2` limbs and `src` has `len` limbs.
        unsafe {
            mul_2_vanilla(
                dst_vanilla.as_mut_ptr(),
                src.as_ptr(),
                len,
                low_scalar,
                high_scalar,
            );
        }

        prop_assert_eq!(&dst_bmi2, &expected, "mul_2 BMI2 != reference");
        prop_assert_eq!(&dst_vanilla, &expected, "mul_2 vanilla != reference");
    }

    #[test]
    fn prop_x86_64_monty_redc_step_cross_backend_agreement(
        case in equal_length_odd_limb_vecs(1..=16),
        input_limb in any::<Limb>(),
    ) {
        if !host_executes_dispatch_backends() {
            return Ok(());
        }

        let (dst, multiplier, modulus) = case;
        let len = dst.len();
        let inverse = montgomery_inverse(modulus[0]);
        let mut dst_adx = dst.clone();
        let mut dst_bmi2 = dst.clone();
        let mut dst_fallback = dst;

        // SAFETY: every operand contains exactly `len` initialized limbs.
        let carry_adx = unsafe {
            monty_adx(
                dst_adx.as_mut_ptr(), multiplier.as_ptr(), modulus.as_ptr(), len, input_limb, inverse,
            )
        };
        // SAFETY: every operand contains exactly `len` initialized limbs.
        let carry_bmi2 = unsafe {
            monty_bmi2(
                dst_bmi2.as_mut_ptr(), multiplier.as_ptr(), modulus.as_ptr(), len, input_limb, inverse,
            )
        };
        // SAFETY: every operand contains exactly `len` initialized limbs.
        let carry_fallback = unsafe {
            monty_fallback(
                dst_fallback.as_mut_ptr(), multiplier.as_ptr(), modulus.as_ptr(), len, input_limb, inverse,
            )
        };

        prop_assert_eq!((carry_adx, &dst_adx), (carry_fallback, &dst_fallback), "monty_redc_step: ADX != fallback");
        prop_assert_eq!((carry_bmi2, &dst_bmi2), (carry_fallback, &dst_fallback), "monty_redc_step: BMI2 != fallback");
    }

    #[test]
    fn prop_x86_lshift_backends_agree(
        initial in proptest::collection::vec(any::<Limb>(), 0..=64),
        shift in 1_u32..Limb::BITS,
    ) {
        if !host_executes_simd_backends() {
            return Ok(());
        }

        let len = initial.len();
        let mut shifted_sse2 = initial.clone();
        let mut shifted_avx2 = initial;

        // SAFETY: both spans contain exactly `len` initialized limbs and the
        // strategy proves 0 < shift < Limb::BITS.
        let carry_sse2 = unsafe { lshift_sse2(shifted_sse2.as_mut_ptr(), len, shift) };
        // SAFETY: both spans contain exactly `len` initialized limbs and the
        // strategy proves 0 < shift < Limb::BITS.
        let carry_avx2 = unsafe { lshift_avx2(shifted_avx2.as_mut_ptr(), len, shift) };

        prop_assert_eq!(
            (carry_sse2, &shifted_sse2),
            (carry_avx2, &shifted_avx2),
            "lshift in-place: SSE2 != AVX2",
        );
    }

    #[test]
    fn prop_x86_rshift_backends_agree(
        initial in proptest::collection::vec(any::<Limb>(), 0..=64),
        shift in 1_u32..Limb::BITS,
    ) {
        if !host_executes_simd_backends() {
            return Ok(());
        }

        let len = initial.len();
        let mut shifted_sse2 = initial.clone();
        let mut shifted_avx2 = initial;

        // SAFETY: both spans contain exactly `len` initialized limbs and the
        // strategy proves 0 < shift < Limb::BITS.
        let carry_sse2 = unsafe { rshift_sse2(shifted_sse2.as_mut_ptr(), len, shift) };
        // SAFETY: both spans contain exactly `len` initialized limbs and the
        // strategy proves 0 < shift < Limb::BITS.
        let carry_avx2 = unsafe { rshift_avx2(shifted_avx2.as_mut_ptr(), len, shift) };

        prop_assert_eq!(
            (carry_sse2, &shifted_sse2),
            (carry_avx2, &shifted_avx2),
            "rshift in-place: SSE2 != AVX2",
        );
    }

    #[test]
    fn prop_x86_lshift_into_backends_agree(
        initial in proptest::collection::vec(any::<Limb>(), 0..=64),
        shift in 1_u32..Limb::BITS,
    ) {
        if !host_executes_simd_backends() {
            return Ok(());
        }

        let len = initial.len();
        let mut dst_sse2 = vec![0; len];
        let mut dst_avx2 = vec![0; len];

        // SAFETY: each destination is writable for `len` limbs, `initial`
        // holds that many readable limbs, the spans are disjoint, and the
        // strategy proves 0 < shift < Limb::BITS.
        let carry_sse2 = unsafe { lshift_into_sse2(dst_sse2.as_mut_ptr(), initial.as_ptr(), len, shift) };
        // SAFETY: each destination is writable for `len` limbs, `initial`
        // holds that many readable limbs, the spans are disjoint, and the
        // strategy proves 0 < shift < Limb::BITS.
        let carry_avx2 = unsafe { lshift_into_avx2(dst_avx2.as_mut_ptr(), initial.as_ptr(), len, shift) };

        prop_assert_eq!(
            (carry_sse2, &dst_sse2),
            (carry_avx2, &dst_avx2),
            "lshift into: SSE2 != AVX2",
        );
    }

    #[test]
    fn prop_x86_lshift_overlapping_backends_agree(
        mut source in proptest::collection::vec(any::<Limb>(), 0..=129),
        offset in 0_usize..=64,
        dirty in any::<Limb>(),
        shift in 1_u32..Limb::BITS,
    ) {
        if !host_executes_simd_backends() {
            return Ok(());
        }

        let len = source.len();
        source.extend(vec![dirty; offset]);
        let mut shifted_sse2 = source;
        let mut shifted_avx2 = shifted_sse2.clone();

        // SAFETY: both buffers contain exactly offset + len initialized writable
        // limbs, and the strategy proves 0 < shift < Limb::BITS.
        let carry_sse2 = unsafe {
            lshift_overlapping_sse2(shifted_sse2.as_mut_ptr(), len, offset, shift)
        };
        // SAFETY: the same complete-span and shift proof applies, and the host
        // feature guard above admits AVX2.
        let carry_avx2 = unsafe {
            lshift_overlapping_avx2(shifted_avx2.as_mut_ptr(), len, offset, shift)
        };

        prop_assert_eq!(
            (carry_sse2, &shifted_sse2),
            (carry_avx2, &shifted_avx2),
            "overlapping lshift: SSE2 != AVX2",
        );
    }

    #[test]
    fn prop_x86_rshift_into_backends_agree(
        initial in proptest::collection::vec(any::<Limb>(), 0..=64),
        shift in 1_u32..Limb::BITS,
    ) {
        if !host_executes_simd_backends() {
            return Ok(());
        }

        let len = initial.len();
        let mut dst_sse2 = vec![0; len];
        let mut dst_avx2 = vec![0; len];

        // SAFETY: each destination is writable for `len` limbs, `initial`
        // holds that many readable limbs, the spans are disjoint, and the
        // strategy proves 0 < shift < Limb::BITS.
        let carry_sse2 = unsafe { rshift_into_sse2(dst_sse2.as_mut_ptr(), initial.as_ptr(), len, shift) };
        // SAFETY: each destination is writable for `len` limbs, `initial`
        // holds that many readable limbs, the spans are disjoint, and the
        // strategy proves 0 < shift < Limb::BITS.
        let carry_avx2 = unsafe { rshift_into_avx2(dst_avx2.as_mut_ptr(), initial.as_ptr(), len, shift) };

        prop_assert_eq!(
            (carry_sse2, &dst_sse2),
            (carry_avx2, &dst_avx2),
            "rshift into: SSE2 != AVX2",
        );
    }

    #[test]
    fn prop_x86_lshift_into_avx512_agrees(
        initial in proptest::collection::vec(any::<Limb>(), 0..=64),
        shift in 1_u32..Limb::BITS,
    ) {
        if !host_executes_avx512_backends() {
            return Ok(());
        }

        let len = initial.len();
        let mut dst_avx2 = vec![0; len];
        let mut dst_avx512 = vec![0; len];

        // SAFETY: each destination is writable for `len` limbs, `initial`
        // holds that many readable limbs, the spans are disjoint, and the
        // strategy proves 0 < shift < Limb::BITS.
        let carry_avx2 = unsafe { lshift_into_avx2(dst_avx2.as_mut_ptr(), initial.as_ptr(), len, shift) };
        // SAFETY: the same span, aliasing, and shift preconditions as the
        // AVX2 call above.
        let carry_avx512 =
            unsafe { lshift_into_avx512(dst_avx512.as_mut_ptr(), initial.as_ptr(), len, shift) };

        prop_assert_eq!(
            (carry_avx2, &dst_avx2),
            (carry_avx512, &dst_avx512),
            "lshift into: AVX2 != AVX-512",
        );
    }

    #[test]
    fn prop_x86_rshift_into_avx512_agrees(
        initial in proptest::collection::vec(any::<Limb>(), 0..=64),
        shift in 1_u32..Limb::BITS,
    ) {
        if !host_executes_avx512_backends() {
            return Ok(());
        }

        let len = initial.len();
        let mut dst_avx2 = vec![0; len];
        let mut dst_avx512 = vec![0; len];

        // SAFETY: each destination is writable for `len` limbs, `initial`
        // holds that many readable limbs, the spans are disjoint, and the
        // strategy proves 0 < shift < Limb::BITS.
        let carry_avx2 = unsafe { rshift_into_avx2(dst_avx2.as_mut_ptr(), initial.as_ptr(), len, shift) };
        // SAFETY: the same span, aliasing, and shift preconditions as the
        // AVX2 call above.
        let carry_avx512 =
            unsafe { rshift_into_avx512(dst_avx512.as_mut_ptr(), initial.as_ptr(), len, shift) };

        prop_assert_eq!(
            (carry_avx2, &dst_avx2),
            (carry_avx512, &dst_avx512),
            "rshift into: AVX2 != AVX-512",
        );
    }

    #[test]
    fn prop_x86_lshift_overlapping_avx512_agrees(
        mut source in proptest::collection::vec(any::<Limb>(), 0..=129),
        offset in 0_usize..=64,
        dirty in any::<Limb>(),
        shift in 1_u32..Limb::BITS,
    ) {
        if !host_executes_avx512_backends() {
            return Ok(());
        }

        let len = source.len();
        source.extend(vec![dirty; offset]);
        let mut shifted_avx2 = source;
        let mut shifted_avx512 = shifted_avx2.clone();

        // SAFETY: both buffers contain exactly offset + len initialized writable
        // limbs, and the strategy proves 0 < shift < Limb::BITS.
        let carry_avx2 = unsafe {
            lshift_overlapping_avx2(shifted_avx2.as_mut_ptr(), len, offset, shift)
        };
        // SAFETY: the same span and shift proof applies, and the feature guard
        // above admits AVX-512F.
        let carry_avx512 = unsafe {
            lshift_overlapping_avx512(shifted_avx512.as_mut_ptr(), len, offset, shift)
        };

        prop_assert_eq!(
            (carry_avx2, &shifted_avx2),
            (carry_avx512, &shifted_avx512),
            "overlapping lshift: AVX2 != AVX-512",
        );
    }
}

#[test]
fn overlapping_shift_backends_agree_at_vector_boundaries() {
    if !host_executes_simd_backends() {
        return;
    }

    for len in [
        0_usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    ] {
        for offset in [0_usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63] {
            let source = (0..len)
                .map(|index| {
                    Limb::MAX
                        .wrapping_sub(index.wrapping_mul(0x9E37_79B9))
                        .rotate_left(11)
                })
                .collect::<Vec<_>>();
            for shift in [1, 31, 63] {
                let mut sse2 = source.clone();
                sse2.extend(vec![Limb::MAX; offset]);
                let mut avx2 = sse2.clone();
                let mut avx512 = sse2.clone();

                // SAFETY: every buffer has offset + len initialized writable
                // limbs and each shift lies in 1..64.
                let carry_sse2 =
                    unsafe { lshift_overlapping_sse2(sse2.as_mut_ptr(), len, offset, shift) };
                // SAFETY: the same span proof applies and the host supports AVX2.
                let carry_avx2 =
                    unsafe { lshift_overlapping_avx2(avx2.as_mut_ptr(), len, offset, shift) };
                assert_eq!((carry_avx2, &avx2), (carry_sse2, &sse2));

                if host_executes_avx512_backends() {
                    // SAFETY: the same span proof applies and the host supports
                    // AVX-512F.
                    let carry_avx512 = unsafe {
                        lshift_overlapping_avx512(avx512.as_mut_ptr(), len, offset, shift)
                    };
                    assert_eq!((carry_avx512, &avx512), (carry_sse2, &sse2));
                }
            }
        }
    }
}

#[test]
fn avx512_shift_into_agrees_across_multiple_vector_blocks() {
    if !host_executes_avx512_backends() {
        return;
    }

    for len in [
        0_usize, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 1_023, 1_024, 1_025,
    ] {
        let source = (0..len)
            .map(|index| {
                Limb::MAX
                    .wrapping_sub(index.wrapping_mul(0x9E37_79B9))
                    .rotate_left(11)
            })
            .collect::<Vec<_>>();
        for shift in [1, 31, 63] {
            let mut left_avx2 = vec![0; len];
            let mut left_avx512 = vec![0; len];
            let mut right_avx2 = vec![0; len];
            let mut right_avx512 = vec![0; len];

            // SAFETY: every destination is disjoint from `source`, all spans
            // have exactly `len` limbs, and each shift is in 1..64.
            let left_carry_avx2 =
                unsafe { lshift_into_avx2(left_avx2.as_mut_ptr(), source.as_ptr(), len, shift) };
            // SAFETY: the identical span and shift proof applies; the feature
            // guard above proves this host can execute AVX-512F.
            let left_carry_avx512 = unsafe {
                lshift_into_avx512(left_avx512.as_mut_ptr(), source.as_ptr(), len, shift)
            };
            // SAFETY: every destination is disjoint from `source`, all spans
            // have exactly `len` limbs, and each shift is in 1..64.
            let right_carry_avx2 =
                unsafe { rshift_into_avx2(right_avx2.as_mut_ptr(), source.as_ptr(), len, shift) };
            // SAFETY: the identical span and shift proof applies; the feature
            // guard above proves this host can execute AVX-512F.
            let right_carry_avx512 = unsafe {
                rshift_into_avx512(right_avx512.as_mut_ptr(), source.as_ptr(), len, shift)
            };

            assert_eq!(
                (left_carry_avx512, left_avx512),
                (left_carry_avx2, left_avx2)
            );
            assert_eq!(
                (right_carry_avx512, right_avx512),
                (right_carry_avx2, right_avx2)
            );
        }
    }
}
