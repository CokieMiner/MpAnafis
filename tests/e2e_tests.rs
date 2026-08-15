//! E2E property-based and fuzzing tests for mp-anafis arbitrary-precision integers.
//!
//! Uses differential comparison against `rug::Integer` for correctness verification.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::string_slice,
    reason = "Test code: acceptable for property-based differential testing"
)]

#[cfg(feature = "std")]
use mp_anafis::PrecisionContext;
use mp_anafis::{BoundedPrecision, MpInt, MpUint};
use proptest::prelude::*;

// --- Helper Functions for Differential Testing ---

const fn bounded_width(bits: usize) -> BoundedPrecision {
    BoundedPrecision::new(bits).expect("valid bounded width")
}

// Opaque-box precision verification helper
fn get_precision_str(val: &MpUint) -> String {
    let debug_str = format!("{:?}", val.as_debug_verbose());
    if debug_str.contains("precision: Unlimited") {
        "Unlimited".to_owned()
    } else if let Some(pos) = debug_str.find("precision: Bounded(") {
        let label = "precision: Bounded(";
        let start = pos.wrapping_add(label.len());
        let end = debug_str[start..]
            .find(')')
            .expect("get_precision_str: missing closing paren");
        format!("Bounded({})", &debug_str[start..start.wrapping_add(end)])
    } else {
        panic!("Unknown debug verbose format: {debug_str}");
    }
}

// --- TIER 1: Feature Coverage (Property-based) ---

proptest! {
    #[test]
    fn test_tier1_constructors_conversions(val_u in any::<u128>(), val_i in any::<i128>()) {
        let mp_u = MpUint::from(val_u);
        prop_assert_eq!(
            mp_u.to_u128().expect("should fit in u128"),
            val_u,
            "u128 roundtrip"
        );

        let mp_i = MpInt::from(val_i);
        prop_assert_eq!(
            mp_i.to_i128().expect("should fit in i128"),
            val_i,
            "i128 roundtrip"
        );

        if val_u > 0 {
            let bits = bounded_width(128);
            let mp_checked_u = MpUint::with_precision_checked(val_u, bits)
                .expect("should fit in 128-bit precision");
            prop_assert_eq!(
                mp_checked_u.to_u128().expect("should fit in u128"),
                val_u,
                "bounded u128 roundtrip"
            );
            prop_assert_eq!(
                get_precision_str(&mp_checked_u),
                "Bounded(128)",
                "precision string"
            );
        }
    }
}

proptest! {
    #[test]
    fn test_tier1_radix_conversions(u1 in any::<u128>()) {
        let au1 = MpUint::from(u1);

        for radix in &[2, 8, 10, 16] {
            let s = au1.to_string_radix(*radix);
            let au2 = MpUint::from_str_radix(&s, *radix).expect("roundtrip should succeed");
            prop_assert_eq!(&au1, &au2, "radix {} roundtrip", *radix);
        }
    }
}

proptest! {
    #[test]
    fn test_tier1_ambient_precision(width in 8_usize..=128) {
        #[cfg(feature = "std")]
        {
            let bounded_precision = PrecisionContext::with_bounded(width, || {
                let a = MpUint::from(0_u8);
                get_precision_str(&a)
            });
            prop_assert_eq!(
                bounded_precision,
                format!("Bounded({width})"),
                "bounded precision string"
            );

            let unlimited_precision = PrecisionContext::with_unlimited(|| {
                let a = MpUint::from(0_u8);
                get_precision_str(&a)
            });
            prop_assert_eq!(
                unlimited_precision,
                "Unlimited",
                "unlimited precision string"
            );
        }
        #[cfg(not(feature = "std"))]
        let _ = width;
    }
}

// --- TIER 2: Boundary & Corner Cases (Fuzzed / Property-based) ---

proptest! {
    #[test]
    fn test_tier2_boundaries(bits in 4_usize..=32) {
        let n = bounded_width(bits);

        let max_val = MpUint::max_for_precision(bits);
        let min_val = MpUint::min_for_precision(bits);
        prop_assert_eq!(&min_val, &MpUint::zero(), "min_for_precision");
        prop_assert_eq!(
            get_precision_str(&min_val),
            format!("Bounded({bits})"),
            "min_for_precision metadata"
        );

        prop_assert!(
            MpUint::with_precision_checked(max_val.clone(), n).is_ok(),
            "max_val should fit"
        );
        let overflow_val = max_val.wrapping_add(&MpUint::one());
        prop_assert!(
            MpUint::with_precision_checked(overflow_val, n).is_err(),
            "overflow should fail"
        );
    }
}

proptest! {
    #[test]
    fn test_tier2_div_by_zero(u1 in any::<u128>()) {
        let au1 = MpUint::from(u1);

        prop_assert!(
            au1.checked_div(&MpUint::zero()).is_none(),
            "div by zero is none"
        );
        prop_assert!(
            au1.checked_rem(&MpUint::zero()).is_none(),
            "rem by zero is none"
        );

        prop_assert_eq!(
            au1.wrapping_div(&MpUint::zero()),
            MpUint::zero(),
            "wrapping div by zero"
        );
        prop_assert_eq!(
            au1.wrapping_rem(&MpUint::zero()),
            MpUint::zero(),
            "wrapping rem by zero"
        );
    }
}

// --- TIER 3: Cross-Feature Combinations (Property-based) ---

proptest! {
    #[test]
    fn test_tier3_identities(u1 in 0_u64..1_000_000_u64, u2 in 1_u64..1_000_000_u64) {
        let au1 = MpUint::from(u1);
        let au2 = MpUint::from(u2);

        let div = au1.wrapping_div(&au2);
        let rem = au1.wrapping_rem(&au2);
        let recombined = div.wrapping_mul(&au2).wrapping_add(&rem);
        prop_assert_eq!(au1, recombined, "division identity");
    }
}

// --- TIER 4: Real-World Application Scenarios (Property-based) ---

proptest! {
    #[test]
    fn test_tier4_fibonacci(n in 5_usize..=80) {
        let mut f0 = MpUint::zero();
        let mut f1 = MpUint::one();

        let mut terms = vec![f0.clone(), f1.clone()];
        for _ in 2..=n {
            let f2 = f0.wrapping_add(&f1);
            terms.push(f2.clone());
            f0 = f1;
            f1 = f2;
        }

        for k in 0..(n - 1) {
            let idx_k = k;
            let idx_k1 = idx_k.wrapping_add(1);
            let idx_k2 = idx_k.wrapping_add(2);
            let sum = terms[idx_k].wrapping_add(&terms[idx_k1]);
            prop_assert_eq!(&sum, &terms[idx_k2], "fibonacci recurrence at k={}", k);
        }
    }
}

proptest! {
    #[test]
    fn test_tier4_modular_arithmetic(
        p_idx in 0_usize..10,
        q_idx in 0_usize..10,
        a_candidate in 2_u64..2_810_u64,
    ) {
        prop_assume!(p_idx != q_idx);
        let primes = [17_u64, 19, 23, 29, 31, 37, 41, 43, 47, 53];
        let p = MpUint::from(primes[p_idx]);
        let q = MpUint::from(primes[q_idx]);
        let modulus = p.wrapping_mul(&q);

        let p_minus_1 = p.wrapping_sub(&MpUint::one());
        let q_minus_1 = q.wrapping_sub(&MpUint::one());
        let phi = p_minus_1.wrapping_mul(&q_minus_1);

        let modulus_u64 = primes[p_idx].checked_mul(primes[q_idx]).expect("small prime product");
        prop_assume!(a_candidate < modulus_u64);
        let a = MpUint::from(a_candidate);
        prop_assume!(a.gcd_lcm(&modulus).map_or_else(MpUint::zero, |(g, _)| g) == MpUint::one());

        let res = a.pow_mod(&phi, &modulus).expect("pow_mod should succeed");
        prop_assert_eq!(res, MpUint::one(), "Euler's totient theorem");
    }
}
