//! Cached powers and guarded workspace reuse during root refinement.

use proptest::{
    prelude::{ProptestConfig, any},
    prop_assert, prop_assert_eq, proptest,
};

use super::super::{DivScratch, InternalMpUint, Limb, MulScratch, NthRootScratch, Roots};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 64 }))]
    #[test]
    fn double_limb_newton_retains_exact_powers(
        low in any::<Limb>(), high in 1_usize..=Limb::MAX, candidate_degree in 2_u32..=18,
    ) {
        let input = InternalMpUint::from_limbs_2(low, high);
        let bits = input.significant_bits();
        let native_degree = usize::try_from(candidate_degree).expect("degree <= 18");
        let exponent = candidate_degree.checked_sub(1).expect("degree >= 2");
        let log_degree = usize::try_from(u32::BITS.checked_sub(exponent.leading_zeros()).expect("positive exponent")).expect("log width <= 5");
        let degree = if bits.div_ceil(native_degree) >= log_degree.checked_add(3).expect("at most eight bits") { candidate_degree } else { 2 };
        let mut scratch = NthRootScratch {
            x_pow_n_minus_1: InternalMpUint::from_limb(Limb::MAX),
            temp_prod: InternalMpUint::from_limb(Limb::MAX),
            ..NthRootScratch::default()
        };
        let root = scratch.nth_root_double_limb::<true>(&input, degree, bits);
        let power_degree = degree.checked_sub(1).expect("positive denominator exponent");
        prop_assert_eq!(&scratch.x_pow_n_minus_1, &root.pow(power_degree));
        prop_assert_eq!(&scratch.temp_prod, &root.pow(degree));
        prop_assert!(scratch.temp_prod <= input);
        prop_assert!(root.add(&InternalMpUint::one()).pow(degree) > input);
        prop_assert_eq!(scratch.nth_root_double_limb::<false>(&input, degree, bits), root);
    }

    #[test]
    fn precision_growth_preserves_brackets_and_cached_powers(
        words in proptest::collection::vec(any::<Limb>(), 2..=if cfg!(miri) { 5 } else { 257 }),
        degree in 3_u32..=130,
    ) {
        let input = InternalMpUint::from_limbs(words);
        let root = input.nth_root(degree);
        prop_assert!(root.pow(degree) <= input);
        prop_assert!(root.add(&InternalMpUint::one()).pow(degree) > input);
        if input.limbs().len() >= 2 {
            let mut scratch = NthRootScratch::default();
            let cached = scratch.nth_root_multi_limb::<true>(&input, degree, input.significant_bits());
            let exponent = degree.checked_sub(1).expect("degree >= 3");
            prop_assert_eq!(&cached, &root);
            prop_assert_eq!(&scratch.x_pow_n_minus_1, &cached.pow(exponent));
            prop_assert_eq!(&scratch.temp_prod, &cached.pow(degree));
        }
    }

    #[test]
    fn normalized_roots_reuse_poisoned_storage_and_preserve_guards(
        mut limbs in proptest::collection::vec(any::<Limb>(), 6..=if cfg!(miri) { 8 } else { 258 }),
    ) {
        if limbs.len() & 1 != 0 { limbs.push(1); }
        *limbs.last_mut().expect("nonempty input") |= 1 << (Limb::BITS - 2);
        let original = InternalMpUint::from_limbs(limbs.clone());
        let width = limbs.len() >> 1;
        let input_end = limbs.len().checked_add(1).expect("bounded input width");
        let root_end = width.checked_add(1).expect("bounded root width");
        let mut input = alloc::vec![31; input_end.checked_add(1).expect("input guard")];
        let mut output = alloc::vec![37; root_end.checked_add(1).expect("root guard")];
        let scratch_len = width.checked_add(width >> 1).and_then(|length| length.checked_add(2)).expect("bounded scratch width");
        let mut scratch = alloc::vec![Limb::MAX; scratch_len];
        let mut division = DivScratch::default();
        let mut multiplication = MulScratch::default();
        let window = input.get_mut(1..input_end).expect("guarded input");
        let root = output.get_mut(1..root_end).expect("guarded root");
        window.copy_from_slice(&limbs);
        Roots::sqrt_rem_recursive::<true>(window, root, &mut scratch, &mut division, &mut multiplication);
        let expected = InternalMpUint::from_limbs(root.to_vec());
        let remainder = InternalMpUint::from_limbs(window.get(..root_end).expect("residue").to_vec());
        prop_assert_eq!(expected.square().add(&remainder), original);
        prop_assert!(remainder <= expected.shl(1));
        window.copy_from_slice(&limbs);
        root.fill(Limb::MAX);
        Roots::sqrt_rem_recursive::<false>(window, root, &mut scratch, &mut division, &mut multiplication);
        prop_assert_eq!(InternalMpUint::from_limbs(root.to_vec()), expected);
        prop_assert_eq!((input.first(), input.last()), (Some(&31), Some(&31)));
        prop_assert_eq!((output.first(), output.last()), (Some(&37), Some(&37)));
    }
}
