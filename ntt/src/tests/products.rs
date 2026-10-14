//! Differential multiplication across every catalog prefix and coefficient width.

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::{Limb, MAX_PRIMES, NttWorkspace, SequentialExecutor, profile::NttProfile};

use super::{planning::forced, schoolbook::product};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 48 }))]
    #[test]
    fn multiplication_and_squaring_match_schoolbook_for_every_prime_count(
        left in collection::vec(any::<Limb>(), 1..=40), right in collection::vec(any::<Limb>(), 1..=40),
    ) {
        let expected = product(&left, &right);
        let expected_square = product(&left, &left);
        let mut workspace = NttWorkspace::new();
        let executor = SequentialExecutor::default();
        for count in 1..=MAX_PRIMES {
            let bits = count.checked_mul(49).expect("catalog capacity").checked_sub(16).expect("positive width").div_euclid(2);
            let profile = NttProfile { np: count, chunk_bits: u32::try_from(bits).expect("coefficient width"), max_transform_len: usize::MAX };
            let plan = forced(left.len(), right.len(), profile).expect("conservative profile");
            workspace.prepare_with_executor(plan, &executor);
            let mut actual = vec![Limb::MAX; expected.len()];
            prop_assert!(workspace.mul_with_executor(&mut actual, &left, &right, &executor));
            prop_assert_eq!(&actual, &expected, "prime count {}", count);
            let square_plan = forced(left.len(), left.len(), profile).expect("conservative square profile");
            workspace.prepare_with_executor(square_plan, &executor);
            actual.resize(expected_square.len(), Limb::MAX);
            prop_assert!(workspace.sqr_with_executor(&mut actual, &left, &executor));
            prop_assert_eq!(&actual, &expected_square, "square prime count {}", count);
        }
    }
}
