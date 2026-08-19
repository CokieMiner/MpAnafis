//! Property tests for the balanced Toom-Cook 6 tier.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::*;
use crate::int::logic::math::mul::{
    Multiplication, Schoolbook, SharedEval,
    dispatch::{MulShape, Widths},
};

type ExactDivision = (Limb, fn(&mut [Limb]));

fn balanced_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            proptest::collection::vec(any::<Limb>(), 6),
            proptest::collection::vec(any::<Limb>(), 6),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 37),
            proptest::collection::vec(any::<Limb>(), 37),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 113),
            proptest::collection::vec(any::<Limb>(), 109),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 257),
            proptest::collection::vec(any::<Limb>(), 257),
        ),
    ]
}

fn half_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            proptest::collection::vec(any::<Limb>(), 70),
            proptest::collection::vec(any::<Limb>(), 60),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 113),
            proptest::collection::vec(any::<Limb>(), 100),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 221),
            proptest::collection::vec(any::<Limb>(), 257),
        ),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn prop_toom6_matches_basecase((a, b) in balanced_operands()) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::toom6_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom6::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_toom6_half_matches_basecase((a, b) in half_operands()) {
        prop_assert!(Widths::new(a.len(), b.len()).toom6_half_suitable());
        prop_assert!(!Widths::new(a.len(), b.len()).toom6_balanced());
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::toom6_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom6::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_toom6_square_matches_basecase(a in proptest::collection::vec(any::<Limb>(), 6..258)) {
        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::sqr(&mut expected, &a);

        let scratch_len = Multiplication::toom6_sqr_scratch_len(a.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom6::sqr(&mut actual, &a, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_specialized_exact_divisions_match_generic(
        value in proptest::collection::vec(any::<Limb>(), 0..65),
    ) {
        let divisions: [ExactDivision; 3] = [
            (9, SharedEval::exact_div9_in_place),
            (15, SharedEval::exact_div_radix_minus_one_in_place::<15>),
            (255, SharedEval::exact_div_radix_minus_one_in_place::<255>),
        ];
        for (divisor, specialized) in divisions {
            let mut expected = value.clone();
            let mut actual = value.clone();
            SharedEval::exact_div_odd_in_place(&mut expected, divisor, SharedEval::invert_odd(divisor));
            specialized(&mut actual);
            prop_assert_eq!(actual, expected);
        }
    }
}

/// `toom6::mul` and `toom6_mul_scratch_len` must resolve the same shape.
///
/// They derived it independently until `Widths::toom6_shape` became the single
/// resolution, and while they did they disagreed: the algorithm recursed under a
/// Toom-4 ceiling while the sizing had sized under the full one, and the buffer
/// came out short. The property tests above did catch it, but only because their
/// random widths happened to land on a lopsided ratio — the unsuitable branch had
/// no deliberate coverage at all.
///
/// This sweeps ratios across all three branches with exactly the scratch the
/// sizing reports, and asserts each branch is actually reached so it cannot
/// silently stop testing what it claims to.
#[test]
fn toom6_sizing_matches_every_shape_branch() {
    let mut seen_balanced = false;
    let mut seen_half = false;
    let mut seen_unsuitable = false;

    // Fractional as well as integer ratios: the half branch admits only a narrow
    // band around seven to six, which integer divisors step straight over.
    for larger in [40_usize, 61, 96, 113, 180, 257] {
        for (num, den) in [
            (1_usize, 1_usize),
            (6, 7),
            (5, 6),
            (3, 4),
            (1, 2),
            (1, 3),
            (1, 6),
            (1, 12),
        ] {
            let smaller = larger.wrapping_mul(num).div_euclid(den);
            if smaller < 6 {
                continue;
            }
            match Widths::new(larger, smaller).toom6_shape() {
                Some(MulShape::Balanced) => seen_balanced = true,
                Some(MulShape::Half) => seen_half = true,
                None => seen_unsuitable = true,
            }

            let a: Vec<Limb> = (0..larger)
                .map(|i| Limb::MAX.wrapping_sub(i.wrapping_mul(0x9E37)))
                .collect();
            let b: Vec<Limb> = (0..smaller)
                .map(|i| Limb::MAX.wrapping_sub(i.wrapping_mul(0x7F4A)))
                .collect();
            let result_len = larger.wrapping_add(smaller);
            let mut expected = vec![0; result_len];
            let mut actual = vec![Limb::MAX; result_len];
            Schoolbook::mul(&mut expected, &a, &b);

            let mut scratch =
                vec![Limb::MAX; Multiplication::toom6_mul_scratch_len(larger, smaller)];
            Toom6::mul(&mut actual, &a, &b, &mut scratch);
            assert_eq!(actual, expected, "Toom-6 wrong at {larger}x{smaller}");
        }
    }

    assert!(seen_balanced, "no shape reached the balanced branch");
    assert!(seen_half, "no shape reached the half branch");
    assert!(seen_unsuitable, "no shape reached the unsuitable branch");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2))]

    #[test]
    fn prop_toom6_recursive_matches_basecase(
        a in proptest::collection::vec(any::<Limb>(), 1024..2050),
        b in proptest::collection::vec(any::<Limb>(), 1024..2050),
    ) {
        let product_len = a.len().wrapping_add(b.len());
        let mut expected_product = vec![0; product_len];
        let mut actual_product = vec![0; product_len];
        Schoolbook::mul(&mut expected_product, &a, &b);
        let product_scratch_len = Multiplication::toom6_mul_scratch_len(a.len(), b.len());
        let mut product_scratch = vec![Limb::MAX; product_scratch_len];
        Toom6::mul(&mut actual_product, &a, &b, &mut product_scratch);
        prop_assert!(
            actual_product == expected_product,
            "recursive Toom-6 mismatch for lengths {} and {}",
            a.len(),
            b.len(),
        );

        let square_len = a.len().wrapping_mul(2);
        let mut expected_square = vec![0; square_len];
        let mut actual_square = vec![Limb::MAX; square_len];
        Schoolbook::sqr(&mut expected_square, &a);
        let square_scratch_len = Multiplication::toom6_sqr_scratch_len(a.len());
        let mut square_scratch = vec![Limb::MAX; square_scratch_len];
        Toom6::sqr(&mut actual_square, &a, &mut square_scratch);
        prop_assert_eq!(actual_square, expected_square);
    }
}
