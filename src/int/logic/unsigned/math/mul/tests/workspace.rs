//! Exact dirty workspaces, guard sentinels, and cancellation across Toom tiers.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::super::{
    ArchKernels, Limb, Multiplication, Recursive, Schoolbook, Toom3, Toom4, Toom6, Toom8, Toom32,
    Toom43,
};

#[test]
fn guarded_coefficients_initialize_dirty_outputs() {
    let kernel = ArchKernels::selected_add_mul_limbs_unchecked();
    for (guard_limbs, guard_max) in [(1_usize, 14_usize), (2, 5461)] {
        for low_len in [0_usize, 1, 2, 5] {
            for low_word in [0, Limb::MAX] {
                for guard in [0, 1, guard_max] {
                    let mut left = vec![low_word; low_len];
                    let mut right = vec![low_word; low_len];
                    left.push(guard);
                    right.push(guard_max - guard);
                    let width = low_len.checked_mul(2).expect("test width fits");
                    let exact = width.checked_add(guard_limbs).expect("guard fits");
                    let mut expected = vec![0; width.checked_add(2).expect("reference fits")];
                    let mut actual = vec![Limb::MAX; exact.checked_add(3).expect("surplus fits")];
                    Schoolbook::mul(&mut expected, &left, &right);
                    if guard_limbs == 1 {
                        Recursive::guarded_evaluation_product::<15, 1, _>(
                            &mut actual,
                            &left,
                            &right,
                            &mut [],
                            kernel,
                            |dst, a, b, _| Schoolbook::mul(dst, a, b),
                        );
                    } else {
                        Recursive::guarded_evaluation_product::<5462, 2, _>(
                            &mut actual,
                            &left,
                            &right,
                            &mut [],
                            kernel,
                            |dst, a, b, _| Schoolbook::mul(dst, a, b),
                        );
                    }
                    let (product, surplus) = actual.split_at(exact);
                    assert_eq!(
                        product,
                        expected.get(..exact).expect("reference prefix fits")
                    );
                    assert!(surplus.iter().all(|limb| *limb == 0));

                    actual.fill(Limb::MAX);
                    Schoolbook::sqr(&mut expected, &left);
                    if guard_limbs == 1 {
                        Recursive::guarded_evaluation_square::<15, 1>(
                            &mut actual,
                            &left,
                            &mut [],
                            kernel,
                            |dst, a, _| Schoolbook::sqr(dst, a),
                        );
                    } else {
                        Recursive::guarded_evaluation_square::<5462, 2>(
                            &mut actual,
                            &left,
                            &mut [],
                            kernel,
                            |dst, a, _| Schoolbook::sqr(dst, a),
                        );
                    }
                    let (square, square_surplus) = actual.split_at(exact);
                    assert_eq!(
                        square,
                        expected.get(..exact).expect("reference prefix fits")
                    );
                    assert!(square_surplus.iter().all(|limb| *limb == 0));
                }
            }
        }
    }
}

type Product = fn(&mut [Limb], &[Limb], &[Limb], &mut [Limb]);
type Square = fn(&mut [Limb], &[Limb], &mut [Limb]);
type MulScratchLen = fn(usize, usize) -> usize;

struct Tier {
    parts: usize,
    multiply: Product,
    square: Square,
    mul_scratch: MulScratchLen,
    sqr_scratch: fn(usize) -> usize,
}

const TIERS: [Tier; 4] = [
    Tier {
        parts: 3,
        multiply: Toom3::mul,
        square: Toom3::sqr,
        mul_scratch: Multiplication::toom3_mul_scratch_len,
        sqr_scratch: Multiplication::toom3_sqr_scratch_len,
    },
    Tier {
        parts: 4,
        multiply: Toom4::mul,
        square: Toom4::sqr,
        mul_scratch: Multiplication::toom4_mul_scratch_len,
        sqr_scratch: Multiplication::toom4_sqr_scratch_len,
    },
    Tier {
        parts: 6,
        multiply: Toom6::mul,
        square: Toom6::sqr,
        mul_scratch: Multiplication::toom6_mul_scratch_len,
        sqr_scratch: Multiplication::toom6_sqr_scratch_len,
    },
    Tier {
        parts: 8,
        multiply: Toom8::mul,
        square: Toom8::sqr,
        mul_scratch: Multiplication::toom8_mul_scratch_len,
        sqr_scratch: Multiplication::toom8_sqr_scratch_len,
    },
];

#[test]
fn toom_sign_changes_and_zero_endpoints_preserve_dirty_guards() {
    for tier in &TIERS {
        for split in [1_usize, 2, 3, 4] {
            for extra in [0_usize, 1] {
                let left_len = split * (tier.parts + extra);
                let right_len = split * tier.parts;
                let mut left = vec![Limb::MAX; left_len];
                let mut right = vec![Limb::MAX; right_len];
                let product_len = left_len + right_len;
                let mut expected = vec![0; product_len];
                let mut output = guarded(product_len);
                let mut scratch = guarded((tier.mul_scratch)(left_len, right_len));
                // Alternate signs at negative points, then equal signs and
                // zero constant blocks, reusing the already dirty workspace.
                for pattern in 0_usize..4 {
                    for (index, limb) in left.iter_mut().enumerate() {
                        *limb = if pattern == 3 || index.div_euclid(split) & 1 == 0 {
                            Limb::MAX
                        } else {
                            0
                        };
                    }
                    for (index, limb) in right.iter_mut().enumerate() {
                        *limb = if pattern == 3 || (index.div_euclid(split) ^ pattern) & 1 == 0 {
                            Limb::MAX
                        } else {
                            0
                        };
                    }
                    if pattern == 2 {
                        left.get_mut(..split).expect("constant block fits").fill(0);
                        right.get_mut(..split).expect("constant block fits").fill(0);
                    }
                    Schoolbook::mul(&mut expected, &left, &right);
                    (tier.multiply)(interior(&mut output), &left, &right, interior(&mut scratch));
                    assert_eq!(
                        interior(&mut output),
                        expected,
                        "Toom-{} at {left_len}x{right_len}, pattern {pattern}",
                        tier.parts
                    );
                    assert!(
                        sentinels_intact(&output) && sentinels_intact(&scratch),
                        "Toom-{} changed a guard sentinel",
                        tier.parts
                    );
                }
            }
        }
    }
}

#[test]
fn fractional_toom_endpoints_reuse_exact_dirty_workspace() {
    let tiers: [(usize, usize, Product, MulScratchLen); 2] = [
        (3, 2, Toom32::mul, Multiplication::toom32_mul_scratch_len),
        (4, 3, Toom43::mul, Multiplication::toom43_mul_scratch_len),
    ];
    for (large_parts, small_parts, multiply, required_scratch) in tiers {
        for split in [1_usize, 2, 3, 8, 16] {
            let left_len = large_parts * split;
            let right_len = small_parts * split;
            let mut left = vec![Limb::MAX; left_len];
            let mut right = vec![Limb::MAX; right_len];
            let mut expected = vec![0; left_len + right_len];
            let mut output = guarded(expected.len());
            let mut scratch = guarded(required_scratch(left_len, right_len));
            for pattern in 0..3 {
                if pattern != 0 {
                    left.get_mut(..split).expect("constant fits").fill(0);
                    right.get_mut(..split).expect("constant fits").fill(0);
                }
                if pattern == 2 {
                    left.get_mut(split..split * 2)
                        .expect("linear part fits")
                        .fill(1);
                }
                Schoolbook::mul(&mut expected, &left, &right);
                multiply(interior(&mut output), &left, &right, interior(&mut scratch));
                assert_eq!(
                    interior(&mut output),
                    expected,
                    "fractional Toom {large_parts}x{small_parts} at split {split}, pattern {pattern}"
                );
                assert!(
                    sentinels_intact(&output) && sentinels_intact(&scratch),
                    "fractional Toom changed a guard sentinel"
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[cfg_attr(miri, ignore = "the complete Toom width matrix is covered by native execution")]
    #[test]
    fn prop_toom_reuses_exact_dirty_workspace(
        split in prop::sample::select(vec![2_usize, 3, 19, 20, 21, 27, 28, 29, 31, 32, 33, 63, 64, 65, 106, 107, 108]),
        tail in 0_usize..3,
        seed in any::<Limb>(),
        repeated_parts in any::<bool>(),
    ) {
        for tier in &TIERS {
            let len = split.checked_mul(tier.parts).and_then(|width| width.checked_sub(tail))
                .expect("test operand dimensions fit");
            let left = operand(len, split, seed, repeated_parts);
            let right = operand(len, split, !seed, repeated_parts);
            let product_len = len.checked_mul(2).expect("test product width fits");
            let mut expected = vec![0; product_len];
            let mut output = guarded(product_len);
            let mut mul_scratch = guarded((tier.mul_scratch)(len, len));
            let mut sqr_scratch = guarded((tier.sqr_scratch)(len));
            for _ in 0..2 {
                Schoolbook::mul(&mut expected, &left, &right);
                (tier.multiply)(interior(&mut output), &left, &right, interior(&mut mul_scratch));
                prop_assert_eq!(interior(&mut output), expected.as_slice(), "Toom-{} product at {} limbs", tier.parts, len);
                prop_assert!(sentinels_intact(&output) && sentinels_intact(&mul_scratch));

                Schoolbook::sqr(&mut expected, &left);
                (tier.square)(interior(&mut output), &left, interior(&mut sqr_scratch));
                prop_assert_eq!(interior(&mut output), expected.as_slice(), "Toom-{} square at {} limbs", tier.parts, len);
                prop_assert!(sentinels_intact(&output) && sentinels_intact(&sqr_scratch));
            }
        }
    }
}

fn operand(len: usize, split: usize, seed: Limb, repeated_parts: bool) -> Vec<Limb> {
    (0..len)
        .map(|index| {
            let position = if repeated_parts {
                index.rem_euclid(split)
            } else {
                index
            };
            // Arithmetic modulo the limb radix gives reproducible dense words;
            // repeated chunks also create exactly cancelling negative points.
            seed.wrapping_add(position.wrapping_mul(0x9e37))
        })
        .collect()
}

fn guarded(len: usize) -> Vec<Limb> {
    vec![Limb::MAX; len.checked_add(2).expect("test sentinels fit")]
}

fn interior(value: &mut [Limb]) -> &mut [Limb] {
    let (_, after_first) = value.split_first_mut().expect("leading sentinel exists");
    after_first
        .split_last_mut()
        .expect("trailing sentinel exists")
        .1
}

fn sentinels_intact(value: &[Limb]) -> bool {
    value.first() == Some(&Limb::MAX) && value.last() == Some(&Limb::MAX)
}
