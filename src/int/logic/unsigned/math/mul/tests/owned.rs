//! Owned multiplication, squaring, assignment, and scalar storage transitions.

use alloc::vec;

use proptest::prelude::*;

use crate::int::logic::unsigned::math::mul::{InternalMpUint, Limb, MulScratch, Schoolbook};

#[test]
fn owned_paths_cover_zero_identity_inline_heap_and_stack_workspaces() {
    let mut shapes = vec![
        (18, 18),
        (19, 19),
        (20, 20),
        (24, 24),
        (32, 32),
        (48, 48),
        (23, 22),
        (64, 48),
    ];
    for left in 0..=5_usize {
        for right in 0..=5_usize {
            shapes.push((left, right));
        }
    }
    for (left_len, right_len) in shapes {
        if cfg!(miri) && left_len.max(right_len) > 5 {
            continue;
        }
        for top in [1, Limb::MAX] {
            let mut left = vec![Limb::MAX; left_len];
            let mut right = vec![Limb::MAX; right_len];
            if let Some(last) = left.last_mut() {
                *last = top;
            }
            if let Some(last) = right.last_mut() {
                *last = top;
            }
            let a = InternalMpUint::from_limbs(left);
            let b = InternalMpUint::from_limbs(right);
            check_owned_paths(&a, &b);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn owned_paths_match_schoolbook_for_arbitrary_limbs(
        left in prop::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 8 } else { 128 }),
        right in prop::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 8 } else { 128 }),
        scalar in any::<Limb>(),
    ) {
        let a = InternalMpUint::from_limbs(left);
        let b = InternalMpUint::from_limbs(right);
        check_owned_paths(&a, &b);
        check_owned_paths(&a, &InternalMpUint::from_limbs(vec![scalar]));
    }
}

fn check_owned_paths(a: &InternalMpUint, b: &InternalMpUint) {
    let width = a
        .limbs()
        .len()
        .checked_add(b.limbs().len())
        .expect("test product width fits");
    let mut full = vec![0; width];
    Schoolbook::mul(&mut full, a.limbs(), b.limbs());
    let expected = InternalMpUint::from_limbs(full);
    let mut scratch = MulScratch::default();
    for (left, right) in [(a, b), (b, a)] {
        assert_eq!(left.mul(right).limbs(), expected.limbs());
        let mut in_place = left.clone();
        in_place.mul_assign(right);
        assert_eq!(in_place.limbs(), expected.limbs());
        assert_eq!(
            left.clone().mul_into(right.clone()).limbs(),
            expected.limbs()
        );
        let mut destination = InternalMpUint::from_limbs(vec![Limb::MAX; width.max(16)]);
        destination.assign_product(left, right);
        assert_eq!(destination.limbs(), expected.limbs());
        scratch.buf.fill(Limb::MAX);
        destination.assign_product_with_scratch(left, right, &mut scratch);
        assert_eq!(destination.limbs(), expected.limbs());
    }
    let duplicate = a.limbs().to_vec();
    let square_width = a
        .limbs()
        .len()
        .checked_mul(2)
        .expect("test square width fits");
    let mut full_square = vec![0; square_width];
    Schoolbook::mul(&mut full_square, a.limbs(), &duplicate);
    let expected_square = InternalMpUint::from_limbs(full_square);
    assert_eq!(a.square().limbs(), expected_square.limbs());
    assert_eq!(a.mul(a).limbs(), expected_square.limbs());
    let mut destination = InternalMpUint::from_limbs(vec![Limb::MAX; square_width.max(16)]);
    destination.assign_square(a);
    assert_eq!(destination.limbs(), expected_square.limbs());
    scratch.buf.fill(Limb::MAX);
    destination.assign_square_with_scratch(a, &mut scratch);
    assert_eq!(destination.limbs(), expected_square.limbs());
    destination.assign_product(a, a);
    assert_eq!(destination.limbs(), expected_square.limbs());
    destination.assign_product_with_scratch(a, a, &mut scratch);
    assert_eq!(destination.limbs(), expected_square.limbs());
}
