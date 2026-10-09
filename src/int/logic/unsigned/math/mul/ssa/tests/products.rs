//! Product and square dispatch checked against independent schoolbook results.

use super::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn products_and_squares_match_schoolbook_with_exact_dirty_arenas(
        (left, right) in if cfg!(miri) { operands().boxed() } else {
            prop_oneof![operands(), transform_operands(), (
                prop::collection::vec(any::<Limb>(), 32..=200),
                prop::collection::vec(any::<Limb>(), 32..=200),
            )].boxed()
        },
    ) {
        check_products(&left, &right);
    }
}

#[cfg_attr(
    miri,
    ignore = "the fixed native sweep exercises wide power-of-two and odd-factor CRT geometries; smaller generated products run under Miri"
)]
#[test]
fn irregular_crt_widths_and_square_untwists_match_schoolbook() {
    for (left_len, right_len) in [
        (64_usize, 64_usize),
        (192, 192),
        (256, 256),
        (384, 384),
        (512, 512),
        (640, 640),
        (768, 768),
        (896, 896),
        (1024, 1024),
        (1088, 1088),
        (1280, 1280),
        (1536, 1536),
        (768, 512),
    ] {
        let left: Vec<_> = (0..left_len)
            .map(|index| {
                Limb::MAX
                    .wrapping_sub(index.wrapping_mul(0x9E37_79B9))
                    .rotate_left(7)
            })
            .collect();
        let right: Vec<_> = (0..right_len)
            .map(|index| {
                Limb::MAX
                    .wrapping_sub(index.wrapping_mul(0x85EB_CA6B))
                    .rotate_left(11)
            })
            .collect();
        check_products(&left, &right);
    }
}

fn check_products(left: &[Limb], right: &[Limb]) {
    let len = left
        .len()
        .checked_add(right.len())
        .expect("test product fits");
    let square_len = left.len().checked_mul(2).expect("test square fits");
    let mut expected = vec![0; len];
    let mut expected_square = vec![0; square_len];
    Schoolbook::mul(&mut expected, left, right);
    Schoolbook::mul(&mut expected_square, left, left);
    let mul_len = Ssa::mul_scratch_len_for_parallelism(left.len(), right.len(), 1);
    let sqr_len = Ssa::sqr_scratch_len_for_parallelism(left.len(), 1);
    let mut scratch = vec![Limb::MAX; mul_len.max(sqr_len).checked_add(2).expect("sentinels fit")];
    let mut product = vec![37; len.checked_add(2).expect("sentinels fit")];
    let mut square = vec![37; square_len.checked_add(2).expect("sentinels fit")];
    for choice in [TransformChoice::PLANNED, TransformChoice::FORCED] {
        scratch
            .get_mut(1..=mul_len)
            .expect("exact multiplication arena")
            .fill(Limb::MAX);
        assert!(Ssa::try_mul_with_executor(
            product.get_mut(1..=len).expect("exact product"),
            left,
            right,
            choice,
            scratch.get_mut(1..=mul_len).expect("exact arena"),
            &SequentialExecutor
        ));
        scratch
            .get_mut(1..=sqr_len)
            .expect("exact square arena")
            .fill(Limb::MAX);
        assert!(Ssa::try_sqr_with_executor(
            square.get_mut(1..=square_len).expect("exact square"),
            left,
            choice,
            scratch.get_mut(1..=sqr_len).expect("exact arena"),
            &SequentialExecutor
        ));
        assert_eq!(product.get(1..=len).expect("product"), expected);
        assert_eq!(square.get(1..=square_len).expect("square"), expected_square);
        for output in [&product, &square] {
            assert_eq!((output.first(), output.last()), (Some(&37), Some(&37)));
        }
        assert_eq!(
            (scratch.first(), scratch.last()),
            (Some(&Limb::MAX), Some(&Limb::MAX))
        );
    }
}
