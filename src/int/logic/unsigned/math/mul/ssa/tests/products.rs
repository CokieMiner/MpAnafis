//! Prepared and dispatched products and squares with exact guarded arenas.

#![expect(
    unsafe_code,
    reason = "Operand-bound plans initialize exact disjoint MaybeUninit product outputs before the test reads them"
)]

use core::mem::MaybeUninit;

use super::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn products_and_squares_match_schoolbook_with_exact_dirty_arenas(
        (mut left, mut right) in if cfg!(miri) { operands().boxed() } else {
            prop_oneof![operands(), transform_operands(), (
                prop::collection::vec(any::<Limb>(), 32..=200),
                prop::collection::vec(any::<Limb>(), 32..=200),
            )].boxed()
        },
        pattern in 0_u8..=6,
    ) {
        match pattern {
            1 => left.fill(0),
            2 => {
                let left_active = left.len().div_ceil(2);
                let right_active = right.len().div_ceil(2);
                left.iter_mut().skip(left_active).for_each(|limb| *limb = 0);
                right.iter_mut().skip(right_active).for_each(|limb| *limb = 0);
            }
            3 => {
                left.fill(Limb::MAX);
                right.fill(Limb::MAX - 1);
            }
            5 => {
                let active = left.len().div_ceil(2);
                left.iter_mut().skip(active).for_each(|limb| *limb = 0);
            }
            6 => {
                let active = right.len().div_ceil(2);
                right.iter_mut().skip(active).for_each(|limb| *limb = 0);
            }
            _ => {}
        }
        if pattern == 4 {
            check_products(&left, &left);
        } else {
            check_products(&left, &right);
        }
    }
}

#[cfg_attr(
    miri,
    ignore = "the fixed native sweep exercises wide power-of-two and odd-factor CRT geometries; smaller generated products run under Miri"
)]
#[test]
fn irregular_crt_widths_and_square_untwists_match_schoolbook() {
    let wide = SSA_BASE_MODULUS_BITS
        .div_ceil(LIMB_BITS)
        .checked_add(1)
        .expect("transform width fits");
    for (left_len, right_len) in [
        (1_usize, 1_usize),
        (1, 2),
        (32, 17),
        (64, 64),
        (192, 192),
        (256, 256),
        (384, 384),
        (512, 512),
        (wide, wide),
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
    check_execution(
        left,
        right,
        &expected,
        &expected_square,
        &SequentialExecutor,
    );
    check_execution(
        left,
        right,
        &expected,
        &expected_square,
        &CountingExecutor::default(),
    );
}

fn check_execution<E: ParallelExecutor>(
    left: &[Limb],
    right: &[Limb],
    expected: &[Limb],
    expected_square: &[Limb],
    executor: &E,
) {
    for choice in [
        TransformChoice::PLANNED,
        TransformChoice::FORCED,
        TransformChoice::FORCED_DIRECT_FERMAT,
    ] {
        let multiplication =
            SsaMultiplicationPlan::try_new(left, right, choice, executor.parallelism())
                .expect("test product geometry fits");
        let squaring = SsaSquaringPlan::try_new(left, choice, executor.parallelism().get())
            .expect("test square geometry fits");
        assert_eq!(multiplication.result_len, expected.len());
        let mut mul_scratch = vec![
            Limb::MAX;
            multiplication
                .scratch_len
                .checked_add(2)
                .expect("canaries fit")
        ];
        let mut sqr_scratch =
            vec![Limb::MAX; squaring.scratch_len.checked_add(2).expect("canaries fit")];
        let mut product =
            vec![MaybeUninit::uninit(); expected.len().checked_add(2).expect("canaries fit")];
        *product.first_mut().expect("prefix canary") = MaybeUninit::new(37);
        *product.last_mut().expect("suffix canary") = MaybeUninit::new(37);
        let mut square = vec![37; expected_square.len().checked_add(2).expect("canaries fit")];
        // Run the same retained plans twice, then replan through both public
        // dispatch boundaries with the same dirty arenas and fresh output state.
        for prepared in [true, true, false] {
            let product_window = product.get_mut(1..=expected.len()).expect("exact product");
            product_window.fill(MaybeUninit::uninit());
            let square_window = square
                .get_mut(1..=expected_square.len())
                .expect("exact square");
            square_window.fill(37);
            let mul_arena = mul_scratch
                .get_mut(1..=multiplication.scratch_len)
                .expect("exact product arena");
            let sqr_arena = sqr_scratch
                .get_mut(1..=squaring.scratch_len)
                .expect("exact square arena");
            if prepared {
                // SAFETY: both operand-bound plans have their exact disjoint
                // output and scratch spans, and their construction-time executor.
                unsafe {
                    multiplication.run_with_scratch(product_window, mul_arena, executor);
                    squaring.run_with_scratch(square_window, sqr_arena, executor);
                }
            } else {
                // Fresh dispatch must reuse the dirty arenas from the prepared run.
                assert!(Ssa::try_mul_with_executor(
                    product_window,
                    left,
                    right,
                    choice,
                    mul_arena,
                    executor
                ));
                assert!(Ssa::try_sqr_with_executor(
                    square_window,
                    left,
                    choice,
                    sqr_arena,
                    executor
                ));
            }
            // SAFETY: successful execution initialized the complete product;
            // its two excluded canaries were initialized before either run.
            let initialized = unsafe { LimbOutput::assume_init(&product) };
            assert_eq!(
                initialized.get(1..=expected.len()).expect("product"),
                expected
            );
            assert_eq!(
                square.get(1..=expected_square.len()).expect("square"),
                expected_square
            );
            for output in [initialized, square.as_slice()] {
                assert_eq!((output.first(), output.last()), (Some(&37), Some(&37)));
            }
            for scratch in [&mul_scratch, &sqr_scratch] {
                assert_eq!(
                    (scratch.first(), scratch.last()),
                    (Some(&Limb::MAX), Some(&Limb::MAX))
                );
            }
        }
    }
}
