//! Tests for worker fixtures, product verification, and parallel catalogs.

use mp_anafis::{
    MpUint,
    tune_api::{
        MultiplicationAlgorithm, MultiplicationBenchState, MultiplicationRunner, SquaringBenchState,
    },
};
#[cfg(feature = "rayon")]
use rayon::{ThreadPoolBuilder, current_num_threads};

use super::{
    DivisionGrid, DivisionWorker, GCD_SCORE_CASES, GcdWorker, HASH_A, HASH_B, ProductGrid,
    ProductWorker, ProfileWorkers,
    division::{SMALL_QUOTIENT_LIMBS, SMALL_QUOTIENT_VALUES},
    gcd::{GCD_OPERATIONS, GcdFixture, GcdShape},
    holdout::{
        CONVERSION_CASES, DIVISION_CASES, GCD_CASES, HOLDOUT_SEED, HoldoutWorker, ProductPattern,
    },
};
#[cfg(feature = "rayon")]
use super::{MUL_SCORE_CELLS, ParallelWorker, SQR_SCORE_CELLS, ScoreCell};

#[test]
fn direct_product_protocol_executes_valid_fixtures_from_one_limb_through_the_cios_boundary() {
    ProductWorker::print_score("1,2,3,31,32,33;193", "")
        .expect("direct products and consumer guards have valid fixtures");
}

#[test]
fn held_out_layout_has_complete_disjoint_families_and_valid_new_fixtures() {
    let (weights, families) = HoldoutWorker::layout();
    assert_eq!(weights.len(), 78);
    assert_eq!(families.len(), 10);
    assert!(weights.iter().all(|&weight| weight > 0));
    let mut offset = 0;
    for family in families {
        assert_eq!(family.start, offset);
        assert!(family.end > family.start);
        offset = family.end;
    }
    assert_eq!(offset, weights.len());
    for (radix, digits) in CONVERSION_CASES {
        let (input, expected) = HoldoutWorker::conversion_fixture(radix, digits);
        assert_eq!(input.len(), digits);
        assert_eq!(MpUint::from_str_radix(&input, radix), Ok(expected.clone()));
        assert_eq!(expected.to_string_radix(radix), input);
    }
    for (len, shape) in GCD_CASES {
        GcdFixture::new(
            len,
            shape,
            u64::try_from(HOLDOUT_SEED).expect("seed fits u64"),
        )
        .verify();
        assert!(
            !GCD_SCORE_CASES
                .iter()
                .any(|&(search_len, _)| len == search_len)
        );
    }
    for case in DIVISION_CASES {
        for (numerator, divisor, quotient, remainder) in
            DivisionWorker::division_fixtures(case.divisor_limbs, case.quotient_limbs, HOLDOUT_SEED)
        {
            assert_eq!(numerator.div_rem(&divisor), Some((quotient, remainder)));
        }
    }
}

#[test]
fn reserved_carry_patterns_match_exact_schoolbook_products_and_squares() {
    for len in [1_usize, 4, 17, 73, 257] {
        for pattern in [
            ProductPattern::Mixed,
            ProductPattern::Maximal,
            ProductPattern::Sparse,
            ProductPattern::Alternating,
        ] {
            let left = pattern.operand(len, HASH_A);
            let right = pattern.operand(len, HASH_B);
            assert_eq!(left.len(), len);
            assert_ne!(left.last(), Some(&0));
            let mut expected = vec![0; len * 2];
            let mut actual = vec![0; len * 2];
            MultiplicationRunner::new(MultiplicationAlgorithm::Schoolbook, len, len).run(
                &mut expected,
                &left,
                &right,
            );
            MultiplicationBenchState::default()
                .prepare(&mut actual, &left, &right)
                .run();
            assert_eq!(actual, expected, "{len} {pattern:?}");
            MultiplicationRunner::new(MultiplicationAlgorithm::Schoolbook, len, len).run(
                &mut expected,
                &left,
                &left,
            );
            SquaringBenchState::default()
                .prepare(&mut actual, &left)
                .run();
            assert_eq!(actual, expected, "square {len} {pattern:?}");
        }
    }
}

#[test]
fn direct_product_objectives_exclude_consumer_work_and_keep_complete_guards() {
    let grid = ProductGrid::parse("31,32,33;192,193,194").expect("valid geometry");
    assert_eq!(ProductGrid::parse(&grid.render()), Ok(grid.clone()));
    for invalid in ["32", "0;193", "32;0", "32;193;2", "x;193", "32;"] {
        assert!(ProductGrid::parse(invalid).is_err(), "{invalid}");
    }
    let all = ProductWorker::cell_weights(&grid, None);
    let cios = ProductWorker::cell_weights(&grid, Some("MONTGOMERY_CIOS_MAX_LIMBS"));
    let cyclic = ProductWorker::cell_weights(&grid, Some("MUL_MOD_BNM1_THRESHOLD"));
    assert_eq!(all.len(), 18);
    assert_eq!(cios.iter().sum::<u32>(), 3);
    assert_eq!(cyclic.iter().sum::<u32>(), 6);
    for ((guard, direct_cios), direct_cyclic) in all.iter().zip(&cios).zip(&cyclic) {
        assert_eq!(*guard, 1);
        assert!(*direct_cios == 0 || *direct_cyclic == 0);
    }
    assert!(cios.iter().zip(&cyclic).any(|(&a, &b)| a == 0 && b == 0));
}

#[test]
fn division_outputs_match_constructed_results_for_both_normalizations() {
    for len in [1_usize, 7, 33, 96] {
        for quotient_len in [
            1,
            len.div_euclid(2).max(1),
            len,
            len.checked_mul(3).expect("test width fits"),
        ] {
            for seed in [0, 1_979] {
                for (numerator, divisor, quotient, remainder) in
                    DivisionWorker::division_fixtures(len, quotient_len, seed)
                {
                    assert_eq!(numerator.checked_div(&divisor), Some(quotient.clone()));
                    assert_eq!(numerator.checked_rem(&divisor), Some(remainder.clone()));
                    assert_eq!(numerator.div_rem(&divisor), Some((quotient, remainder)));
                }
            }
        }
    }
}

#[test]
fn small_quotient_cells_cover_exact_and_corrected_remainders() {
    let grid = DivisionGrid::default();
    let cases = DivisionWorker::cases(&grid);
    let count: usize = cases
        .iter()
        .map(|case| {
            if case.scalar_quotient.is_some() {
                24
            } else {
                16
            }
        })
        .sum();
    assert_eq!(
        DivisionWorker::cell_weights(&DivisionWorker::cases(&grid)).len(),
        count
    );
    for width in [1, 2, 3, 4, 5] {
        assert!(cases.iter().any(|case| case.divisor_limbs == width));
    }
    for quotient_limbs in [15, 16, 17, 20, 21, 22, 32, 64, 65, 96, 192] {
        assert!(
            cases
                .iter()
                .any(|case| case.divisor_limbs == 64 && case.quotient_limbs == quotient_limbs)
        );
    }
    let neighbours = DivisionWorker::cases(&DivisionGrid {
        divisor_widths: vec![47, 48, 49],
        ..grid
    });
    for width in [47, 48, 49] {
        assert!(neighbours.iter().any(|case| case.divisor_limbs == width));
    }
    for (len, quotient) in SMALL_QUOTIENT_LIMBS.into_iter().flat_map(|len| {
        SMALL_QUOTIENT_VALUES
            .into_iter()
            .map(move |quotient| (len, quotient))
    }) {
        let fixtures = DivisionWorker::small_quotient_fixtures(len, quotient);
        assert_eq!(fixtures.len(), 6);
        for (numerator, denominator, expected_q, expected_r) in fixtures {
            assert_eq!(
                numerator.div_rem(&denominator),
                Some((expected_q, expected_r))
            );
        }
    }
}

#[cfg(feature = "rayon")]
#[test]
fn parallel_weights_preserve_individual_pool_widths() {
    let per_pool = ScoreCell::cell_weights(&MUL_SCORE_CELLS, &SQR_SCORE_CELLS);
    for workers in [2, 4, 8] {
        let weights = ParallelWorker::cell_weights(workers, false, &[]);
        assert_eq!(weights.len(), workers * per_pool.len());
        assert!(
            weights
                .chunks_exact(per_pool.len())
                .all(|chunk| chunk == per_pool)
        );
    }
}

#[test]
fn gcd_cells_have_four_complete_families_and_valid_fixtures() {
    assert_eq!(
        GcdWorker::cell_weights(&GCD_SCORE_CASES).len(),
        GCD_OPERATIONS
            .len()
            .checked_mul(GCD_SCORE_CASES.len())
            .expect("cell count fits")
    );
    for len in [1, 2, 16, 64, 129] {
        for shape in [
            GcdShape::Random,
            GcdShape::NearEqual,
            GcdShape::Uneven,
            GcdShape::SharedFactor,
            GcdShape::Fibonacci,
            GcdShape::Scalar,
            GcdShape::ExactMultiple,
            GcdShape::BitGap(16),
        ] {
            for seed in 0..4 {
                let fixture = GcdFixture::new(len, shape, seed);
                fixture.verify();
            }
        }
    }
}

#[test]
fn modular_product_oracle_accepts_valid_wide_product() {
    let left = [usize::MAX, 1];
    let right = [2];
    let product = [usize::MAX - 1, 3];
    ProfileWorkers::verify_product_residues(&product, &left, &right);
}

#[test]
#[should_panic(expected = "independent modular-product oracle")]
fn modular_product_oracle_rejects_corruption() {
    ProfileWorkers::verify_product_residues(&[7], &[2], &[3]);
}

#[test]
fn division_geometry_round_trips_independent_units_and_rejects_invalid_spans() {
    let grid = DivisionGrid {
        divisor_widths: vec![47, 48, 49],
        quotient_widths: vec![15, 16, 17],
        scalar_quotients: vec![63, 64, 65],
        block_ratios: vec![2, 3, 4],
    };
    assert_eq!(DivisionGrid::parse(&grid.render()), Ok(grid));
    assert_eq!(DivisionGrid::parse(""), Ok(DivisionGrid::default()));
    assert_eq!(
        DivisionGrid::parse(&DivisionGrid::default().render()),
        Ok(DivisionGrid::default())
    );
    for encoded in [
        "0",
        "1,",
        ";q=0",
        ";scalar=0",
        ";ratios=0",
        ";q=2;q=3",
        ";unknown=4",
        ";q",
    ] {
        assert!(DivisionGrid::parse(encoded).is_err(), "{encoded}");
    }
    assert!(DivisionGrid::parse(&usize::MAX.to_string()).is_err());
    assert!(DivisionGrid::parse(&format!(";q={}", usize::MAX)).is_err());
}

#[test]
fn division_cases_probe_explicit_quotient_and_ratio_boundaries() {
    let grid = DivisionGrid {
        quotient_widths: vec![11, 12, 13],
        block_ratios: vec![5],
        scalar_quotients: vec![67],
        ..DivisionGrid::default()
    };
    let cases = DivisionWorker::cases(&grid);
    for quotient in [10, 11, 12, 13] {
        assert!(
            cases
                .iter()
                .any(|case| case.divisor_limbs == 64 && case.quotient_limbs == quotient)
        );
    }
    assert!(
        cases
            .iter()
            .any(|case| case.divisor_limbs == 1 && case.quotient_limbs == 12)
    );
    assert!(cases.iter().any(|case| case.scalar_quotient == Some(67)));
    assert!(!cases.iter().any(|case| case.divisor_limbs == 67));
}

#[cfg(feature = "rayon")]
#[test]
fn parallel_production_catalog_preserves_pool_families_and_boundary_shapes() {
    let (multiplication, squaring) = ParallelWorker::cell_catalog(true, &[127, 128, 129]);
    for width in [1, 2, 4, 8, 127, 128, 129] {
        assert!(
            multiplication
                .iter()
                .any(|cell| cell.len_a == width && cell.len_b == width)
        );
        assert!(
            multiplication
                .iter()
                .any(|cell| cell.len_a == 2 * width && cell.len_b == width)
        );
        assert!(squaring.iter().any(|cell| cell.len_a == width));
    }
    assert!(
        multiplication
            .iter()
            .chain(&squaring)
            .all(|cell| cell.samples == 3)
    );
    let per_pool = ScoreCell::cell_weights(&multiplication, &squaring);
    assert!(per_pool.iter().all(|&weight| weight > 0));
    let weights = ParallelWorker::cell_weights(3, true, &[127, 128, 129]);
    assert_eq!(weights.len(), 3 * per_pool.len());
    assert!(
        weights
            .chunks_exact(per_pool.len())
            .all(|chunk| chunk == per_pool)
    );
}

#[cfg(feature = "rayon")]
#[test]
fn production_products_are_correct_inside_explicit_pools_without_timing() {
    for workers in [1, 2, 3] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test pool");
        pool.install(|| {
            assert_eq!(current_num_threads(), workers);
            for width in [1_usize, 2, 4, 8, 33, 257] {
                let left = ScoreCell::operand(width, HASH_A);
                let right = ScoreCell::operand(width * 2, HASH_B);
                let mut product = vec![0; left.len() + right.len()];
                MultiplicationBenchState::default()
                    .prepare(&mut product, &left, &right)
                    .run();
                ProfileWorkers::verify_product_residues(&product, &left, &right);
                let mut square = vec![0; left.len() * 2];
                SquaringBenchState::default()
                    .prepare(&mut square, &left)
                    .run();
                ProfileWorkers::verify_product_residues(&square, &left, &left);
            }
        });
    }
}
