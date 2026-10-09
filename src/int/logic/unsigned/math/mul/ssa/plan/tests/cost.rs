//! Search exponents, candidate efficiency, and the conventional work recurrence.

use super::{CostMemo, Geometry, SsaOperation, SsaPlan};

#[test]
fn search_centres_and_basecase_cost_follow_the_structural_bounds() {
    assert_eq!(
        SsaPlan::search_centre(1 << 24),
        SsaPlan::search_centre(3 << 23)
    );
    for logarithm in 6_u32..usize::BITS {
        let centre = SsaPlan::search_centre(1_usize << logarithm);
        let classical = logarithm.div_euclid(2);
        assert!(centre.abs_diff(classical) <= 1, "logarithm={logarithm}");
        if (16..30).contains(&logarithm) {
            assert_eq!(centre, classical);
        }
    }
    let mut previous = 0;
    for limbs in 1_usize..=if cfg!(miri) { 32 } else { 512 } {
        let cost = SsaPlan::basecase_product_cost(limbs);
        assert!(cost >= previous, "limbs={limbs}");
        previous = cost;
    }
    assert!(SsaPlan::basecase_product_cost(256) < 256_usize.saturating_mul(256));
}

#[test]
fn pricing_rejects_low_efficiency_and_compares_nested_alignment_bumps() {
    let bits = 1 << 20;
    let mut memo = CostMemo::new();
    let mut priced = 0;
    for geometry in Geometry::for_exponent_candidates(12, bits, false)
        .into_iter()
        .flatten()
    {
        priced += 1;
        assert!(
            SsaPlan::price_geometry(&geometry, bits, 0, SsaOperation::Multiply, &mut memo)
                .is_none()
        );
    }
    assert!(priced > 0, "at least one low-efficiency candidate exists");
    let mut previous = 0;
    let mut distinct = 0;
    for geometry in Geometry::for_exponent_candidates(10, 1 << 30, false)
        .into_iter()
        .flatten()
    {
        assert!(2 * geometry.chunk_bits.get() + geometry.transform_log < geometry.inner_bits);
        if geometry.inner_bits != previous {
            distinct += 1;
            previous = geometry.inner_bits;
        }
    }
    assert!(
        distinct >= 2,
        "nested alignments produce competing ring widths"
    );
}
