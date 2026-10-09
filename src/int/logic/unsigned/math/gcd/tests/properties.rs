//! GCD identities, recursive reductions, and matrix-action properties.

use alloc::vec::Vec;

use proptest::prelude::*;

use crate::int::logic::unsigned::math::gcd::{hgcd::hgcd_block, lehmer::lehmer_update};

use super::{
    DivScratch, Gcd, HGCD_CROSSOVER_THRESHOLD, HgcdWorkspace, InternalMpUint, LIMB_BITS, Limb,
};

#[test]
fn large_gcd_preserves_a_known_common_factor() {
    let common = InternalMpUint::from_limb(3);
    let mut left = common.clone();
    left.shl_assign(70_usize.wrapping_mul(LIMB_BITS));
    let mut right = left.clone();
    right.add_assign(&common);

    // gcd(3*B^70, 3*(B^70 + 1)) = 3 because consecutive integers are
    // coprime. Both operands cross the HGCD dispatch boundary.
    assert_eq!(left.gcd(&right), common);
}

#[test]
fn fast_small_div_step_accepts_equal_leading_limbs() {
    // v = (B/2+1)*B^4 - 1 and u = (B/2)*B^5 share the leading limb B/2, yet
    // u = (B-2)*v + (2*B^4 + B - 2) with a remainder below v, so the exact
    // single-limb quotient is q = B-2. The early top-limb guard must reject
    // only strictly greater leading limbs.
    let half = Limb::MAX.wrapping_shr(1).wrapping_add(1);
    let full = Limb::MAX;
    let v = InternalMpUint::from_limbs([full, full, full, full, half].into_iter().collect());
    let original_u = InternalMpUint::from_limbs([0, 0, 0, 0, 0, half].into_iter().collect());
    let mut u = original_u.clone();
    let quotient =
        Gcd::fast_small_div_step(&mut u, &v).expect("equal leading limbs admit quotient B-2");
    assert_eq!(quotient, full.wrapping_sub(1));
    let expected_rem =
        InternalMpUint::from_limbs([full.wrapping_sub(1), 0, 0, 0, 2].into_iter().collect());
    assert_eq!(u, expected_rem);
    // Self-validating identity: q*v + r reconstructs the input exactly.
    let reconstructed = v.mul(&InternalMpUint::from_limb(quotient)).add(&u);
    assert_eq!(reconstructed, original_u);
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Thousand-limb asymmetric shifts exercise native allocation and reduction scale; bounded valuation properties run under Miri."
)]
fn independent_stripping_handles_asymmetric_shifts() {
    // Odd cofactors with different widths; one operand carries a thousand
    // extra zero limbs the other lacks. A common shift would copy those
    // limbs and open with a huge remainder.
    let odd_a = InternalMpUint::from_limbs(
        (0..300_usize)
            .map(|index| {
                Limb::try_from(index)
                    .expect("test index fits every supported limb")
                    .wrapping_mul(Limb::try_from(40_503_u32).expect("constant fits every limb"))
                    | 1
            })
            .collect(),
    );
    let odd_b = InternalMpUint::from_limbs(
        (0..200_usize)
            .map(|index| {
                Limb::try_from(index)
                    .expect("test index fits every supported limb")
                    .wrapping_mul(Limb::try_from(34_283_u32).expect("constant fits every limb"))
                    | 1
            })
            .collect(),
    );
    let shift_a = 1_000_usize
        .checked_mul(LIMB_BITS)
        .and_then(|bits| bits.checked_add(3))
        .expect("test shift fits");
    let shift_b = 5_usize
        .checked_mul(LIMB_BITS)
        .and_then(|bits| bits.checked_add(7))
        .expect("test shift fits");
    let mut a = odd_a.clone();
    a.shl_assign(shift_a);
    let mut b = odd_b.clone();
    b.shl_assign(shift_b);
    let mut expected = odd_a.gcd(&odd_b);
    expected.shl_assign(shift_b);
    assert_eq!(a.gcd(&b), expected);
    let mut workspace = HgcdWorkspace::default();
    assert_eq!(Gcd::compute_half_gcd(&a, &b, &mut workspace), expected);
    assert_eq!(Gcd::compute_lehmer(&a, &b), expected);
}

#[test]
#[cfg_attr(
    miri,
    ignore = "The 1024-limb fixture forces recursive HGCD; smaller matrix-action properties cover its memory contracts under Miri."
)]
fn recursive_half_gcd_block_preserves_gcd() {
    // This width exercises recursive construction and exact matrix composition.
    let left_limbs = (0..1_024_usize)
        .map(|index| {
            Limb::try_from(index)
                .expect("test index fits every supported limb")
                .wrapping_mul(Limb::try_from(40_503_u32).expect("constant fits every limb"))
                .rotate_left(u32::try_from(index & 15).expect("rotation is at most 15"))
                | 1
        })
        .collect();
    let right_limbs = (0..1_024_usize)
        .map(|index| {
            Limb::try_from(index)
                .expect("test index fits every supported limb")
                .wrapping_mul(Limb::try_from(34_283_u32).expect("constant fits every limb"))
                .rotate_right(u32::try_from(index & 15).expect("rotation is at most 15"))
                | 1
        })
        .collect();
    let left = InternalMpUint::from_limbs(left_limbs);
    let right = InternalMpUint::from_limbs(right_limbs);
    let expected = left.gcd(&right);
    let (mut u, mut v) = if left.cmp(&right).is_gt() {
        (left, right)
    } else {
        (right, left)
    };
    let mut scratch = DivScratch::default();
    let mut workspace = HgcdWorkspace::default();
    let input_len = u.limbs().len();
    let high_len = input_len.div_ceil(3);
    let expected_limit = input_len
        .wrapping_sub(high_len)
        .wrapping_add(high_len.wrapping_add(1) >> 1)
        .wrapping_add(2);

    assert!(
        hgcd_block::<false>(&mut u, &mut v, &mut scratch, &mut workspace, &mut 0),
        "forced recursive HGCD block must make progress"
    );
    assert!(
        u.limbs().len() <= expected_limit && v.limbs().len() <= expected_limit,
        "recursive HGCD block left ({}, {}) limbs above its prefix reduction bound {expected_limit}",
        u.limbs().len(),
        v.limbs().len()
    );
    assert_eq!(u.gcd(&v), expected);
}

#[test]
fn recursive_half_gcd_handles_a_wide_exact_quotient() {
    // Construct u = v + r and v = q*r + s with a 75-limb q. The first exact
    // Euclidean step has quotient one; the next has a multiprecision quotient
    // while r is still above the 128-limb half-GCD target.
    let mut large_remainder = InternalMpUint::one();
    large_remainder.shl_assign(180_usize.wrapping_mul(LIMB_BITS));
    large_remainder.add_assign(&InternalMpUint::from_limb(17));
    let mut wide_quotient = InternalMpUint::one();
    wide_quotient.shl_assign(75_usize.wrapping_mul(LIMB_BITS));
    wide_quotient.add_assign(&InternalMpUint::from_limb(3));
    let mut small_remainder = InternalMpUint::one();
    small_remainder.shl_assign(127_usize.wrapping_mul(LIMB_BITS));
    small_remainder.add_assign(&InternalMpUint::from_limb(5));
    let mut right = wide_quotient.mul(&large_remainder);
    right.add_assign(&small_remainder);
    let mut left = right.clone();
    left.add_assign(&large_remainder);
    assert_eq!(left.limbs().len(), right.limbs().len());
    let expected = left.gcd(&right);
    let original_len = left.limbs().len();
    let mut scratch = DivScratch::default();
    let mut workspace = HgcdWorkspace::default();

    assert!(hgcd_block::<false>(
        &mut left,
        &mut right,
        &mut scratch,
        &mut workspace,
        &mut 0,
    ));
    assert!(left.cmp(&right).is_ge());
    assert!(left.limbs().len() <= original_len);
    assert!(right.limbs().len() < original_len);
    assert_eq!(left.gcd(&right), expected);
}

#[test]
#[cfg_attr(
    miri,
    ignore = "2048 to 8192-limb dispatch comparisons require native execution."
)]
fn production_dynamic_splits_match_lehmer() {
    for len in [2_048_usize, 4_096, 8_192] {
        let left = InternalMpUint::from_limbs(
            (0..len)
                .map(|index| {
                    index
                        .wrapping_mul(Limb::try_from(40_503_u32).expect("constant fits every limb"))
                        | 1
                })
                .collect(),
        );
        let right = InternalMpUint::from_limbs(
            (0..len)
                .map(|index| {
                    index
                        .wrapping_mul(Limb::try_from(34_283_u32).expect("constant fits every limb"))
                        | 1
                })
                .collect(),
        );

        assert_eq!(
            left.gcd(&right),
            Gcd::compute_lehmer(&left, &right),
            "production dynamic HGCD disagreed with Lehmer at {len} limbs"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 24 }))]

    #[test]
    #[cfg_attr(miri, ignore = "128 to 512-limb tier comparison uses native execution; bounded GCD and matrix properties run under Miri.")]
    fn half_gcd_matches_lehmer(
        a_limbs in proptest::collection::vec(any::<Limb>(), 128..=512),
        b_limbs in proptest::collection::vec(any::<Limb>(), 128..=512),
    ) {
        let a = InternalMpUint::from_limbs(a_limbs);
        let b = InternalMpUint::from_limbs(b_limbs);

        let mut workspace = HgcdWorkspace::default();
        prop_assert_eq!(Gcd::compute_half_gcd(&a, &b, &mut workspace), Gcd::compute_lehmer(&a, &b));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    #[cfg_attr(miri, ignore = "The configured production HGCD crossover requires large native operands.")]
    fn half_gcd_matches_lehmer_across_the_production_crossover(
        a_limbs in proptest::collection::vec(
            any::<Limb>(),
            HGCD_CROSSOVER_THRESHOLD.saturating_sub(256)
                ..=HGCD_CROSSOVER_THRESHOLD.saturating_add(256),
        ),
        b_limbs in proptest::collection::vec(
            any::<Limb>(),
            HGCD_CROSSOVER_THRESHOLD.saturating_sub(256)
                ..=HGCD_CROSSOVER_THRESHOLD.saturating_add(256),
        ),
    ) {
        let a = InternalMpUint::from_limbs(a_limbs);
        let b = InternalMpUint::from_limbs(b_limbs);

        let mut workspace = HgcdWorkspace::default();
        prop_assert_eq!(Gcd::compute_half_gcd(&a, &b, &mut workspace), Gcd::compute_lehmer(&a, &b));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 512 }))]
    #[test]
    fn gcd_lcm_and_bezout_identities_hold_across_storage_widths(
        a_limbs in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 8 }),
        b_limbs in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 8 }),
    ) {
        let a = InternalMpUint::from_limbs(a_limbs);
        let b = InternalMpUint::from_limbs(b_limbs);

        let g = a.gcd(&b);

        // gcd divides both inputs
        if !g.is_zero() {
            let r_a = a.rem(&g);
            prop_assert!(r_a.is_zero());
            let r_b = b.rem(&g);
            prop_assert!(r_b.is_zero());
        }

        // Commutativity
        prop_assert_eq!(&g, &b.gcd(&a));

        // is_coprime matches gcd == 1
        prop_assert_eq!(a.is_coprime(&b), g.is_one());

        // lcm property: lcm(a,b) * gcd(a,b) == a * b
        let l = a.lcm(&b);
        let lhs = l.mul(&g);
        let rhs = a.mul(&b);
        prop_assert_eq!(lhs, rhs);

        // extended_gcd properties
        if !b.is_zero() {
            let result = a.extended_gcd(&b);
            prop_assert!(result.gcd == g);
            let ax = a.mul(&result.x_magnitude);
            let by = b.mul(&result.y_magnitude);
            let identity = if result.x_is_positive { ax.sub(&by) } else { by.sub(&ax) };
            prop_assert_eq!(identity, g);
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "The 65000-limb fixture verifies production SSA-scale HGCD and requires native execution."
)]
fn production_hgcd_preserves_a_known_common_factor_at_ssa_scale() {
    // Both operands share the odd factor p; the coprime odd cofactors force the
    // gcd to be exactly p. At this width the matrix applications dispatch into
    // the fused shared-operand SSA tier, which the smaller tests never reach.
    const COFACTOR_LEN: usize = 65_000;
    let mut p = InternalMpUint::one();
    p.shl_assign(4_000_usize.wrapping_mul(LIMB_BITS));
    p.sub_assign(&InternalMpUint::one());

    let a: Vec<Limb> = (0..COFACTOR_LEN)
        .map(|index| {
            Limb::try_from(index)
                .expect("test index fits every limb")
                .wrapping_mul(Limb::try_from(40_503_u32).expect("constant fits every limb"))
                | 1
        })
        .collect();
    let cofactor_a = InternalMpUint::from_limbs(a);
    let mut cofactor_b = cofactor_a.clone();
    // b = a + 2: both odd, consecutive-odd => coprime.
    cofactor_b.add_assign(&InternalMpUint::from_limb(2));

    let left = p.mul(&cofactor_a);
    let right = p.mul(&cofactor_b);
    assert_eq!(left.limbs().len(), right.limbs().len());
    let result = left.gcd(&right);
    assert_eq!(result, p, "production HGCD lost the common factor");
}

#[test]
#[expect(
    clippy::similar_names,
    reason = "Dual u/v test variables for streaming vs interleaved verification"
)]
fn lehmer_update_stream_matches_interleaved_across_boundary_cases() {
    let mut u1 = InternalMpUint::from_limbs(alloc::vec![123_456_789, 987_654_321, 111_222_333]);
    let mut v1 = InternalMpUint::from_limbs(alloc::vec![555_666_777, 888_999_000, 444_333_222]);
    let mut u2 = u1.clone();
    let mut v2 = v1.clone();
    let mut next_u1 = InternalMpUint::zero();
    let mut next_v1 = InternalMpUint::zero();
    let mut next_u2 = InternalMpUint::zero();
    let mut next_v2 = InternalMpUint::zero();

    let (u0, v0, u1_c, v1_c, even) = (3, 2, 1, 1, true);
    let ok1 = lehmer_update::<false>(
        &mut u1,
        &mut v1,
        &mut next_u1,
        &mut next_v1,
        u0,
        v0,
        u1_c,
        v1_c,
        even,
    );
    let ok2 = lehmer_update::<true>(
        &mut u2,
        &mut v2,
        &mut next_u2,
        &mut next_v2,
        u0,
        v0,
        u1_c,
        v1_c,
        even,
    );
    assert_eq!(ok1, ok2);
    assert_eq!(next_u1, next_u2);
    assert_eq!(next_v1, next_v2);
}

#[test]
fn hgcd_block_succeeds_when_operands_differ_by_one_limb() {
    let mut left = InternalMpUint::one();
    left.shl_assign(200_usize.wrapping_mul(LIMB_BITS));
    left.add_assign(&InternalMpUint::from_limb(23));

    let mut right = InternalMpUint::one();
    right.shl_assign(199_usize.wrapping_mul(LIMB_BITS));
    right.add_assign(&InternalMpUint::from_limb(17));

    assert_eq!(left.limbs().len(), 201);
    assert_eq!(right.limbs().len(), 200);

    let expected = left.gcd(&right);
    let mut scratch = DivScratch::default();
    let mut workspace = HgcdWorkspace::default();

    assert!(
        hgcd_block::<false>(&mut left, &mut right, &mut scratch, &mut workspace, &mut 0),
        "HGCD block must progress when operands differ by one limb"
    );
    assert_eq!(left.gcd(&right), expected);
}
