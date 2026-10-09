//! Borrowed division buffers and recursive quotient certificates.

use core::mem::MaybeUninit;

use proptest::prelude::*;

use super::{
    BURNIKEL_ZIEGLER_THRESHOLD, DivScratch, Division, InternalMpUint, Limb,
    NEWTON_RAPHSON_THRESHOLD, PreparedDivisor, ScratchBuffer,
};

#[test]
fn limb_shift_reuses_reserved_capacity_when_no_carry_is_produced() {
    for width in [0, 1, 4, 64, 65] {
        for shift in 0..Limb::BITS {
            let mut input = alloc::vec![Limb::MAX; width];
            if let Some(top) = input.last_mut() {
                *top = Limb::MAX >> shift;
            }
            let expected = InternalMpUint::from_limbs(input.clone())
                .shl(usize::try_from(shift).expect("sub-limb shift fits usize"));
            let mut output = ScratchBuffer::acquire(width);
            let capacity = output.capacity();
            let pointer = output.as_ptr();
            Division::shift_limbs_left::<false>(&input, shift, &mut output);
            assert_eq!(output.as_slice(), expected.limbs());
            assert_eq!(output.capacity(), capacity);
            assert_eq!(output.as_ptr(), pointer);
            let mut guarded = ScratchBuffer::acquire(width.checked_add(1).expect("guard width"));
            let guarded_capacity = guarded.capacity();
            let guarded_pointer = guarded.as_ptr();
            Division::shift_limbs_left::<true>(&input, shift, &mut guarded);
            let mut expected_guarded = expected.limbs().to_vec();
            expected_guarded.push(0);
            assert_eq!(guarded.as_slice(), expected_guarded.as_slice());
            assert_eq!(guarded.capacity(), guarded_capacity);
            assert_eq!(guarded.as_ptr(), guarded_pointer);
        }
    }
}

#[test]
fn division_outputs_cover_normalization_and_tier_boundaries() {
    let mut scratch = DivScratch::default();
    let mut quotient = InternalMpUint::zero();
    let mut remainder = InternalMpUint::from_limb(17);
    for width in [
        2,
        3,
        BURNIKEL_ZIEGLER_THRESHOLD - 1,
        BURNIKEL_ZIEGLER_THRESHOLD,
        BURNIKEL_ZIEGLER_THRESHOLD + 1,
        NEWTON_RAPHSON_THRESHOLD - 1,
        NEWTON_RAPHSON_THRESHOLD,
        NEWTON_RAPHSON_THRESHOLD + 1,
    ] {
        if cfg!(miri) && width > 65 {
            continue;
        }
        for top in [1, Limb::MAX >> 1, Limb::MAX] {
            let mut limbs = alloc::vec![Limb::MAX; width];
            *limbs.last_mut().expect("nonempty divisor") = top;
            let divisor = InternalMpUint::from_limbs(limbs);
            for digits in [1, width >> 1, width, width + 1] {
                let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; digits]);
                let residue = divisor.sub(&InternalMpUint::one());
                let numerator = divisor.mul(&expected).add(&residue);
                Division::div_rem_into(
                    &numerator,
                    &divisor,
                    &mut quotient,
                    &mut remainder,
                    &mut scratch,
                );
                assert_eq!(quotient, expected);
                assert_eq!(remainder, residue);
                Division::div_into::<true, false>(
                    &numerator,
                    &divisor,
                    &mut quotient,
                    &mut scratch,
                );
                assert_eq!(quotient, expected);
                assert_eq!(
                    remainder, residue,
                    "quotient-only output preserves remainder storage"
                );
            }
            Division::div_rem_into(
                &divisor,
                &divisor,
                &mut quotient,
                &mut remainder,
                &mut scratch,
            );
            assert!(quotient.is_one() && remainder.is_zero());
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Borrowed normalization is checked at the configured Newton crossover, which requires native execution."
)]
fn recursive_division_borrows_already_normalized_divisors() {
    let width = NEWTON_RAPHSON_THRESHOLD;
    let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
    let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
    let numerator = divisor.mul(&expected);
    let mut quotient = InternalMpUint::zero();
    let mut remainder = InternalMpUint::zero();
    let mut scratch = DivScratch::default();
    Division::burnikel_ziegler::<true>(
        &numerator,
        &divisor,
        &mut quotient,
        &mut remainder,
        &mut scratch,
    );
    assert_eq!(quotient, expected);
    assert!(remainder.is_zero());
    assert_eq!(scratch.v_norm.capacity(), 0);
    Division::newton::<true, true, false>(
        &numerator,
        &divisor,
        &mut quotient,
        &mut remainder,
        &mut scratch,
    );
    assert_eq!(quotient, expected);
    assert!(remainder.is_zero());
    assert_eq!(scratch.newton_v_norm.capacity(), 0);
}

#[test]
fn guarded_block_repair_propagates_borrow_without_a_product_guard() {
    for (width, digits) in [(5_usize, 2_usize), (7, 3)] {
        let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        let prepared = PreparedDivisor::new(divisor.limbs());
        let mut scratch = DivScratch::default();
        for fill in [1, Limb::MAX] {
            let expected = InternalMpUint::from_limbs(alloc::vec![fill; digits]);
            let product = divisor.mul(&expected);
            for residue in [
                InternalMpUint::zero(),
                divisor.shr(1),
                divisor.sub(&InternalMpUint::one()),
            ] {
                let numerator = product.add(&residue);
                let mut window = numerator.limbs().to_vec();
                window.resize(width.checked_add(digits).expect("bounded block"), 0);
                let mut output = alloc::vec![23; digits];
                let mut workspace =
                    alloc::vec![MaybeUninit::uninit(); width.checked_add(2).expect("sentinels")];
                let (low, upper) = workspace.split_at_mut(1);
                let (repair, high) = upper.split_at_mut(width);
                let low_guard = low.first_mut().expect("low sentinel").write(29);
                let high_guard = high.first_mut().expect("high sentinel").write(29);
                let high_bit = Division::burnikel_div_block::<false>(
                    &mut output,
                    &mut window,
                    divisor.limbs(),
                    repair,
                    &mut scratch.mul_scratch,
                    &prepared,
                );
                assert_eq!(high_bit, 0);
                assert_eq!(InternalMpUint::from_limbs(output), expected);
                assert_eq!((*low_guard, *high_guard), (29, 29));
            }
        }
    }
}

proptest! {
    #[test]
    fn limb_shift_preserves_value_and_padded_width(
        mut input in proptest::collection::vec(
            prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 0..=129,
        ),
        padding in 0_usize..=4,
        shift in 0_u32..Limb::BITS,
        reserved in prop_oneof![
            Just(0_usize), Just(1), Just(64), Just(128), Just(135), 0_usize..=135,
        ],
    ) {
        let width = input.len().checked_add(padding).expect("bounded padded width");
        input.resize(width, 0);
        let expected = InternalMpUint::from_limbs(input.clone())
            .shl(usize::try_from(shift).expect("sub-limb shift fits usize"));
        let mut output = ScratchBuffer::acquire(reserved);
        output.resize(reserved, Limb::MAX);
        Division::shift_limbs_left::<false>(&input, shift, &mut output);
        prop_assert_eq!(output.len(), width.max(expected.limbs().len()));
        prop_assert_eq!(&InternalMpUint::from_limbs_slice(&output), &expected);
        Division::shift_limbs_left::<true>(&input, shift, &mut output);
        prop_assert_eq!(output.len(), width.checked_add(1).expect("bounded guard width"));
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&output), expected);
    }

    #[test]
    fn recursive_remainders_preserve_outputs_and_sentinels(
        mut denominator in proptest::collection::vec(any::<Limb>(), 2..=300),
        mut digits in proptest::collection::vec(any::<Limb>(), 1..=600),
        guard in prop_oneof![Just(0), Just(1), Just(2), Just(Limb::MAX), any::<Limb>()],
    ) {
        *denominator.last_mut().expect("nonempty divisor") |= 1 << Limb::BITS.wrapping_sub(1);
        if let Some(digit) = digits.get_mut(1) {
            *digit = guard;
        }
        let divisor = InternalMpUint::from_limbs(denominator);
        let expected = InternalMpUint::from_limbs(digits);
        let product = divisor.mul(&expected);
        let mut scratch = DivScratch::default();
        for residue in [InternalMpUint::zero(), InternalMpUint::one(), divisor.sub(&InternalMpUint::one())] {
            let numerator = product.add(&residue);
            let n = divisor.limbs().len();
            let num_len = numerator.limbs().len().max(n).checked_add(1).expect("guarded numerator");
            let q_len = num_len.checked_sub(n).expect("positive quotient span");
            let mut storage = alloc::vec![19; num_len.checked_add(2).expect("input sentinels")];
            let window = storage.get_mut(1..num_len.checked_add(1).expect("bounded input")).expect("input span");
            window.fill(0);
            window.get_mut(..numerator.limbs().len()).expect("fitting input").copy_from_slice(numerator.limbs());
            let mut output = alloc::vec![23; q_len.checked_add(2).expect("output sentinels")];
            let quotient = output.get_mut(1..q_len.checked_add(1).expect("bounded output")).expect("output span");
            let mut products = alloc::vec![MaybeUninit::uninit(); n.checked_add(2).expect("product sentinels")];
            let (low, upper) = products.split_at_mut(1);
            let (work, high) = upper.split_at_mut(n);
            let low_guard = low.first_mut().expect("low product sentinel").write(29);
            let high_guard = high.first_mut().expect("high product sentinel").write(29);
            Division::burnikel_div_rem_normalized::<true>(
                window, divisor.limbs(), quotient, work, &mut scratch.mul_scratch,
            );
            prop_assert_eq!(InternalMpUint::from_limbs(quotient.to_vec()), expected.clone());
            prop_assert_eq!(InternalMpUint::from_limbs(window.get(..n).expect("remainder").to_vec()), residue);
            window.fill(0);
            window.get_mut(..numerator.limbs().len()).expect("fitting input").copy_from_slice(numerator.limbs());
            quotient.fill(23);
            Division::burnikel_div_rem_normalized::<false>(
                window, divisor.limbs(), quotient, work, &mut scratch.mul_scratch,
            );
            prop_assert_eq!(InternalMpUint::from_limbs(quotient.to_vec()), expected.clone());
            prop_assert_eq!((storage.first(), storage.last()), (Some(&19), Some(&19)));
            prop_assert_eq!((output.first(), output.last()), (Some(&23), Some(&23)));
            prop_assert_eq!((*low_guard, *high_guard), (29, 29));
        }
    }

    #[test]
    fn recursive_quotient_certificate_handles_exact_and_adjacent_products(
        mut denominator in proptest::collection::vec(any::<Limb>(), 32..=257),
        digits in proptest::collection::vec(any::<Limb>(), 16..=600),
        top in prop_oneof![Just(1), Just(Limb::MAX), any::<Limb>()],
    ) {
        *denominator.last_mut().expect("nonempty divisor") = top.max(1);
        let divisor = InternalMpUint::from_limbs(denominator);
        let expected = InternalMpUint::from_limbs(digits);
        let product = divisor.mul(&expected);
        let mut scratch = DivScratch::default();
        let mut quotient = InternalMpUint::zero();
        let mut unused = InternalMpUint::from_limb(17);
        for remainder in [
            InternalMpUint::zero(), InternalMpUint::one(),
            divisor.shr(1), divisor.sub(&InternalMpUint::one()),
        ] {
            let numerator = product.add(&remainder);
            Division::burnikel_ziegler::<false>(
                &numerator, &divisor, &mut quotient, &mut unused, &mut scratch,
            );
            prop_assert_eq!(&quotient, &expected);
            prop_assert_eq!(&unused, &InternalMpUint::from_limb(17));
        }
        if !product.is_zero() {
            Division::burnikel_ziegler::<false>(
                &product.sub(&InternalMpUint::one()), &divisor,
                &mut quotient, &mut unused, &mut scratch,
            );
            prop_assert_eq!(quotient, expected.sub(&InternalMpUint::one()));
        }
    }
}

#[test]
fn normalized_division_preserves_guards_across_dispatch_boundaries() {
    let mut scratch = DivScratch::default();
    for width in [
        2,
        3,
        BURNIKEL_ZIEGLER_THRESHOLD - 1,
        BURNIKEL_ZIEGLER_THRESHOLD,
        BURNIKEL_ZIEGLER_THRESHOLD + 1,
        NEWTON_RAPHSON_THRESHOLD - 1,
        NEWTON_RAPHSON_THRESHOLD,
        NEWTON_RAPHSON_THRESHOLD + 1,
    ] {
        if cfg!(miri) && width > 65 {
            continue;
        }
        let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        for quotient_width in [1, width >> 1, width, width + 1] {
            let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; quotient_width]);
            let product = divisor.mul(&expected);
            for residue in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
                let numerator = product.add(&residue);
                let mut storage = alloc::vec![19; width + quotient_width + 2];
                let window = storage
                    .get_mut(1..width + quotient_width + 1)
                    .expect("guarded numerator");
                window.fill(0);
                window
                    .get_mut(..numerator.limbs().len())
                    .expect("fitting numerator")
                    .copy_from_slice(numerator.limbs());
                let mut output = alloc::vec![23; quotient_width + 2];
                let quotient = output
                    .get_mut(1..quotient_width + 1)
                    .expect("guarded quotient");
                Division::div_rem_normalized::<true>(
                    window,
                    divisor.limbs(),
                    quotient,
                    &mut scratch,
                );
                assert_eq!(InternalMpUint::from_limbs(quotient.to_vec()), expected);
                assert_eq!(
                    InternalMpUint::from_limbs(window.get(..width).expect("remainder").to_vec()),
                    residue
                );
                assert_eq!((storage.first(), storage.last()), (Some(&19), Some(&19)));
                assert_eq!((output.first(), output.last()), (Some(&23), Some(&23)));
            }
        }
    }
}
