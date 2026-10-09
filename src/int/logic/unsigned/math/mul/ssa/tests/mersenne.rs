//! Mersenne products, admission rollback, and reserved output initialization.

#![expect(
    unsafe_code,
    reason = "Reserved-capacity fixtures initialize suffix sentinels before inspecting their MaybeUninit storage after exact output writes"
)]

use super::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 4 }))]

    #[test]
    fn modular_products_match_independent_digit_reduction(
        left in prop::collection::vec(any::<Limb>(), if cfg!(miri) { 4..=12 } else { 4096..=4100 }),
        right in prop::collection::vec(any::<Limb>(), if cfg!(miri) { 1..=4 } else { 1024..=4096 }),
    ) {
        let minimum = left.len().checked_add(1).expect("test modulus fits");
        let mut storage = ScratchBuffer::acquire(2);
        storage.extend_from_slice(&[17, 29]);
        if Multiplication::try_mul_mod_bnm1::<true, false>(&left, &right, minimum, &mut storage, &mut MulScratch::default()) {
            prop_assert!(storage.len() >= minimum);
            let expected = reduce_product(&left, &right, storage.len());
            if storage.iter().all(|&limb| limb == Limb::MAX) { storage.fill(0); }
            prop_assert_eq!(storage.as_slice(), expected);
        } else {
            prop_assert_eq!(storage.as_slice(), &[17, 29]);
        }
    }
}

#[test]
fn rejected_widths_and_consumer_cutoffs_preserve_destination_storage() {
    let left = vec![Limb::MAX; 192];
    let right = vec![Limb::MAX; 128];
    for minimum in [127, 191] {
        let mut storage = ScratchBuffer::acquire(2);
        storage.extend_from_slice(&[17, 29]);
        assert!(!Multiplication::try_mul_mod_bnm1::<true, false>(
            &left,
            &right,
            minimum,
            &mut storage,
            &mut MulScratch::default()
        ));
        assert_eq!(storage.as_slice(), &[17, 29]);
    }
    let mut storage = ScratchBuffer::acquire(2);
    assert!(Multiplication::try_mul_mod_bnm1::<true, true>(
        &left,
        &right,
        193,
        &mut storage,
        &mut MulScratch::default()
    ));
    assert!(
        storage.len() >= 193
            && storage.len()
                < left
                    .len()
                    .checked_add(right.len())
                    .expect("test product fits")
    );
    assert!(storage.capacity() >= left.len().checked_add(right.len()).expect("product fits"));
    if MUL_MOD_BNM1_THRESHOLD == 0 || MUL_MOD_BNM1_THRESHOLD > 193 {
        let expected = storage.to_vec();
        let pointer = storage.as_ptr();
        let capacity = storage.capacity();
        assert!(!Multiplication::try_mul_mod_bnm1::<false, true>(
            &left,
            &right,
            193,
            &mut storage,
            &mut MulScratch::default()
        ));
        assert_eq!(storage.as_slice(), expected);
        assert_eq!(storage.as_ptr(), pointer);
        assert_eq!(storage.capacity(), capacity);
    }
}

#[test]
fn fresh_and_reused_reserved_outputs_initialize_only_their_active_spans() {
    for len in if cfg!(miri) {
        &[8_usize][..]
    } else {
        &[8_usize, 64, 193][..]
    } {
        let left = vec![Limb::MAX; *len];
        let right = vec![Limb::MAX - 1; len.checked_sub(2).expect("positive right width")];
        let minimum = len.checked_add(1).expect("modulus fits");
        let mut storage = ScratchBuffer::acquire(0);
        let mut scratch = MulScratch::default();
        assert!(Multiplication::try_mul_mod_bnm1::<true, false>(
            &left,
            &right,
            minimum,
            &mut storage,
            &mut scratch
        ));
        let width = storage.len();
        let required = scratch.buf.len();
        let expected = reduce_product(&left, &right, width);
        assert_eq!(storage.as_slice(), expected);
        for occupied in [
            0,
            1,
            width,
            required,
            required.checked_add(3).expect("suffix fits"),
        ] {
            scratch.buf = ScratchBuffer::acquire(required.checked_add(3).expect("suffix fits"));
            scratch.buf.resize(occupied, Limb::MAX);
            storage.reset_with_capacity(width.checked_add(3).expect("suffix fits"));
            storage.resize(width.checked_add(3).expect("suffix fits"), Limb::MAX);
            let output_pointer = storage.as_ptr();
            let scratch_pointer = scratch.buf.as_ptr();
            for _ in 0..2 {
                assert!(Multiplication::try_mul_mod_bnm1::<true, false>(
                    &left,
                    &right,
                    minimum,
                    &mut storage,
                    &mut scratch
                ));
                assert_eq!(storage.as_slice(), expected);
                assert_eq!(storage.as_ptr(), output_pointer);
                assert_eq!(scratch.buf.as_ptr(), scratch_pointer);
                for sentinel in storage.spare_capacity_mut().iter().take(3) {
                    // SAFETY: resize initialized three limbs above width. The
                    // operation writes exactly width limbs and its length commit
                    // retains those initialized excluded suffix sentinels.
                    assert_eq!(unsafe { sentinel.assume_init() }, Limb::MAX);
                }
            }
        }
    }
}

fn reduce_product(left: &[Limb], right: &[Limb], width: usize) -> Vec<Limb> {
    let mut exact = vec![
        0;
        left.len()
            .checked_add(right.len())
            .expect("test product fits")
    ];
    Schoolbook::mul(&mut exact, left, right);
    let mut expected = vec![0_usize; width];
    // B^width=1 reduces each digit independently, including end-around carry.
    for (position, &digit) in exact.iter().enumerate() {
        let mut index = position.rem_euclid(width);
        let mut carry = digit;
        while carry != 0 {
            let slot = expected.get_mut(index).expect("cyclic test digit");
            let (sum, overflow) = slot.overflowing_add(carry);
            *slot = sum;
            carry = Limb::from(overflow);
            index = index
                .checked_add(1)
                .expect("bounded index")
                .rem_euclid(width);
        }
    }
    if expected.iter().all(|&limb| limb == Limb::MAX) {
        expected.fill(0);
    }
    expected
}
