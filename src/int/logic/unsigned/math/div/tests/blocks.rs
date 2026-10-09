//! Newton quotient and remainder identities across uneven block geometries.

use alloc::vec;

use super::{DivScratch, Division, InternalMpUint, Limb};

#[test]
fn newton_short_and_uneven_quotients_preserve_all_output_policies() {
    for n in [3_usize, 4, 7, 32, 64, 65, 127, 128, 129, 257] {
        if cfg!(miri) && n > 7 {
            continue;
        }
        let mut limbs = vec![Limb::MAX; n];
        *limbs.first_mut().expect("nonempty divisor") = 7;
        *limbs.last_mut().expect("nonempty divisor") =
            (Limb::MAX >> 1).checked_add(1).expect("high bit fits");
        let divisor = InternalMpUint::from_limbs(limbs);
        let third = n.div_euclid(3);
        let long = n
            .checked_mul(5)
            .expect("bounded quotient width")
            .div_euclid(2);
        let short_tail = if n <= 65 {
            n.checked_mul(n.checked_add(1).expect("bounded block count"))
                .and_then(|width| width.checked_add(1))
                .expect("bounded long quotient")
        } else {
            n.checked_mul(9)
                .and_then(|width| width.checked_add(1))
                .expect("bounded long quotient")
        };
        let widths = [
            1,
            third.saturating_sub(1).max(1),
            third.max(1),
            third.checked_add(1).expect("bounded width"),
            n.checked_sub(1).expect("n>=3"),
            n,
            n.checked_add(1).expect("bounded width"),
            long.checked_sub(1).expect("long>=7"),
            long,
            long.checked_add(1).expect("bounded width"),
            short_tail,
        ];
        let mut scratch = DivScratch::default();
        let mut quotient = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        for qn in widths {
            let expected = InternalMpUint::from_limbs(vec![Limb::MAX; qn]);
            let product = divisor.mul(&expected);
            for residue in [
                InternalMpUint::zero(),
                InternalMpUint::one(),
                divisor.sub(&InternalMpUint::one()),
            ] {
                let numerator = product.add(&residue);
                Division::newton::<true, true, false>(
                    &numerator,
                    &divisor,
                    &mut quotient,
                    &mut remainder,
                    &mut scratch,
                );
                assert_eq!(quotient, expected, "full quotient n={n}, qn={qn}");
                assert_eq!(remainder, residue, "full remainder n={n}, qn={qn}");
                Division::newton::<true, false, false>(
                    &numerator,
                    &divisor,
                    &mut quotient,
                    &mut remainder,
                    &mut scratch,
                );
                assert_eq!(quotient, expected, "quotient-only n={n}, qn={qn}");
                Division::newton::<false, true, false>(
                    &numerator,
                    &divisor,
                    &mut quotient,
                    &mut remainder,
                    &mut scratch,
                );
                assert_eq!(remainder, residue, "remainder-only n={n}, qn={qn}");
                if residue.is_zero() {
                    Division::newton::<true, false, true>(
                        &numerator,
                        &divisor,
                        &mut quotient,
                        &mut remainder,
                        &mut scratch,
                    );
                    assert_eq!(quotient, expected, "exact quotient n={n}, qn={qn}");
                }
            }
        }
    }
}
