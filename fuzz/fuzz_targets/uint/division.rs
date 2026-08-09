//! Division modes and divisibility fuzzing checks for `MpUint`.

use mp_anafis::MpUint;
use rug::Integer;

pub fn fuzz_all(mp_a: &MpUint, mp_b: &MpUint, rug_b: &Integer, selector: u8) {
    match selector % 3 {
        0 => {
            if *rug_b != 0 {
                assert_eq!(mp_a.div_trunc(mp_b), mp_a.clone() / mp_b.clone());
                assert_eq!(mp_a.rem_trunc(mp_b), mp_a.clone() % mp_b.clone());
                assert_eq!(mp_a.checked_div_trunc(mp_b), Some(mp_a.clone() / mp_b.clone()));
                assert_eq!(mp_a.checked_rem_trunc(mp_b), Some(mp_a.clone() % mp_b.clone()));
                assert_eq!(mp_a.div_euclid(mp_b), mp_a.clone() / mp_b.clone());
                assert_eq!(mp_a.rem_euclid(mp_b), mp_a.clone() % mp_b.clone());
                assert_eq!(mp_a.checked_div_euclid(mp_b), Some(mp_a.clone() / mp_b.clone()));
                assert_eq!(mp_a.checked_rem_euclid(mp_b), Some(mp_a.clone() % mp_b.clone()));
                if let Some((q, r)) = mp_a.div_rem_euclid(mp_b) {
                    assert_eq!(q, mp_a.clone() / mp_b.clone());
                    assert_eq!(r, mp_a.clone() % mp_b.clone());
                }
            } else {
                assert_eq!(mp_a.checked_div_trunc(mp_b), None);
                assert_eq!(mp_a.checked_rem_trunc(mp_b), None);
                assert_eq!(mp_a.checked_div_euclid(mp_b), None);
                assert_eq!(mp_a.checked_rem_euclid(mp_b), None);
                assert_eq!(mp_a.div_rem_euclid(mp_b), None);
            }
        }
        1 => {
            if *rug_b != 0 {
                assert_eq!(mp_a.div_floor(mp_b), mp_a.clone() / mp_b.clone());
                assert_eq!(mp_a.mod_floor(mp_b), mp_a.clone() % mp_b.clone());
                assert_eq!(mp_a.checked_div_floor(mp_b), Some(mp_a.clone() / mp_b.clone()));
                assert_eq!(mp_a.checked_mod_floor(mp_b), Some(mp_a.clone() % mp_b.clone()));
                if let Some((q, r)) = mp_a.div_rem_floor(mp_b) {
                    assert_eq!(q, mp_a.clone() / mp_b.clone());
                    assert_eq!(r, mp_a.clone() % mp_b.clone());
                }
                let ceil = mp_a.div_ceil(mp_b);
                assert_eq!(mp_a.checked_div_ceil(mp_b), Some(ceil.clone()));
                if (mp_a.clone() % mp_b.clone()) == MpUint::zero() {
                    assert_eq!(ceil, mp_a.clone() / mp_b.clone());
                } else {
                    assert_eq!(ceil, (mp_a.clone() / mp_b.clone()) + MpUint::one());
                }
            } else {
                assert_eq!(mp_a.checked_div_floor(mp_b), None);
                assert_eq!(mp_a.checked_mod_floor(mp_b), None);
                assert_eq!(mp_a.div_rem_floor(mp_b), None);
                assert_eq!(mp_a.checked_div_ceil(mp_b), None);
            }
        }
        _ => {
            if *rug_b != 0 {
                let div = mp_a.is_divisible_by(mp_b);
                assert_eq!(div, (mp_a.clone() % mp_b.clone()) == MpUint::zero());
                let div_of = mp_b.is_divisor_of(mp_a);
                assert_eq!(div_of, div);
            }
        }
    }
}
