//! Division modes and divisibility fuzzing checks for `ArbiInt`.

use arbi_anafis::ArbiInt;
use rug::{Integer, ops::DivRounding};

pub fn fuzz_all(
    arbi_a: &ArbiInt,
    arbi_b: &ArbiInt,
    rug_a: &Integer,
    rug_b: &Integer,
    selector: u8,
) {
    match selector % 3 {
        0 => {
            if *rug_b != 0 {
                assert_eq!(arbi_a.div_trunc(arbi_b), arbi_a.clone() / arbi_b.clone());
                assert_eq!(arbi_a.rem_trunc(arbi_b), arbi_a.clone() % arbi_b.clone());
                assert_eq!(arbi_a.checked_div_trunc(arbi_b), Some(arbi_a.clone() / arbi_b.clone()));
                assert_eq!(arbi_a.checked_rem_trunc(arbi_b), Some(arbi_a.clone() % arbi_b.clone()));
                let (rug_q, rug_r) = rug_a.clone().div_rem_euc(rug_b.clone());
                assert_eq!(format!("{}", arbi_a.div_euclid(arbi_b)), rug_q.to_string());
                assert_eq!(format!("{}", arbi_a.rem_euclid(arbi_b)), rug_r.to_string());
                assert_eq!(
                    arbi_a.checked_div_euclid(arbi_b).map(|value| value.to_string()),
                    Some(rug_q.to_string())
                );
                assert_eq!(
                    arbi_a.checked_rem_euclid(arbi_b).map(|value| value.to_string()),
                    Some(rug_r.to_string())
                );
                if let Some((q, r)) = arbi_a.div_rem_euclid(arbi_b) {
                    assert_eq!(q.to_string(), rug_q.to_string());
                    assert_eq!(r.to_string(), rug_r.to_string());
                }
            } else {
                assert_eq!(arbi_a.checked_div_trunc(arbi_b), None);
                assert_eq!(arbi_a.checked_rem_trunc(arbi_b), None);
                assert_eq!(arbi_a.checked_div_euclid(arbi_b), None);
                assert_eq!(arbi_a.checked_rem_euclid(arbi_b), None);
                assert_eq!(arbi_a.div_rem_euclid(arbi_b), None);
            }
        }
        1 => {
            if *rug_b != 0 {
                let (rug_q, rug_r) = rug_a.clone().div_rem_floor(rug_b.clone());
                assert_eq!(arbi_a.div_floor(arbi_b).to_string(), rug_q.to_string());
                assert_eq!(arbi_a.mod_floor(arbi_b).to_string(), rug_r.to_string());
                assert_eq!(
                    arbi_a.checked_div_floor(arbi_b).map(|value| value.to_string()),
                    Some(rug_q.to_string())
                );
                assert_eq!(
                    arbi_a.checked_mod_floor(arbi_b).map(|value| value.to_string()),
                    Some(rug_r.to_string())
                );
                if let Some((q, r)) = arbi_a.div_rem_floor(arbi_b) {
                    assert_eq!(q.to_string(), rug_q.to_string());
                    assert_eq!(r.to_string(), rug_r.to_string());
                }
                let ceil = arbi_a.div_ceil(arbi_b);
                let rug_ceil = rug_a.clone().div_ceil(rug_b);
                assert_eq!(ceil.to_string(), rug_ceil.to_string());
                assert_eq!(arbi_a.checked_div_ceil(arbi_b), Some(ceil.clone()));
            } else {
                assert_eq!(arbi_a.checked_div_floor(arbi_b), None);
                assert_eq!(arbi_a.checked_mod_floor(arbi_b), None);
                assert_eq!(arbi_a.div_rem_floor(arbi_b), None);
                assert_eq!(arbi_a.checked_div_ceil(arbi_b), None);
            }
        }
        _ => {
            if *rug_b != 0 {
                let div = arbi_a.is_divisible_by(arbi_b);
                assert_eq!(div, (arbi_a.clone() % arbi_b.clone()) == ArbiInt::zero());
                let div_of = arbi_b.is_divisor_of(arbi_a);
                assert_eq!(div_of, div);
            }
        }
    }
}
