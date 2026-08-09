//! Number theory and powers fuzzing checks for `MpInt`.

use mp_anafis::MpInt;
use rug::{Integer, ops::Pow};

pub fn fuzz_all(mp_a: &MpInt, mp_b: &MpInt, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 2 {
        0 => {
            let exp = (mp_b.to_i64().unwrap_or(0).unsigned_abs() % 16) as u32;
            let mp = mp_a.pow(exp);
            let rug = Integer::from(rug_a.pow(exp));
            assert_eq!(format!("{mp:x}"), format!("{rug:x}"));
        }
        _ => {
            if *rug_b != 0 && let Some((g, _x, _y)) = mp_a.extended_gcd(mp_b) {
                let g_abs = g.abs();
                let rug_g = rug_a.clone().gcd(rug_b).abs();
                assert_eq!(format!("{g_abs:x}"), format!("{rug_g:x}"));
            }
        }
    }
}
