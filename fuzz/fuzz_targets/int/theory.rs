//! Number theory and powers fuzzing checks for `ArbiInt`.

use arbi_anafis::ArbiInt;
use rug::{Integer, ops::Pow};

pub fn fuzz_all(arbi_a: &ArbiInt, arbi_b: &ArbiInt, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 2 {
        0 => {
            let exp = (arbi_b.to_i64().unwrap_or(0).unsigned_abs() % 16) as u32;
            let arbi = arbi_a.pow(exp);
            let rug = Integer::from(rug_a.pow(exp));
            assert_eq!(format!("{arbi:x}"), format!("{rug:x}"));
        }
        _ => {
            if *rug_b != 0 && let Some((g, _x, _y)) = arbi_a.extended_gcd(arbi_b) {
                let g_abs = g.abs();
                let rug_g = rug_a.clone().gcd(rug_b).abs();
                assert_eq!(format!("{g_abs:x}"), format!("{rug_g:x}"));
            }
        }
    }
}
