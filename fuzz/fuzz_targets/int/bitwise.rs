//! Bitwise operators, shifts, and bit manipulation fuzzing for `ArbiInt`.

use arbi_anafis::ArbiInt;
use rug::Integer;

pub fn fuzz_all(arbi_a: &ArbiInt, arbi_b: &ArbiInt, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 7 {
        0 => {
            let arbi = arbi_a.clone() & arbi_b.clone();
            let rug = rug_a.clone() & rug_b.clone();
            assert_eq!(format!("{arbi:x}"), format!("{rug:x}"));
        }
        1 => {
            let arbi = arbi_a.clone() | arbi_b.clone();
            let rug = rug_a.clone() | rug_b.clone();
            assert_eq!(format!("{arbi:x}"), format!("{rug:x}"));
        }
        2 => {
            let arbi = arbi_a.clone() ^ arbi_b.clone();
            let rug = rug_a.clone() ^ rug_b.clone();
            assert_eq!(format!("{arbi:x}"), format!("{rug:x}"));
        }
        3 => {
            let shift = (arbi_b.to_i64().unwrap_or(0).unsigned_abs() % 1024) as u32;
            let arbi = arbi_a.clone() << shift;
            let rug = rug_a.clone() << shift;
            assert_eq!(format!("{arbi:x}"), format!("{rug:x}"));
        }
        4 => {
            let shift = (arbi_b.to_i64().unwrap_or(0).unsigned_abs() % 1024) as u32;
            let arbi = arbi_a.clone() >> shift;
            let rug = rug_a.clone() >> shift;
            assert_eq!(format!("{arbi:x}"), format!("{rug:x}"));
        }
        5 => {
            let shift = (arbi_b.to_i64().unwrap_or(0).unsigned_abs() % 512) as usize;
            assert_eq!(arbi_a.checked_shl(shift), Some(arbi_a.clone() << shift as u32));
            assert_eq!(arbi_a.wrapping_shl(shift), arbi_a.clone() << shift as u32);
            assert_eq!(arbi_a.saturating_shl(shift), arbi_a.clone() << shift as u32);
        }
        _ => {
            let bit = (arbi_b.to_i64().unwrap_or(0).unsigned_abs() % 256) as usize;
            let set = arbi_a.set_bit(bit);
            assert!(set.get_bit(bit));
            let cleared = set.clear_bit(bit);
            assert!(!cleared.get_bit(bit));
            let toggled = cleared.toggle_bit(bit);
            assert!(toggled.get_bit(bit));
        }
    }
}
