//! Bitwise operators, shifts, and bit manipulation fuzzing for `MpInt`.

use mp_anafis::MpInt;
use rug::Integer;

pub fn fuzz_all(mp_a: &MpInt, mp_b: &MpInt, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 7 {
        0 => {
            let mp = mp_a.clone() & mp_b.clone();
            let rug = rug_a.clone() & rug_b.clone();
            assert_eq!(format!("{mp:x}"), format!("{rug:x}"));
        }
        1 => {
            let mp = mp_a.clone() | mp_b.clone();
            let rug = rug_a.clone() | rug_b.clone();
            assert_eq!(format!("{mp:x}"), format!("{rug:x}"));
        }
        2 => {
            let mp = mp_a.clone() ^ mp_b.clone();
            let rug = rug_a.clone() ^ rug_b.clone();
            assert_eq!(format!("{mp:x}"), format!("{rug:x}"));
        }
        3 => {
            let shift = (mp_b.to_i64().unwrap_or(0).unsigned_abs() % 1024) as u32;
            let mp = mp_a.clone() << shift;
            let rug = rug_a.clone() << shift;
            assert_eq!(format!("{mp:x}"), format!("{rug:x}"));
        }
        4 => {
            let shift = (mp_b.to_i64().unwrap_or(0).unsigned_abs() % 1024) as u32;
            let mp = mp_a.clone() >> shift;
            let rug = rug_a.clone() >> shift;
            assert_eq!(format!("{mp:x}"), format!("{rug:x}"));
        }
        5 => {
            let shift = (mp_b.to_i64().unwrap_or(0).unsigned_abs() % 512) as usize;
            assert_eq!(mp_a.checked_shl(shift), Some(mp_a.clone() << shift as u32));
            assert_eq!(mp_a.wrapping_shl(shift), mp_a.clone() << shift as u32);
            assert_eq!(mp_a.saturating_shl(shift), mp_a.clone() << shift as u32);
        }
        _ => {
            let bit = (mp_b.to_i64().unwrap_or(0).unsigned_abs() % 256) as usize;
            let set = mp_a.set_bit(bit);
            assert!(set.get_bit(bit));
            let cleared = set.clear_bit(bit);
            assert!(!cleared.get_bit(bit));
            let toggled = cleared.toggle_bit(bit);
            assert!(toggled.get_bit(bit));
        }
    }
}
