//! Bitwise operators, shifts, and bit manipulation fuzzing for `MpUint`.

use mp_anafis::MpUint;
use rug::Integer;

pub fn fuzz_all(mp_a: &MpUint, mp_b: &MpUint, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 9 {
        0 => {
            let mp = mp_a.clone() & mp_b.clone();
            let rug = rug_a.clone() & rug_b.clone();
            assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
        }
        1 => {
            let mp = mp_a.clone() | mp_b.clone();
            let rug = rug_a.clone() | rug_b.clone();
            assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
        }
        2 => {
            let mp = mp_a.clone() ^ mp_b.clone();
            let rug = rug_a.clone() ^ rug_b.clone();
            assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
        }
        3 => {
            let shift = (mp_b.to_u64().unwrap_or(0) % 1024) as u32;
            let mp = mp_a.clone() << shift;
            let rug = rug_a.clone() << shift;
            assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
        }
        4 => {
            let shift = (mp_b.to_u64().unwrap_or(0) % 1024) as u32;
            let mp = mp_a.clone() >> shift;
            let rug = rug_a.clone() >> shift;
            assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
        }
        5 => {
            let shift = (mp_b.to_u64().unwrap_or(0) % 512) as usize;
            assert_eq!(mp_a.checked_shl(shift), Some(mp_a.clone() << shift as u32));
            assert_eq!(mp_a.wrapping_shl(shift), mp_a.clone() << shift as u32);
            assert_eq!(mp_a.saturating_shl(shift), mp_a.clone() << shift as u32);
        }
        6 => {
            let mp = mp_a.trailing_zeros();
            let rug = rug_a.find_one(0);
            match rug {
                Some(r) => assert_eq!(mp, r as usize),
                None => assert_eq!(mp, 0),
            }
            let mp_ones = mp_a.count_ones();
            let rug_ones = rug_a.count_ones().unwrap_or(0);
            assert_eq!(mp_ones, rug_ones as usize);
            assert_eq!(mp_a.significant_bits(), rug_a.significant_bits() as usize);
            assert_eq!(
                mp_a.significant_bits(),
                if mp_a.is_zero() { 0 } else { mp_a.to_string_radix(2).len() }
            );
        }
        7 => {
            let bit = (mp_b.to_u64().unwrap_or(0) % 256) as usize;
            let set = mp_a.set_bit(bit);
            assert!(set.get_bit(bit));
            let cleared = set.clear_bit(bit);
            assert!(!cleared.get_bit(bit));
            let toggled = cleared.toggle_bit(bit);
            assert!(toggled.get_bit(bit));
        }
        _ => {
            let swapped = mp_a.swap_bytes();
            let _back = swapped.swap_bytes();
        }
    }
}
