//! Bitwise operators, shifts, and bit manipulation fuzzing for `ArbiUint`.

use arbi_anafis::ArbiUint;
use rug::Integer;

pub fn fuzz_all(arbi_a: &ArbiUint, arbi_b: &ArbiUint, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 9 {
        0 => {
            let arbi = arbi_a.clone() & arbi_b.clone();
            let rug = rug_a.clone() & rug_b.clone();
            assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
        }
        1 => {
            let arbi = arbi_a.clone() | arbi_b.clone();
            let rug = rug_a.clone() | rug_b.clone();
            assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
        }
        2 => {
            let arbi = arbi_a.clone() ^ arbi_b.clone();
            let rug = rug_a.clone() ^ rug_b.clone();
            assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
        }
        3 => {
            let shift = (arbi_b.to_u64().unwrap_or(0) % 1024) as u32;
            let arbi = arbi_a.clone() << shift;
            let rug = rug_a.clone() << shift;
            assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
        }
        4 => {
            let shift = (arbi_b.to_u64().unwrap_or(0) % 1024) as u32;
            let arbi = arbi_a.clone() >> shift;
            let rug = rug_a.clone() >> shift;
            assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
        }
        5 => {
            let shift = (arbi_b.to_u64().unwrap_or(0) % 512) as usize;
            assert_eq!(arbi_a.checked_shl(shift), Some(arbi_a.clone() << shift as u32));
            assert_eq!(arbi_a.wrapping_shl(shift), arbi_a.clone() << shift as u32);
            assert_eq!(arbi_a.saturating_shl(shift), arbi_a.clone() << shift as u32);
        }
        6 => {
            let arbi = arbi_a.trailing_zeros();
            let rug = rug_a.find_one(0);
            match rug {
                Some(r) => assert_eq!(arbi, r as usize),
                None => assert_eq!(arbi, 0),
            }
            let arbi_ones = arbi_a.count_ones();
            let rug_ones = rug_a.count_ones().unwrap_or(0);
            assert_eq!(arbi_ones, rug_ones as usize);
            assert_eq!(arbi_a.significant_bits(), rug_a.significant_bits() as usize);
            assert_eq!(
                arbi_a.significant_bits(),
                if arbi_a.is_zero() { 0 } else { arbi_a.to_string_radix(2).len() }
            );
        }
        7 => {
            let bit = (arbi_b.to_u64().unwrap_or(0) % 256) as usize;
            let set = arbi_a.set_bit(bit);
            assert!(set.get_bit(bit));
            let cleared = set.clear_bit(bit);
            assert!(!cleared.get_bit(bit));
            let toggled = cleared.toggle_bit(bit);
            assert!(toggled.get_bit(bit));
        }
        _ => {
            let swapped = arbi_a.swap_bytes();
            let _back = swapped.swap_bytes();
        }
    }
}
