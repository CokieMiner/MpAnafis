//! String conversions, serialization, and primitive casts for `ArbiUint`.

use arbi_anafis::ArbiUint;

pub fn fuzz_all(arbi_a: &ArbiUint, selector: u8) {
    match selector % 7 {
        0 => {
            let s = arbi_a.to_string_radix(2);
            let back = ArbiUint::from_str_radix(&s, 2).unwrap();
            assert_eq!(arbi_a, &back);
        }
        1 => {
            let s = arbi_a.to_string_radix(8);
            let back = ArbiUint::from_str_radix(&s, 8).unwrap();
            assert_eq!(arbi_a, &back);
        }
        2 => {
            let s = arbi_a.to_string_radix(10);
            let back = ArbiUint::from_str_radix(&s, 10).unwrap();
            assert_eq!(arbi_a, &back);
        }
        3 => {
            let s = arbi_a.to_string_radix(16);
            let back = ArbiUint::from_str_radix(&s, 16).unwrap();
            assert_eq!(arbi_a, &back);
        }
        4 => {
            let le = arbi_a.to_le_bytes();
            let back_le = ArbiUint::from_le_bytes(&le);
            assert_eq!(arbi_a, &back_le);
            let be = arbi_a.to_be_bytes();
            let back_be = ArbiUint::from_be_bytes(&be);
            assert_eq!(arbi_a, &back_be);
        }
        5 => {
            let _u64 = arbi_a.to_u64();
            let _u128 = arbi_a.to_u128();
            let _usize = arbi_a.to_usize();
            let _i64 = arbi_a.to_i64();
            let _i128 = arbi_a.to_i128();
            let _isize = arbi_a.to_isize();
            let _f64 = arbi_a.to_f64();
            let _f32 = arbi_a.to_f32();
        }
        _ => {
            let bits = (arbi_a.to_u64().unwrap_or(0) % 2047 + 1) as usize;
            let _max = ArbiUint::max_for_precision(bits);
            let _min = ArbiUint::min_for_precision(bits);
            let _cap = ArbiUint::with_capacity(bits / 64 + 1);
        }
    }
}
