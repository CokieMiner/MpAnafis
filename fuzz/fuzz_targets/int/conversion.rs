//! String conversions, serialization, and primitive casts for `MpInt`.

use mp_anafis::MpInt;

pub fn fuzz_all(mp_a: &MpInt, selector: u8) {
    match selector % 7 {
        0 => {
            let s = mp_a.to_string_radix(2);
            let back = MpInt::from_str_radix(&s, 2).unwrap();
            assert_eq!(mp_a, &back);
        }
        1 => {
            let s = mp_a.to_string_radix(8);
            let back = MpInt::from_str_radix(&s, 8).unwrap();
            assert_eq!(mp_a, &back);
        }
        2 => {
            let s = mp_a.to_string_radix(10);
            let back = MpInt::from_str_radix(&s, 10).unwrap();
            assert_eq!(mp_a, &back);
        }
        3 => {
            let s = mp_a.to_string_radix(16);
            let back = MpInt::from_str_radix(&s, 16).unwrap();
            assert_eq!(mp_a, &back);
        }
        4 => {
            let le = mp_a.to_le_bytes();
            let back_le = MpInt::from_le_bytes(&le);
            assert_eq!(mp_a, &back_le);
            let be = mp_a.to_be_bytes();
            let back_be = MpInt::from_be_bytes(&be);
            assert_eq!(mp_a, &back_be);
        }
        5 => {
            let _i64 = mp_a.to_i64();
            let _i128 = mp_a.to_i128();
            let _isize = mp_a.to_isize();
            let _u64 = mp_a.to_u64();
            let _u128 = mp_a.to_u128();
            let _usize = mp_a.to_usize();
            let _f64 = mp_a.to_f64();
            let _f32 = mp_a.to_f32();
        }
        _ => {
            let bits = (mp_a.to_i64().unwrap_or(0).unsigned_abs() % 2047 + 1) as usize;
            let _max = MpInt::max_for_precision(bits);
            let _min = MpInt::min_for_precision(bits);
            let _cap = MpInt::with_capacity(bits / 64 + 1);
        }
    }
}
