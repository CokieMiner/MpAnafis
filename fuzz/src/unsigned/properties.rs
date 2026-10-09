//! Unsigned bit inspection, finite transformations, and ordering against GMP.

use mp_anafis::{BoundedPrecision, MpError, MpUint, Precision};
use rug::Integer;

use crate::{BitReference, Bounds, Input, assert_integer, assert_optional};

pub fn fuzz_all(a: &MpUint, b: &MpUint, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    let width = usize::from(input.parameter % 512) + 1;
    let bits = (input.flags & 1 == 0).then_some(width);
    let bounds = Bounds {
        bits,
        signed: false,
    };
    let a = bits.map_or_else(
        || a.clone(),
        |_| MpUint::with_precision_wrapping(a.clone(), BoundedPrecision::new(width).unwrap()),
    );
    let ra = bounds.wrap(ra);
    match input.operation % 4 {
        0 => {
            let ones = BitReference::ones(&ra).unwrap();
            assert_eq!(
                a.significant_bits(),
                usize::try_from(ra.significant_bits()).unwrap()
            );
            assert_eq!(a.count_ones(), ones);
            assert_eq!(a.count_zeros(), bits.map(|width| width - ones));
            assert_eq!(
                a.leading_zeros(),
                bits.map(|width| width - usize::try_from(ra.significant_bits()).unwrap())
            );
            assert_eq!(
                a.leading_ones(),
                bits.map(|width| width
                    - usize::try_from(
                        BitReference::residue(&!ra.clone(), width).significant_bits()
                    )
                    .unwrap())
            );
            assert_eq!(
                a.trailing_zeros(),
                if ra == 0 {
                    0
                } else {
                    BitReference::next(&ra, 0, true).unwrap()
                }
            );
            assert_eq!(
                a.trailing_ones(),
                BitReference::next(&ra, 0, false).unwrap()
            );
        }
        1 => {
            let bit = usize::from(input.parameter % 1024);
            let actual = ra.get_bit(u32::try_from(bit).unwrap());
            assert_eq!(a.get_bit(bit), actual);
            assert_eq!(a.test_bit(bit), actual);
            for (value, changed) in [
                (true, a.set_bit(bit)),
                (false, a.clear_bit(bit)),
                (!actual, a.toggle_bit(bit)),
                (
                    input.flags & 0x20 != 0,
                    a.set_bit_to(bit, input.flags & 0x20 != 0),
                ),
            ] {
                let mut expected = ra.clone();
                if bits.is_none_or(|width| bit < width) {
                    expected.set_bit(u32::try_from(bit).unwrap(), value);
                }
                assert_integer(&changed, &expected);
                assert_eq!(changed.precision(), a.precision());
            }
            assert_eq!(a.find_first_set_bit(), BitReference::next(&ra, 0, true));
            assert_eq!(a.find_next_set_bit(bit), BitReference::next(&ra, bit, true));
            assert_eq!(
                a.find_first_zero_bit(),
                BitReference::next(&ra, 0, false).unwrap()
            );
            assert_eq!(
                a.find_next_zero_bit(bit),
                BitReference::next(&ra, bit, false).unwrap()
            );
            let to = usize::from(input.parameter >> 8) * 4;
            let expected = if to <= bit {
                Integer::new()
            } else {
                BitReference::residue(&(ra.clone() >> bit), to - bit)
            };
            let range = a.bit_range(bit, to);
            assert_integer(&range, &expected);
            assert_eq!(range.precision(), a.precision());
        }
        2 => {
            assert_eq!(a.is_zero(), ra == 0);
            assert_eq!(a.is_one(), ra == 1);
            assert_eq!(a.is_even(), ra.is_even());
            assert_eq!(a.is_odd(), ra.is_odd());
            assert_eq!(
                a.is_power_of_two(),
                ra > 0 && (ra.clone() & (ra.clone() - 1_u32)) == 0
            );
            let next = if ra <= 1 {
                Integer::from(1)
            } else {
                Integer::from(1) << (ra.clone() - 1_u32).significant_bits()
            };
            assert_optional(
                a.checked_next_power_of_two(),
                bounds.fits(&next).then_some(next),
            );
            assert_eq!(a.cmp(b), ra.cmp(rb));
            assert_eq!(a.partial_cmp(b), Some(ra.cmp(rb)));
            assert_integer(a.clone().min(b.clone()), &ra.clone().min(rb.clone()));
            assert_integer(a.clone().max(b.clone()), &ra.clone().max(rb.clone()));
            let (low, high) = if ra <= *rb {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            };
            assert_integer(a.clone().clamp(low, high), &ra);
        }
        _ => {
            let explicit = Bounds {
                bits: Some(width),
                signed: false,
            };
            let shift = u32::from(input.parameter);
            let residue = BitReference::residue(&ra, width);
            for (result, expected) in [
                (
                    a.rotate_left(shift, width),
                    BitReference::rotate(&ra, width, usize::from(input.parameter), true),
                ),
                (
                    a.rotate_right(shift, width),
                    BitReference::rotate(&ra, width, usize::from(input.parameter), false),
                ),
                (a.reverse_bits(width), BitReference::reverse(&ra, width)),
                (a.not_with_width(width), explicit.wrap(&!residue)),
            ] {
                let result = result.unwrap();
                assert_integer(&result, &expected);
                assert_eq!(result.precision(), Precision::new_bounded(width).unwrap());
            }
            let swapped = BitReference::swap_bytes(&ra, bits);
            assert_integer(a.swap_bytes(), &bounds.wrap(&swapped));
            if bits.is_some() {
                let expected = bounds.wrap(&!ra.clone());
                assert_integer(a.try_not().unwrap(), &expected);
                assert_integer(!&a, &expected);
                assert_integer(!a.clone(), &expected);
            } else {
                assert_eq!(a.try_not(), Err(MpError::WidthRequired));
            }
            for invalid in [0, usize::MAX] {
                assert_eq!(a.rotate_left(shift, invalid), None);
                assert_eq!(a.rotate_right(shift, invalid), None);
                assert_eq!(a.reverse_bits(invalid), None);
                assert_eq!(a.not_with_width(invalid), None);
            }
        }
    }
}
