//! Signed two's-complement inspection and ordering against GMP.

use mp_anafis::{BoundedPrecision, MpError, MpInt, Precision};
use rug::Integer;

use crate::{BitReference, Bounds, Input, assert_integer, assert_optional};

pub fn fuzz_all(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    let width = usize::from(input.parameter % 512) + 1;
    let bits = (input.flags & 1 == 0).then_some(width);
    let bounds = Bounds { bits, signed: true };
    let a = bits.map_or_else(
        || a.clone(),
        |_| MpInt::with_precision_wrapping(a.clone(), BoundedPrecision::new(width).unwrap()),
    );
    let ra = bounds.wrap(ra);
    match input.operation % 4 {
        0 => {
            let residue =
                bits.map_or_else(|| ra.clone(), |width| BitReference::residue(&ra, width));
            let ones = BitReference::ones(&residue);
            assert_eq!(
                a.significant_bits(),
                usize::try_from(ra.significant_bits()).unwrap()
            );
            assert_eq!(a.count_ones(), ones);
            assert_eq!(a.count_zeros(), bits.map(|width| width - ones.unwrap()));
            assert_eq!(
                a.leading_zeros(),
                bits.map(|width| width - usize::try_from(residue.significant_bits()).unwrap())
            );
            assert_eq!(
                a.leading_ones(),
                bits.map(|width| width
                    - usize::try_from(
                        BitReference::residue(&!residue.clone(), width).significant_bits()
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
            let trailing = if bits.is_none() && ra < 0 {
                None
            } else {
                BitReference::next(&residue, 0, false)
            };
            assert_eq!(a.trailing_ones(), trailing);
        }
        1 => {
            let bit = usize::from(input.parameter % 1024);
            let actual =
                bits.is_none_or(|width| bit < width) && ra.get_bit(u32::try_from(bit).unwrap());
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
                assert_integer(&changed, &bounds.wrap(&expected));
                assert_eq!(changed.precision(), a.precision());
            }
            let set = |from| {
                BitReference::next(&ra, from, true)
                    .filter(|&bit| bits.is_none_or(|width| bit < width))
            };
            let zero = |from| {
                BitReference::next(&ra, from, false)
                    .unwrap_or(usize::MAX)
                    .min(bits.unwrap_or(usize::MAX))
            };
            assert_eq!(a.find_first_set_bit(), set(0));
            assert_eq!(a.find_next_set_bit(bit), set(bit));
            assert_eq!(
                a.find_first_zero_bit(),
                (zero(0) < bits.unwrap_or(usize::MAX)).then(|| zero(0))
            );
            assert_eq!(a.find_next_zero_bit(bit), zero(bit));
            let to = usize::from(input.parameter >> 8) * 4;
            let expected = if to <= bit {
                Integer::new()
            } else {
                BitReference::residue(&(ra.clone() >> bit), to - bit)
            };
            let range = a.bit_range(bit, to);
            assert_integer(&range, &expected);
            let needed = usize::try_from(expected.significant_bits()).unwrap();
            let precision = if bits.is_some_and(|width| needed >= width) {
                Precision::new_bounded(needed + 1).unwrap()
            } else {
                a.precision()
            };
            assert_eq!(range.precision(), precision);
        }
        2 => {
            assert_eq!(a.is_zero(), ra == 0);
            assert_eq!(a.is_one(), ra == 1);
            assert_eq!(a.is_minus_one(), ra == -1);
            assert_eq!(a.is_positive(), ra > 0);
            assert_eq!(a.is_negative(), ra < 0);
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
                (ra >= 0 && bounds.fits(&next)).then_some(next),
            );
            let sign = Integer::from(i32::from(ra > 0) - i32::from(ra < 0));
            if bounds.fits(&sign) {
                assert_integer(a.signum(), &sign);
            }
            assert_integer(a.abs_diff(b), &(Integer::from(&ra - rb)).abs());
            let positive_difference = Integer::from(&ra - rb).max(Integer::new());
            assert_integer(a.abs_sub(b), &positive_difference);
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
                signed: true,
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
                (a.not_with_width(width), !residue),
            ] {
                let result = result.unwrap();
                assert_integer(&result, &explicit.wrap(&expected));
                assert_eq!(result.precision(), Precision::new_bounded(width).unwrap());
            }
            assert_optional(
                a.swap_bytes(),
                bits.map(|width| {
                    bounds.wrap(&BitReference::swap_bytes(
                        &BitReference::residue(&ra, width),
                        bits,
                    ))
                }),
            );
            let expected = bounds.wrap(&!ra.clone());
            assert_integer(!&a, &expected);
            assert_integer(!a.clone(), &expected);
            if bits.is_some() {
                assert_integer(a.try_not().unwrap(), &expected);
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
