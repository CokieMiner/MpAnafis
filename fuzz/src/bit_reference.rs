//! Finite bit transformations expressed as GMP integer operations.

use rug::{Integer, integer::Order};

pub struct BitReference;

impl BitReference {
    pub fn residue(value: &Integer, width: usize) -> Integer {
        value.clone().modulo(&(Integer::from(1) << width))
    }

    pub fn rotate(value: &Integer, width: usize, shift: usize, left: bool) -> Integer {
        let value = Self::residue(value, width);
        let shift = shift % width;
        let shift = if left { shift } else { (width - shift) % width };
        Self::residue(
            &((value.clone() << shift) | (value >> (width - shift))),
            width,
        )
    }

    pub fn reverse(value: &Integer, width: usize) -> Integer {
        let value = Self::residue(value, width);
        let mut result = Integer::new();
        for bit in 0..width {
            result.set_bit(
                u32::try_from(width - bit - 1).unwrap(),
                value.get_bit(u32::try_from(bit).unwrap()),
            );
        }
        result
    }

    pub fn swap_bytes(value: &Integer, width: Option<usize>) -> Integer {
        let mut bytes = value.to_digits::<u8>(Order::Lsf);
        if let Some(width) = width {
            bytes.resize(width.div_ceil(8), 0);
        }
        Integer::from_digits(&bytes, Order::Msf)
    }

    pub fn next(value: &Integer, from: usize, set: bool) -> Option<usize> {
        let from = u32::try_from(from).unwrap();
        let bit = if set {
            value.find_one(from)
        } else {
            value.find_zero(from)
        };
        bit.map(|index| usize::try_from(index).unwrap())
    }

    pub fn ones(value: &Integer) -> Option<usize> {
        value
            .count_ones()
            .map(|count| usize::try_from(count).unwrap())
    }
}
