//! GMP models for bounded ranges, wrapping residues, and arithmetic failures.

use core::fmt::Display;

use mp_anafis::MpError;
use rug::Integer;

use super::{assert_integer, assert_optional};

pub struct Bounds {
    pub bits: Option<usize>,
    pub signed: bool,
}

pub struct PolicyResults<T> {
    pub checked: Option<T>,
    pub tried: Result<T, MpError>,
    pub wrapping: Option<T>,
    pub saturating: T,
    pub overflowing: (T, bool),
}

impl Bounds {
    pub fn check<T: Display>(
        &self,
        actual: PolicyResults<T>,
        exact: Option<Integer>,
        rejected: bool,
    ) {
        let fits = !rejected && exact.as_ref().is_some_and(|value| self.fits(value));
        let checked = exact.clone().filter(|_| fits);
        assert_optional(actual.checked, checked.clone());
        let error = match exact.as_ref() {
            None => MpError::DivisionByZero,
            Some(value) if !self.signed && value < &0 => MpError::Underflow,
            Some(_) => MpError::Overflow,
        };
        assert_eq!(
            actual.tried.map(|value| value.to_string()),
            checked.map(|value| value.to_string()).ok_or(error)
        );
        let zero = Integer::new();
        let value = exact.as_ref().unwrap_or(&zero);
        let wrapped = if self.bits.is_none() && !self.signed && value < &0 {
            Integer::new()
        } else {
            self.wrap(value)
        };
        if let Some(actual) = actual.wrapping {
            assert_integer(actual, &wrapped);
        }
        assert_integer(actual.saturating, &self.saturate(value));
        assert_integer(actual.overflowing.0, &wrapped);
        assert_eq!(actual.overflowing.1, !fits);
    }

    pub fn fits(&self, value: &Integer) -> bool {
        if !self.signed && value < &0 {
            return false;
        }
        let Some(bits) = self.bits else {
            return true;
        };
        if self.signed {
            let endpoint = Integer::from(1) << (bits - 1);
            value >= &-endpoint.clone() && value < &endpoint
        } else {
            value.significant_bits() <= u32::try_from(bits).expect("bounded fuzz width")
        }
    }

    pub fn wrap(&self, value: &Integer) -> Integer {
        let Some(bits) = self.bits else {
            return value.clone();
        };
        let radix = Integer::from(1) << bits;
        let residue = value.clone().modulo(&radix);
        if self.signed && residue.get_bit(u32::try_from(bits - 1).expect("bounded fuzz width")) {
            residue - radix
        } else {
            residue
        }
    }

    pub fn saturate(&self, value: &Integer) -> Integer {
        if !self.signed && value < &0 {
            return Integer::new();
        }
        if self.fits(value) {
            return value.clone();
        }
        let bits = self.bits.expect("finite upper bound");
        let endpoint = Integer::from(1) << (bits - usize::from(self.signed));
        if value < &0 {
            -endpoint
        } else {
            endpoint - 1_u8
        }
    }
}
