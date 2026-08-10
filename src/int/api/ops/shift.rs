//! Shift operator trait implementations.

use core::ops::{Shl, ShlAssign, Shr, ShrAssign};

use super::{ArbiInt, ArbiUint};

macro_rules! impl_arbiuint_shl {
    ($shift_ty:ty, $convert:expr) => {
        impl Shl<$shift_ty> for ArbiUint {
            type Output = Self;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shl(self, shift: $shift_ty) -> Self::Output {
                let shift = $convert(shift);
                let result = Self {
                    value: self.value.shl(shift),
                    precision: self.precision,
                };
                result.assert_fits("shift");
                result.debug_assert_valid();
                result
            }
        }

        impl Shl<$shift_ty> for &ArbiUint {
            type Output = ArbiUint;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shl(self, shift: $shift_ty) -> Self::Output {
                let shift = $convert(shift);
                let result = ArbiUint {
                    value: self.value.shl(shift),
                    precision: self.precision,
                };
                result.assert_fits("shift");
                result.debug_assert_valid();
                result
            }
        }
    };
}

macro_rules! impl_arbiuint_shr {
    ($shift_ty:ty, $convert:expr) => {
        impl Shr<$shift_ty> for ArbiUint {
            type Output = Self;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shr(self, shift: $shift_ty) -> Self::Output {
                let shift = $convert(shift);
                let result = Self {
                    value: self.value.shr(shift),
                    precision: self.precision,
                };
                // Right shift only shrinks the magnitude while the precision is
                // unchanged, so the result provably fits: no fit check needed.
                result.debug_assert_valid();
                result
            }
        }

        impl Shr<$shift_ty> for &ArbiUint {
            type Output = ArbiUint;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shr(self, shift: $shift_ty) -> Self::Output {
                let shift = $convert(shift);
                let result = ArbiUint {
                    value: self.value.shr(shift),
                    precision: self.precision,
                };
                // Right shift only shrinks the magnitude while the precision is
                // unchanged, so the result provably fits: no fit check needed.
                result.debug_assert_valid();
                result
            }
        }
    };
}

macro_rules! impl_arbiint_shl {
    ($shift_ty:ty, $convert:expr) => {
        impl Shl<$shift_ty> for ArbiInt {
            type Output = Self;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shl(self, shift: $shift_ty) -> Self::Output {
                let shift = $convert(shift);
                let result = Self {
                    value: Shl::shl(self.value, shift),
                    precision: self.precision,
                };
                result.assert_fits("shift");
                result.debug_assert_valid();
                result
            }
        }

        impl Shl<$shift_ty> for &ArbiInt {
            type Output = ArbiInt;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shl(self, shift: $shift_ty) -> Self::Output {
                let shift = $convert(shift);
                let result = ArbiInt {
                    value: Shl::shl(&self.value, shift),
                    precision: self.precision,
                };
                result.assert_fits("shift");
                result.debug_assert_valid();
                result
            }
        }
    };
}

macro_rules! impl_arbiint_shr {
    ($shift_ty:ty, $convert:expr) => {
        impl Shr<$shift_ty> for ArbiInt {
            type Output = Self;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shr(self, shift: $shift_ty) -> Self::Output {
                let shift = $convert(shift);
                let result = Self {
                    value: Shr::shr(self.value, shift),
                    precision: self.precision,
                };
                // Right shift only shrinks the magnitude while the precision is
                // unchanged, so the result provably fits: no fit check needed.
                result.debug_assert_valid();
                result
            }
        }

        impl Shr<$shift_ty> for &ArbiInt {
            type Output = ArbiInt;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shr(self, shift: $shift_ty) -> Self::Output {
                let shift = $convert(shift);
                let result = ArbiInt {
                    value: Shr::shr(&self.value, shift),
                    precision: self.precision,
                };
                // Right shift only shrinks the magnitude while the precision is
                // unchanged, so the result provably fits: no fit check needed.
                result.debug_assert_valid();
                result
            }
        }
    };
}

macro_rules! impl_arbiuint_shl_assign {
    ($shift_ty:ty, $convert:expr) => {
        impl ShlAssign<$shift_ty> for ArbiUint {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shl_assign(&mut self, shift: $shift_ty) {
                let shift = $convert(shift);
                if !self.precision.is_unlimited() {
                    // Compute the candidate into a fresh value so a caught
                    // bounded-overflow panic leaves `self` unchanged.
                    let result = Self {
                        value: self.value.shl(shift),
                        precision: self.precision,
                    };
                    result.assert_fits("shift");
                    result.debug_assert_valid();
                    *self = result;
                    return;
                }
                self.value.shl_assign(shift);
                self.assert_fits("shift");
                self.debug_assert_valid();
            }
        }
    };
}

macro_rules! impl_arbiuint_shr_assign {
    ($shift_ty:ty, $convert:expr) => {
        impl ShrAssign<$shift_ty> for ArbiUint {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shr_assign(&mut self, shift: $shift_ty) {
                let shift = $convert(shift);
                // Right shift only shrinks the magnitude while the receiver
                // precision is unchanged, so the result provably fits.
                self.value.shr_assign(shift);
                self.debug_assert_valid();
            }
        }
    };
}

macro_rules! impl_arbiint_shl_assign {
    ($shift_ty:ty, $convert:expr) => {
        impl ShlAssign<$shift_ty> for ArbiInt {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shl_assign(&mut self, shift: $shift_ty) {
                let shift = $convert(shift);
                if !self.precision.is_unlimited() {
                    // Compute the candidate into a fresh value so a caught
                    // bounded-overflow panic leaves `self` unchanged.
                    let result = Self {
                        value: Shl::shl(&self.value, shift),
                        precision: self.precision,
                    };
                    result.assert_fits("shift");
                    result.debug_assert_valid();
                    *self = result;
                    return;
                }
                self.value.shl_assign(shift);
                self.assert_fits("shift");
                self.debug_assert_valid();
            }
        }
    };
}

macro_rules! impl_arbiint_shr_assign {
    ($shift_ty:ty, $convert:expr) => {
        impl ShrAssign<$shift_ty> for ArbiInt {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "shift trait impls normalize counts to usize; checked conversions panic instead of truncating on narrow targets"
            )]
            #[inline]
            #[track_caller]
            fn shr_assign(&mut self, shift: $shift_ty) {
                let shift = $convert(shift);
                // Right shift only shrinks the magnitude while the receiver
                // precision is unchanged, so the result provably fits.
                self.value.shr_assign(shift);
                self.debug_assert_valid();
            }
        }
    };
}

macro_rules! impl_shift_all {
    ($macro_name:ident) => {
        $macro_name!(u8, |x: u8| usize::from(x));
        $macro_name!(u16, |x: u16| usize::from(x));
        $macro_name!(u32, |x: u32| usize::try_from(x)
            .expect("shift count overflow"));
        $macro_name!(u64, |x: u64| usize::try_from(x)
            .expect("shift count overflow"));
        $macro_name!(u128, |x: u128| usize::try_from(x)
            .expect("shift count overflow"));
        $macro_name!(usize, |x: usize| x);
        $macro_name!(i8, |x: i8| usize::try_from(x)
            .expect("negative shift count"));
        $macro_name!(i16, |x: i16| usize::try_from(x)
            .expect("negative shift count"));
        $macro_name!(i32, |x: i32| usize::try_from(x)
            .expect("negative shift count"));
        $macro_name!(i64, |x: i64| usize::try_from(x)
            .expect("negative shift count"));
        $macro_name!(i128, |x: i128| usize::try_from(x)
            .expect("negative shift count"));
        $macro_name!(isize, |x: isize| usize::try_from(x)
            .expect("negative shift count"));
    };
}

impl_shift_all!(impl_arbiuint_shl);
impl_shift_all!(impl_arbiuint_shr);
impl_shift_all!(impl_arbiuint_shl_assign);
impl_shift_all!(impl_arbiuint_shr_assign);
impl_shift_all!(impl_arbiint_shl);
impl_shift_all!(impl_arbiint_shr);
impl_shift_all!(impl_arbiint_shl_assign);
impl_shift_all!(impl_arbiint_shr_assign);
