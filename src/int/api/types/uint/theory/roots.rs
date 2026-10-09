//! Unsigned root APIs.

use super::MpUint;

impl MpUint {
    /// Returns the integer square root $\lfloor \sqrt{\text{self}} \rfloor$.
    ///
    /// This operation is total on unsigned integers and always returns `Some`.
    #[must_use]
    pub fn isqrt(&self) -> Option<Self> {
        let result = Self {
            value: self.value.isqrt(),
            precision: self.precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns the integer square root $s = \lfloor \sqrt{\text{self}} \rfloor$
    /// and remainder $r = \text{self} - s^2$.
    ///
    /// This operation is total on unsigned integers and always returns `Some`.
    #[must_use]
    pub fn sqrt_rem(&self) -> Option<(Self, Self)> {
        let (s, r) = self.value.sqrt_rem();
        let sq = Self {
            value: s,
            precision: self.precision,
        };
        sq.debug_assert_valid();
        let rem = Self {
            value: r,
            precision: self.precision,
        };
        rem.debug_assert_valid();
        Some((sq, rem))
    }

    /// Returns the floor of the $n$-th root $\lfloor \text{self}^{1/n} \rfloor$,
    /// or `None` if $n = 0$.
    #[must_use]
    pub fn nth_root(&self, n: u32) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let result = Self {
            value: self.value.nth_root(n),
            precision: self.precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns `true` if `self` is a perfect square.
    #[must_use]
    pub fn is_perfect_square(&self) -> bool {
        self.value.is_perfect_square()
    }
}
