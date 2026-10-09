//! Signed root APIs.

use super::{InternalMpInt, MpInt};

impl MpInt {
    /// Returns the integer square root $\lfloor \sqrt{\text{self}} \rfloor$, or `None` if `self` is negative.
    #[must_use]
    pub fn checked_isqrt(&self) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        let result = Self {
            value: InternalMpInt {
                abs: self.value.abs.isqrt(),
                is_positive: true,
            },
            precision: self.precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns the integer square root $s = \lfloor \sqrt{|\text{self}|} \rfloor$
    /// and remainder $r = |\text{self}| - s^2$ of the absolute value.
    ///
    /// Returns `None` if a positive result does not fit the signed precision.
    #[must_use]
    pub fn sqrt_rem(&self) -> Option<(Self, Self)> {
        // A one-bit signed width admits only -1 and 0; sqrt(|-1|) = 1 does
        // not fit. For w >= 2, even |MIN_w| = A = 2^(w-1) has sqrt(A) < A
        // and remainder A-sqrt(A)^2 < A. All other magnitudes are below A.
        if self.precision.significant_bits() == Some(1) && self.is_negative() {
            return None;
        }
        let (s, r) = self.value.abs.sqrt_rem();
        let sq = Self {
            value: InternalMpInt {
                abs: s,
                is_positive: true,
            },
            precision: self.precision,
        };
        let rem = Self {
            value: InternalMpInt {
                abs: r,
                is_positive: true,
            },
            precision: self.precision,
        };
        sq.debug_assert_valid();
        rem.debug_assert_valid();
        Some((sq, rem))
    }

    /// Returns $\lfloor |\text{self}|^{1/n} \rfloor$,
    /// or `None` if $n = 0$ or the positive result exceeds signed precision.
    #[must_use]
    pub fn nth_root(&self, n: u32) -> Option<Self> {
        if n == 0 {
            return None;
        }
        if let Some(bits) = self.precision.significant_bits()
            && (n == 1 || bits == 1)
            && self.value.is_signed_min_for_width(bits)
        {
            return None;
        }
        // Degree one is absolute value, whose sole signed overflow is MIN_w.
        // For n >= 2 and w >= 2, root(|MIN_w|) < |MIN_w|; every other
        // magnitude already lies in the positive representable interval.
        let result = Self {
            value: InternalMpInt {
                abs: self.value.abs.nth_root(n),
                is_positive: true,
            },
            precision: self.precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns `true` if the absolute value is a perfect square.
    #[must_use]
    pub fn is_perfect_square(&self) -> bool {
        self.value.abs.is_perfect_square()
    }
}
