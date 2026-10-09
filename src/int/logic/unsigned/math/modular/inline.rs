//! Montgomery exponentiation at fixed inline limb widths.
//!
//! Residues and odd powers retain their padded width throughout the ladder.
//! Product writers initialize exactly 2N limbs. Reduction cancels one low
//! limb per row and defers each high carry to the following row.

#![expect(
    unsafe_code,
    reason = "the fixed-width dispatch proves initialized product spans, table indices, and bounded carry recurrences"
)]

use core::{array::from_fn, mem::MaybeUninit};

use super::{
    ArchKernels, DoubleLimb, Exponentiation, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb,
    MontgomeryDomain, Schoolbook,
};

/// Montgomery constants for a compile-time width within inline storage.
#[derive(Debug)]
pub struct InlineMontgomery<const N: usize> {
    modulus: [Limb; N],
    inverse: Limb,
}

impl<const N: usize> InlineMontgomery<N> {
    /// Computes a positive-exponent power with `2<=N<=INLINE_LIMBS`.
    /// The caller supplies an odd N-limb modulus. `raw` retains Montgomery form.
    pub fn pow(
        domain: &MontgomeryDomain,
        base: &InternalMpUint,
        exponent: &InternalMpUint,
        raw: bool,
    ) -> InternalMpUint {
        debug_assert!((2..=INLINE_LIMBS).contains(&N), "inline Montgomery width");
        debug_assert_eq!(domain.modulus.limbs().len(), N, "fixed modulus width");
        debug_assert!(!exponent.is_zero(), "positive exponent");
        let arithmetic = Self {
            // SAFETY: dispatch proves the modulus contains exactly N limbs.
            modulus: from_fn(|i| unsafe { *domain.modulus.limbs().get_unchecked(i) }),
            inverse: domain.m_inv,
        };
        let reduced;
        let source = if base.limbs().len() > N {
            reduced = base.rem(&domain.modulus);
            reduced.limbs()
        } else {
            base.limbs()
        };
        let mut input = [0; N];
        // SAFETY: source.len()<=N, and the fresh initialized array is disjoint.
        unsafe {
            input
                .get_unchecked_mut(..source.len())
                .copy_from_slice(source);
        }
        let mut radix_square = [0; N];
        // SAFETY: r2<M<R, so its initialized magnitude occupies at most N limbs.
        unsafe {
            radix_square
                .get_unchecked_mut(..domain.r2.limbs().len())
                .copy_from_slice(domain.r2.limbs());
        }
        let encoded = arithmetic.product::<false>(&input, &radix_square);
        let bits = exponent.significant_bits();
        let plan = Exponentiation::window_plan::<true>(exponent, bits);
        let exp_limbs = exponent.limbs();
        let window = plan.width;
        let count = plan.powers;
        let mut powers = [MaybeUninit::<[Limb; N]>::uninit(); Exponentiation::ODD_POWER_CAPACITY];
        // SAFETY: the array includes slot zero, initialized by this write.
        unsafe {
            let _ = powers.get_unchecked_mut(0).write(encoded);
        }
        if count > 1 {
            let square = arithmetic.product::<true>(&encoded, &encoded);
            let mut previous = encoded;
            for i in 1..count {
                previous = arithmetic.product::<false>(&previous, &square);
                // SAFETY: i<count<=32 initializes the next table slot.
                unsafe {
                    let _ = powers.get_unchecked_mut(i).write(previous);
                }
            }
        }
        let (initial, consumed) = plan.initial;
        // SAFETY: the leading odd window addresses the initialized table prefix.
        let mut value = unsafe { powers.get_unchecked(initial).assume_init() };
        // SAFETY: the leading window consumes between one and bits bits.
        let mut remaining = unsafe { bits.unchecked_sub(consumed) };
        while remaining != 0 {
            // SAFETY: remaining>0 and remaining<=bits bound the exponent bit.
            let index = unsafe { remaining.unchecked_sub(1) };
            // SAFETY: index<bits addresses an initialized exponent limb.
            let bit = unsafe { *exp_limbs.get_unchecked(index >> LIMB_BITS.trailing_zeros()) }
                >> (index & (LIMB_BITS - 1))
                & 1;
            if bit == 0 {
                value = arithmetic.product::<true>(&value, &value);
                remaining = index;
            } else {
                let (slot, length) = Exponentiation::window(exp_limbs, remaining, window);
                for _ in 0..length {
                    value = arithmetic.product::<true>(&value, &value);
                }
                // SAFETY: planning bounds every odd digit using the exponent's
                // minimum set-bit spacing; that prefix is fully initialized.
                value = arithmetic.product::<false>(&value, unsafe {
                    powers.get_unchecked(slot).assume_init_ref()
                });
                // SAFETY: the window consumes 1..=remaining bits.
                remaining = unsafe { remaining.unchecked_sub(length) };
            }
        }
        if !raw {
            // For n>1, one has a single low limb and is below the modulus.
            // Decode with the same product and reduction recurrence.
            let mut one = [0; N];
            // SAFETY: N>=2, so index zero exists.
            unsafe {
                *one.get_unchecked_mut(0) = 1;
            }
            value = arithmetic.product::<false>(&value, &one);
        }
        InternalMpUint::from_limbs_slice(&value)
    }

    /// Returns a*b/R modulo M, or a^2/R when SQUARE is true.
    /// The input bound a*b<M*R gives an unreduced result below 2M.
    #[inline]
    fn product<const SQUARE: bool>(&self, a: &[Limb; N], b: &[Limb; N]) -> [Limb; N] {
        let mut product = [MaybeUninit::<Limb>::uninit(); 2 * INLINE_LIMBS];
        // SAFETY: N<=INLINE_LIMBS=4 bounds its double by eight.
        let double_width = unsafe { N.unchecked_mul(2) };
        // SAFETY: 2<=N<=INLINE_LIMBS bounds the full product destination.
        let digits = unsafe { product.get_unchecked_mut(..double_width) };
        if SQUARE {
            Schoolbook::sqr_nonempty(digits, a);
        } else {
            Schoolbook::mul_fixed_equal_distinct::<N>(digits, a, b);
        }
        let mut pending = 0;
        for row in 0..N {
            // SAFETY: each product writer filled 2N limbs, and row<N.
            let low = unsafe { product.get_unchecked(row).assume_init() };
            let correction = low.wrapping_mul(self.inverse);
            // The low limb cancels exactly. Its sum carries iff low!=0;
            // neither adding nor storing that zero is necessary.
            // SAFETY: N>=2 bounds modulus[0]; its immutable array is initialized.
            let (_, high) =
                ArchKernels::mul_limb_lo_hi(correction, unsafe { *self.modulus.get_unchecked(0) });
            // SAFETY: a limb product has high<=B-2; adding a bit fits a limb.
            let mut carry = unsafe { high.unchecked_add(Limb::from(low != 0)) };
            for column in 1..N {
                // SAFETY: row<N and column<N give row+column<2N; both
                // the product digit and the modulus limb are initialized.
                let (destination, modulus) = unsafe {
                    (
                        product.get_unchecked_mut(row.unchecked_add(column)),
                        *self.modulus.get_unchecked(column),
                    )
                };
                let (lo, hi) = ArchKernels::mul_limb_lo_hi(correction, modulus);
                // SAFETY: correction*modulus+old+carry <= (B-1)^2+2(B-1)
                // = B^2-1 fits DoubleLimb. The preceding writer initializes
                // old; every conversion widens a native limb exactly.
                let wide = unsafe {
                    (DoubleLimb::try_from(lo).unwrap_unchecked()
                        | (DoubleLimb::try_from(hi).unwrap_unchecked() << Limb::BITS))
                        .unchecked_add(
                            DoubleLimb::try_from(destination.assume_init()).unwrap_unchecked(),
                        )
                        .unchecked_add(DoubleLimb::try_from(carry).unwrap_unchecked())
                };
                // SAFETY: masking and shifting each select one native limb.
                unsafe {
                    let _ = destination.write(
                        Limb::try_from(wide & DoubleLimb::try_from(Limb::MAX).unwrap_unchecked())
                            .unwrap_unchecked(),
                    );
                    carry = Limb::try_from(wide >> Limb::BITS).unwrap_unchecked();
                }
            }
            // SAFETY: row<N and N<=INLINE_LIMBS prove row+N<2N.
            let destination = unsafe { product.get_unchecked_mut(row.unchecked_add(N)) };
            // SAFETY: old+carry+pending<=2B-1 fits DoubleLimb. The full
            // product writer initializes old, and all native limbs widen.
            let wide = unsafe {
                DoubleLimb::try_from(destination.assume_init())
                    .unwrap_unchecked()
                    .unchecked_add(DoubleLimb::try_from(carry).unwrap_unchecked())
                    .unchecked_add(DoubleLimb::try_from(pending).unwrap_unchecked())
            };
            // SAFETY: the low part fits a limb; wide<2B bounds pending by one.
            unsafe {
                let _ = destination.write(
                    Limb::try_from(wide & DoubleLimb::try_from(Limb::MAX).unwrap_unchecked())
                        .unwrap_unchecked(),
                );
                pending = Limb::try_from(wide >> Limb::BITS).unwrap_unchecked();
            }
        }
        // SAFETY: N+i<2N for i<N addresses the initialized upper half.
        let value: [Limb; N] =
            from_fn(|i| unsafe { product.get_unchecked(N.unchecked_add(i)).assume_init() });
        let mut borrow = false;
        let difference = from_fn(|i| {
            // SAFETY: from_fn passes i<N, bounding both initialized arrays.
            let (digit, modulus) =
                unsafe { (*value.get_unchecked(i), *self.modulus.get_unchecked(i)) };
            let (reduced, next) = digit.borrowing_sub(modulus, borrow);
            borrow = next;
            reduced
        });
        // A candidate with pending=1 is >=R>M. Since it is below 2M,
        // its low part must borrow when subtracting M. Thus subtraction is
        // admissible exactly when its borrow consumes the pending high bit.
        if pending == Limb::from(borrow) {
            difference
        } else {
            value
        }
    }
}
