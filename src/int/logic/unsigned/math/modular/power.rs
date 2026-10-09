//! Modular powers by sliding-window exponentiation.

#![expect(
    unsafe_code,
    reason = "Positive exponent widths and windows in 1..=6 bound exact decrements, initialized exponent reads, and odd-power table indices."
)]

use core::{
    array::from_fn,
    cmp::Ordering,
    mem::{MaybeUninit, swap},
};

use super::{
    BarrettDomain, BarrettScratch, DivScratch, Division, Exponentiation, InlineMontgomery,
    InternalMpUint, LIMB_BITS, Limb, LimbMontgomery, MontgomeryDomain, MontgomeryScratch,
    MulScratch,
};

impl MontgomeryDomain {
    /// Computes `base^exp` in this Montgomery domain.
    ///
    /// Construction must request `ENCODE=true`. When `raw` is true, the
    /// result remains in Montgomery form.
    pub fn pow(
        &self,
        base: &InternalMpUint,
        exp: &InternalMpUint,
        scratch: &mut MontgomeryScratch,
        raw: bool,
    ) -> InternalMpUint {
        let domain = self;
        let mut temp_prod = InternalMpUint::zero();

        let bits = exp.significant_bits();
        if bits == 0 {
            if domain.modulus.is_one() {
                return InternalMpUint::zero();
            }
            return if raw {
                domain.transform_into_with_scratch(&InternalMpUint::one(), &mut temp_prod, scratch)
            } else {
                InternalMpUint::one()
            };
        }

        match domain.modulus.limbs().len() {
            2 => return InlineMontgomery::<2>::pow(domain, base, exp, raw),
            3 => return InlineMontgomery::<3>::pow(domain, base, exp, raw),
            4 => return InlineMontgomery::<4>::pow(domain, base, exp, raw),
            _ => {}
        }

        let plan = Exponentiation::window_plan::<false>(exp, bits);
        let exp_limbs = exp.limbs();
        let window = plan.width;
        let table_size = plan.powers;

        let mut g: [InternalMpUint; Exponentiation::ODD_POWER_CAPACITY] =
            from_fn(|_| InternalMpUint::zero());
        // SAFETY: the table has 32 elements, so index zero is in bounds.
        *unsafe { g.get_unchecked_mut(0) } =
            domain.transform_into_with_scratch(base, &mut temp_prod, scratch);

        if table_size > 1 {
            let mut base2 = InternalMpUint::zero();
            // SAFETY: the table has 32 elements, so index zero is in bounds.
            domain.square_into_with_scratch(
                unsafe { g.get_unchecked(0) },
                &mut base2,
                &mut temp_prod,
                scratch,
            );

            for i in 1..table_size {
                // SAFETY: the window plan bounds table_size by g.len()=32;
                // 1<=i<table_size leaves disjoint prior powers and output slots.
                let (left, right) = unsafe { g.split_at_mut_unchecked(i) };
                // SAFETY: `left.len() = i > 0`, so `i - 1` is in bounds.
                let left_factor = unsafe { left.get_unchecked(i.unchecked_sub(1)) };
                // SAFETY: `i < table_size <= 32 = g.len()`, so the split
                // leaves at least one element in `right`.
                let destination = unsafe { right.get_unchecked_mut(0) };
                domain.mul_into_with_scratch(
                    left_factor,
                    &base2,
                    destination,
                    &mut temp_prod,
                    scratch,
                );
            }
        }

        let (initial, initial_len) = plan.initial;
        // SAFETY: planning includes the leading index in the initialized prefix.
        let mut result = unsafe { g.get_unchecked(initial) }.clone();
        let mut next_res = InternalMpUint::zero();
        // SAFETY: the first window consumes 1..=min(window,bits) bits.
        let mut bit_pos = unsafe { bits.unchecked_sub(initial_len) };
        while bit_pos > 0 {
            // SAFETY: 0<bit_pos<=bits bounds the limb containing bit_pos-1
            // within the exponent's initialized canonical magnitude.
            let bit = unsafe {
                let index = bit_pos.unchecked_sub(1);
                (*exp_limbs.get_unchecked(index >> LIMB_BITS.trailing_zeros())
                    >> (index & (LIMB_BITS - 1)))
                    & 1
            };

            if bit == 0 {
                domain.square_into_with_scratch(&result, &mut next_res, &mut temp_prod, scratch);
                swap(&mut result, &mut next_res);
                // SAFETY: the loop condition proves bit_pos>0.
                bit_pos = unsafe { bit_pos.unchecked_sub(1) };
            } else {
                let (table_index, consumed) = Exponentiation::window(exp_limbs, bit_pos, window);
                for _ in 0..consumed {
                    domain.square_into_with_scratch(
                        &result,
                        &mut next_res,
                        &mut temp_prod,
                        scratch,
                    );
                    swap(&mut result, &mut next_res);
                }
                domain.mul_into_with_scratch(
                    &result,
                    // SAFETY: planning scans these same odd windows and includes
                    // every referenced index in the initialized table prefix.
                    unsafe { g.get_unchecked(table_index) },
                    &mut next_res,
                    &mut temp_prod,
                    scratch,
                );
                swap(&mut result, &mut next_res);
                // SAFETY: 1<=consumed<=bit_pos follows from window extraction.
                bit_pos = unsafe { bit_pos.unchecked_sub(consumed) };
            }
        }

        if raw {
            result
        } else {
            // Decoding is REDC(result); consume the encoded value and reuse
            // the existing output buffer without multiplying by one.
            domain.reduce_into(&mut result, &mut next_res, scratch);
            next_res
        }
    }
}

impl BarrettDomain {
    /// Computes `base^exp` in this Barrett domain.
    pub fn pow(
        &self,
        base: &InternalMpUint,
        exp: &InternalMpUint,
        mul_scratch: &mut MulScratch,
    ) -> InternalMpUint {
        let domain = self;
        let mut temp_prod = InternalMpUint::zero();
        let mut barrett_scratch = BarrettScratch::default();

        let bits = exp.significant_bits();
        if bits == 0 {
            return if domain.modulus.is_one() {
                InternalMpUint::zero()
            } else {
                InternalMpUint::one()
            };
        }

        let plan = Exponentiation::window_plan::<false>(exp, bits);
        let exp_limbs = exp.limbs();
        let window = plan.width;
        let table_size = plan.powers;

        let mut g: [InternalMpUint; Exponentiation::ODD_POWER_CAPACITY] =
            from_fn(|_| InternalMpUint::zero());

        // A base larger than `b^{2k}` requires standard division to reduce it initially.
        let mut reduced_base = InternalMpUint::zero();
        if base.cmp(&domain.modulus) == Ordering::Less {
            reduced_base.clone_from(base);
        } else {
            let mut scratch = DivScratch::default();
            Division::rem_into(base, &domain.modulus, &mut reduced_base, &mut scratch);
        }
        // SAFETY: the table has 32 elements, so index zero is in bounds.
        *unsafe { g.get_unchecked_mut(0) } = reduced_base;

        if table_size > 1 {
            let mut base2 = InternalMpUint::zero();
            temp_prod.assign_square_with_scratch(
                // SAFETY: the table has 32 elements, so index zero is in bounds.
                unsafe { g.get_unchecked(0) },
                mul_scratch,
            );
            domain.reduce_into_with_barrett_scratch(
                &temp_prod,
                &mut base2,
                mul_scratch,
                &mut barrett_scratch,
            );

            for i in 1..table_size {
                // SAFETY: the window plan bounds table_size by g.len()=32;
                // 1<=i<table_size leaves disjoint prior powers and output slots.
                let (left, right) = unsafe { g.split_at_mut_unchecked(i) };
                // SAFETY: `left.len() = i > 0`, so `i - 1` is in bounds.
                let left_factor = unsafe { left.get_unchecked(i.unchecked_sub(1)) };
                // SAFETY: `i < table_size <= 32 = g.len()`, so the split
                // leaves at least one element in `right`.
                let destination = unsafe { right.get_unchecked_mut(0) };
                temp_prod.assign_product_with_scratch(left_factor, &base2, mul_scratch);
                domain.reduce_into_with_barrett_scratch(
                    &temp_prod,
                    destination,
                    mul_scratch,
                    &mut barrett_scratch,
                );
            }
        }

        let (initial, initial_len) = plan.initial;
        // SAFETY: planning includes the leading index in the initialized prefix.
        let mut result = unsafe { g.get_unchecked(initial) }.clone();
        let mut next_res = InternalMpUint::zero();
        // SAFETY: the first window consumes 1..=min(window,bits) bits.
        let mut bit_pos = unsafe { bits.unchecked_sub(initial_len) };
        while bit_pos > 0 {
            // SAFETY: 0<bit_pos<=bits bounds the limb containing bit_pos-1
            // within the exponent's initialized canonical magnitude.
            let bit = unsafe {
                let index = bit_pos.unchecked_sub(1);
                (*exp_limbs.get_unchecked(index >> LIMB_BITS.trailing_zeros())
                    >> (index & (LIMB_BITS - 1)))
                    & 1
            };

            if bit == 0 {
                temp_prod.assign_square_with_scratch(&result, mul_scratch);
                domain.reduce_into_with_barrett_scratch(
                    &temp_prod,
                    &mut next_res,
                    mul_scratch,
                    &mut barrett_scratch,
                );
                swap(&mut result, &mut next_res);
                // SAFETY: the loop condition proves bit_pos>0.
                bit_pos = unsafe { bit_pos.unchecked_sub(1) };
            } else {
                let (table_index, consumed) = Exponentiation::window(exp_limbs, bit_pos, window);
                for _ in 0..consumed {
                    temp_prod.assign_square_with_scratch(&result, mul_scratch);
                    domain.reduce_into_with_barrett_scratch(
                        &temp_prod,
                        &mut next_res,
                        mul_scratch,
                        &mut barrett_scratch,
                    );
                    swap(&mut result, &mut next_res);
                }
                temp_prod.assign_product_with_scratch(
                    &result,
                    // SAFETY: planning scans these same odd windows and includes
                    // every referenced index in the initialized table prefix.
                    unsafe { g.get_unchecked(table_index) },
                    mul_scratch,
                );
                domain.reduce_into_with_barrett_scratch(
                    &temp_prod,
                    &mut next_res,
                    mul_scratch,
                    &mut barrett_scratch,
                );
                swap(&mut result, &mut next_res);
                // SAFETY: 1<=consumed<=bit_pos follows from window extraction.
                bit_pos = unsafe { bit_pos.unchecked_sub(consumed) };
            }
        }

        result
    }
}

impl LimbMontgomery {
    /// Computes `base^exp mod modulus` for a positive exponent.
    ///
    /// Odd powers and the accumulator occupy native limbs; every recurrence
    /// multiplies canonical residues, so its product is below `modulus*B`.
    pub fn pow(&self, base: &InternalMpUint, exp: &InternalMpUint) -> InternalMpUint {
        debug_assert!(
            !exp.is_zero(),
            "the scalar power requires a positive exponent"
        );
        let reduced = if let [value] = base.limbs() {
            // A full limb times radix_square<modulus already meets REDC's bound.
            *value
        } else {
            Division::div_rem_1::<false>(base.limbs(), self.modulus, &mut InternalMpUint::zero())
        };
        let bits = exp.significant_bits();
        let plan = Exponentiation::window_plan::<true>(exp, bits);
        let exp_limbs = exp.limbs();
        let window = plan.width;
        let table_size = plan.powers;
        let encoded = self.multiply(reduced, self.radix_square);
        let mut powers = [MaybeUninit::<Limb>::uninit(); Exponentiation::ODD_POWER_CAPACITY];
        // SAFETY: the fixed array contains index zero; this initializes its first power.
        unsafe {
            let _ = powers.get_unchecked_mut(0).write(encoded);
        }
        if table_size > 1 {
            let square = self.multiply(encoded, encoded);
            let mut previous = encoded;
            for index in 1..table_size {
                previous = self.multiply(previous, square);
                // SAFETY: index<table_size<=32; each loop initializes the next odd power.
                unsafe {
                    let _ = powers.get_unchecked_mut(index).write(previous);
                }
            }
        }
        let (initial, consumed) = plan.initial;
        // SAFETY: the leading odd window indexes the initialized table_size prefix.
        let mut value = unsafe { powers.get_unchecked(initial).assume_init() };
        // SAFETY: the leading window consumes 1..=bits initialized exponent bits.
        let mut remaining = unsafe { bits.unchecked_sub(consumed) };
        while remaining != 0 {
            // SAFETY: remaining>0 and remaining<=bits bound the initialized exponent bit.
            let index = unsafe { remaining.unchecked_sub(1) };
            // SAFETY: the exponent bit index is below its significant width.
            let bit = unsafe { *exp_limbs.get_unchecked(index >> LIMB_BITS.trailing_zeros()) }
                >> (index & (LIMB_BITS - 1))
                & 1;
            if bit == 0 {
                value = self.multiply(value, value);
                remaining = index;
            } else {
                let (slot, length) = Exponentiation::window(exp_limbs, remaining, window);
                for _ in 0..length {
                    value = self.multiply(value, value);
                }
                // SAFETY: planning bounds every odd digit using the exponent's
                // minimum set-bit spacing; that prefix is fully initialized.
                value = self.multiply(value, unsafe { powers.get_unchecked(slot).assume_init() });
                // SAFETY: 1<=length<=remaining follows from window extraction.
                remaining = unsafe { remaining.unchecked_sub(length) };
            }
        }
        InternalMpUint::from_limb(self.multiply(value, 1))
    }
}
