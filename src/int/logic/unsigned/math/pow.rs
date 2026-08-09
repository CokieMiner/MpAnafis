//! Exponentiation operations (standard and Montgomery sliding window exponentiation).

use core::{array::from_fn, cmp::Ordering, mem::swap};

use super::{
    BarrettDomain, BarrettScratch, DivScratch, Division, InternalMpUint, LIMB_BITS, Limb,
    MontgomeryDomain, MulScratch,
};

impl InternalMpUint {
    /// Computes `self^exp`.
    #[must_use]
    pub fn pow(&self, exp: u32) -> Self {
        let base_val = self;
        if exp == 0 {
            return Self::one();
        }
        let mut result = Self::zero();
        let mut started = false;
        let mut temp = Self::zero();
        let mut scratch = MulScratch::default();

        for i in (0..32).rev() {
            if started {
                temp.assign_product_with_scratch(&result, &result, &mut scratch);
                swap(&mut result, &mut temp);
            }
            if (exp >> i) & 1 == 1 {
                if started {
                    temp.assign_product_with_scratch(&result, base_val, &mut scratch);
                    swap(&mut result, &mut temp);
                } else {
                    result.clone_from(base_val);
                    started = true;
                }
            }
        }
        result
    }
}

impl MontgomeryDomain {
    /// Computes `base^exp` in this Montgomery domain.
    ///
    /// When `raw` is true, the result remains in Montgomery form.
    #[allow(
        clippy::too_many_lines,
        reason = "Window precomputation and left-to-right exponentiation remain together to share reusable Montgomery scratch."
    )]
    #[allow(
        unsafe_code,
        reason = "select_window_size returns 3..=6, so table_size is 4..=32; precompute indices are below table_size and each best_val >> 1 is below 2^(window-1) = table_size."
    )]
    pub fn pow(
        &self,
        base: &InternalMpUint,
        exp: &InternalMpUint,
        mul_scratch: &mut MulScratch,
        raw: bool,
    ) -> InternalMpUint {
        let domain = self;
        let mut temp_prod = InternalMpUint::zero();

        let bits = exp.significant_bits();
        if bits == 0 {
            return if raw {
                domain.transform_into_with_scratch(
                    &InternalMpUint::one(),
                    &mut temp_prod,
                    mul_scratch,
                )
            } else {
                InternalMpUint::one()
            };
        }

        let window = select_window_size(bits);
        let table_size = 1_usize << window.wrapping_sub(1);

        let mut g: [InternalMpUint; 32] = from_fn(|_| InternalMpUint::zero());
        // SAFETY: the table has 32 elements, so index zero is in bounds.
        *unsafe { g.get_unchecked_mut(0) } =
            domain.transform_into_with_scratch(base, &mut temp_prod, mul_scratch);

        if table_size > 1 {
            let mut base2 = InternalMpUint::zero();
            // SAFETY: the table has 32 elements, so index zero is in bounds.
            domain.square_into_with_scratch(
                unsafe { g.get_unchecked(0) },
                &mut base2,
                &mut temp_prod,
                mul_scratch,
            );

            for i in 1..table_size {
                let (left, right) = g.split_at_mut(i);
                // SAFETY: `left.len() = i > 0`, so `i - 1` is in bounds.
                let left_factor = unsafe { left.get_unchecked(i.wrapping_sub(1)) };
                // SAFETY: `i < table_size <= 32 = g.len()`, so the split
                // leaves at least one element in `right`.
                let destination = unsafe { right.get_unchecked_mut(0) };
                domain.mul_into_with_scratch(
                    left_factor,
                    &base2,
                    destination,
                    &mut temp_prod,
                    mul_scratch,
                );
            }
        }

        let exp_limbs = exp.limbs();
        let mut result = InternalMpUint::zero();
        let mut next_res = InternalMpUint::zero();
        let mut started = false;

        let mut bit_pos = bits;
        while bit_pos > 0 {
            let bit = get_bit(exp_limbs, bit_pos.wrapping_sub(1));

            if bit == 0 {
                if started {
                    domain.square_into_with_scratch(
                        &result,
                        &mut next_res,
                        &mut temp_prod,
                        mul_scratch,
                    );
                    swap(&mut result, &mut next_res);
                }
                bit_pos = bit_pos.wrapping_sub(1);
            } else {
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "window max size is 6, safely fits in usize"
                )]
                let max_len = usize::min(window as usize, bit_pos);
                let mut window_val = 1;
                let mut best_val = 1;
                let mut best_len = 1;

                for l in 1..max_len {
                    let next_bit = get_bit(exp_limbs, bit_pos.wrapping_sub(1).wrapping_sub(l));
                    window_val = (window_val << 1) | next_bit;

                    if next_bit == 1 {
                        best_val = window_val;
                        best_len = l.wrapping_add(1);
                    }
                }

                if started {
                    for _ in 0..best_len {
                        domain.square_into_with_scratch(
                            &result,
                            &mut next_res,
                            &mut temp_prod,
                            mul_scratch,
                        );
                        swap(&mut result, &mut next_res);
                    }
                    domain.mul_into_with_scratch(
                        &result,
                        // SAFETY: `best_val` contains at most `window` bits, so
                        // `best_val >> 1 < 2^(window - 1) = table_size <= 32`.
                        unsafe { g.get_unchecked(best_val >> 1) },
                        &mut next_res,
                        &mut temp_prod,
                        mul_scratch,
                    );
                    swap(&mut result, &mut next_res);
                } else {
                    // SAFETY: `best_val >> 1 < table_size <= 32`, as proved
                    // by the bounded window construction above.
                    result.clone_from(unsafe { g.get_unchecked(best_val >> 1) });
                    started = true;
                }

                bit_pos = bit_pos.wrapping_sub(best_len);
            }
        }

        if raw {
            result
        } else {
            domain.transform_out_with_scratch(&result, &mut temp_prod, mul_scratch)
        }
    }
}

impl BarrettDomain {
    /// Computes `base^exp` in this Barrett domain.
    #[allow(
        clippy::too_many_lines,
        reason = "Window precomputation and left-to-right exponentiation remain together to share Barrett and multiplication scratch."
    )]
    #[allow(
        unsafe_code,
        reason = "select_window_size returns 3..=6, so table_size is 4..=32; precompute indices are below table_size and each best_val >> 1 is below 2^(window-1) = table_size."
    )]
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
            return InternalMpUint::one();
        }

        let window = select_window_size(bits);
        let table_size = 1_usize << window.wrapping_sub(1);

        let mut g: [InternalMpUint; 32] = from_fn(|_| InternalMpUint::zero());

        // Base might be larger than b^{2k}, so we MUST use standard division to reduce initially
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
            domain.square_into_with_barrett_scratch(
                // SAFETY: the table has 32 elements, so index zero is in bounds.
                unsafe { g.get_unchecked(0) },
                &mut base2,
                &mut temp_prod,
                mul_scratch,
                &mut barrett_scratch,
            );

            for i in 1..table_size {
                let (left, right) = g.split_at_mut(i);
                // SAFETY: `left.len() = i > 0`, so `i - 1` is in bounds.
                let left_factor = unsafe { left.get_unchecked(i.wrapping_sub(1)) };
                // SAFETY: `i < table_size <= 32 = g.len()`, so the split
                // leaves at least one element in `right`.
                let destination = unsafe { right.get_unchecked_mut(0) };
                domain.mul_into_with_barrett_scratch(
                    left_factor,
                    &base2,
                    destination,
                    &mut temp_prod,
                    mul_scratch,
                    &mut barrett_scratch,
                );
            }
        }

        let exp_limbs = exp.limbs();
        let mut result = InternalMpUint::zero();
        let mut next_res = InternalMpUint::zero();
        let mut started = false;

        let mut bit_pos = bits;
        while bit_pos > 0 {
            let bit = get_bit(exp_limbs, bit_pos.wrapping_sub(1));

            if bit == 0 {
                if started {
                    domain.square_into_with_barrett_scratch(
                        &result,
                        &mut next_res,
                        &mut temp_prod,
                        mul_scratch,
                        &mut barrett_scratch,
                    );
                    swap(&mut result, &mut next_res);
                }
                bit_pos = bit_pos.wrapping_sub(1);
            } else {
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "window max size is 6, safely fits in usize"
                )]
                let max_len = usize::min(window as usize, bit_pos);
                let mut window_val = 1;
                let mut best_val = 1;
                let mut best_len = 1;

                for l in 1..max_len {
                    let next_bit = get_bit(exp_limbs, bit_pos.wrapping_sub(1).wrapping_sub(l));
                    window_val = (window_val << 1) | next_bit;

                    if next_bit == 1 {
                        best_val = window_val;
                        best_len = l.wrapping_add(1);
                    }
                }

                if started {
                    for _ in 0..best_len {
                        domain.square_into_with_barrett_scratch(
                            &result,
                            &mut next_res,
                            &mut temp_prod,
                            mul_scratch,
                            &mut barrett_scratch,
                        );
                        swap(&mut result, &mut next_res);
                    }
                    domain.mul_into_with_barrett_scratch(
                        &result,
                        // SAFETY: `best_val` contains at most `window` bits, so
                        // `best_val >> 1 < 2^(window - 1) = table_size <= 32`.
                        unsafe { g.get_unchecked(best_val >> 1) },
                        &mut next_res,
                        &mut temp_prod,
                        mul_scratch,
                        &mut barrett_scratch,
                    );
                    swap(&mut result, &mut next_res);
                } else {
                    // SAFETY: `best_val >> 1 < table_size <= 32`, as proved
                    // by the bounded window construction above.
                    result.clone_from(unsafe { g.get_unchecked(best_val >> 1) });
                    started = true;
                }

                bit_pos = bit_pos.wrapping_sub(best_len);
            }
        }

        result
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Bit indexing on Limb requires shifts as usize/u32 composites"
)]
fn get_bit(limbs: &[Limb], bit_idx: usize) -> usize {
    let word_idx = bit_idx >> LIMB_BITS.trailing_zeros();
    let bit_offset = bit_idx & LIMB_BITS.wrapping_sub(1);
    limbs
        .get(word_idx)
        .map_or(0, |&val| (val >> bit_offset) & 1)
}

const fn select_window_size(bits: usize) -> u32 {
    if bits <= 64 {
        3
    } else if bits <= 256 {
        4
    } else if bits <= 1024 {
        5
    } else {
        6
    }
}
