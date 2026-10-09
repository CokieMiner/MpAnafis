//! Reusable state for production-tower benchmark sweeps.

use super::{BenchValidation, Limb, MulScratch, Multiplication};

/// Reusable state for the configured multiplication dispatcher.
#[derive(Debug, Default)]
pub struct MultiplicationBenchState {
    scratch: MulScratch,
}

impl MultiplicationBenchState {
    /// Validates one multiplication shape and binds it to reusable scratch.
    ///
    /// The first `a.len() + b.len()` destination limbs receive the product.
    /// Any remaining suffix is unchanged.
    ///
    /// # Panics
    ///
    /// Panics if either operand is empty or `dst` has fewer than
    /// `a.len() + b.len()` limbs.
    pub fn prepare<'state, 'data>(
        &'state mut self,
        dst: &'data mut [Limb],
        a: &'data [Limb],
        b: &'data [Limb],
    ) -> PreparedMultiplication<'state, 'data> {
        assert!(
            !a.is_empty() && !b.is_empty(),
            "reused multiplication operands must be nonempty"
        );
        let product_len = BenchValidation::product(dst, a, b);
        let (product, _) = dst.split_at_mut(product_len);
        PreparedMultiplication {
            scratch: &mut self.scratch,
            dst: product,
            a,
            b,
        }
    }
}

/// Validated production multiplication with reusable destination and scratch.
#[derive(Debug)]
pub struct PreparedMultiplication<'state, 'data> {
    scratch: &'state mut MulScratch,
    dst: &'data mut [Limb],
    a: &'data [Limb],
    b: &'data [Limb],
}

impl PreparedMultiplication<'_, '_> {
    /// Executes the configured multiplication tower with retained operands and scratch.
    #[inline]
    pub fn run(&mut self) {
        Multiplication::mul_limbs_with_scratch(self.a, self.b, self.dst, self.scratch);
    }
}

/// Reusable state for the configured squaring dispatcher.
#[derive(Debug, Default)]
pub struct SquaringBenchState {
    scratch: MulScratch,
}

impl SquaringBenchState {
    /// Validates one squaring shape and binds it to reusable scratch.
    ///
    /// The first `2 * a.len()` destination limbs receive the square.
    /// Any remaining suffix is unchanged.
    ///
    /// # Panics
    ///
    /// Panics if the operand is empty or `dst` has fewer than `2 * a.len()` limbs.
    pub fn prepare<'state, 'data>(
        &'state mut self,
        dst: &'data mut [Limb],
        a: &'data [Limb],
    ) -> PreparedSquaring<'state, 'data> {
        assert!(!a.is_empty(), "reused squaring operand must be nonempty");
        let square_len = BenchValidation::square(dst, a);
        let (square, _) = dst.split_at_mut(square_len);
        PreparedSquaring {
            scratch: &mut self.scratch,
            dst: square,
            a,
        }
    }
}

/// Validated production squaring with reusable destination and scratch.
#[derive(Debug)]
pub struct PreparedSquaring<'state, 'data> {
    scratch: &'state mut MulScratch,
    dst: &'data mut [Limb],
    a: &'data [Limb],
}

impl PreparedSquaring<'_, '_> {
    /// Executes the configured squaring tower with retained operands and scratch.
    #[inline]
    pub fn run(&mut self) {
        Multiplication::sqr_limbs_with_scratch(self.a, self.dst, self.scratch);
    }
}
