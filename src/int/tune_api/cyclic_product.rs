//! Equivalent cyclic products with reusable operands and workspaces.

#![expect(
    unsafe_code,
    reason = "The materialized residue slice bounds high-product indices below twice a representable native-limb count"
)]

use core::fmt::{Debug, Formatter, Result as FmtResult};

use super::{InternalMpUint, Limb, MulScratch, Multiplication, ScratchBuffer};

/// Strategy for a product in the ring `Z/(B^w-1)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CyclicProductAlgorithm {
    /// Complete multiplication followed by an end-around-carry fold.
    Full,
    /// CRT when mathematically admitted, independent of the empirical cutoff.
    Cyclic,
    /// Shared production cutoff with complete multiplication as fallback.
    Production,
}

/// Direct cyclic-product comparison, excluding Newton and Montgomery work.
pub struct CyclicProductRunner {
    left: InternalMpUint,
    right: InternalMpUint,
    minimum: usize,
    width: usize,
    full: InternalMpUint,
    residue: ScratchBuffer,
    scratch: MulScratch,
}

impl Debug for CyclicProductRunner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("CyclicProductRunner")
            .field("minimum", &self.minimum)
            .field("width", &self.width)
            .finish_non_exhaustive()
    }
}

impl CyclicProductRunner {
    /// Retains positive operands and resolves a common output modulus once.
    ///
    /// If CRT has no shorter legal geometry, `w=minimum` and all strategies
    /// use the full-product fold. Otherwise `w` is the admitted CRT width.
    /// Setup and its first allocation are outside repeated execution.
    ///
    /// # Panics
    ///
    /// Panics for zero operands, `minimum=0`, or operands wider than `minimum`.
    #[must_use]
    pub fn new(left_limbs: &[Limb], right_limbs: &[Limb], minimum: usize) -> Self {
        assert!(
            minimum > 0 && left_limbs.len() <= minimum && right_limbs.len() <= minimum,
            "cyclic operands must fit the positive minimum width"
        );
        let left = InternalMpUint::from_limbs(left_limbs.to_vec());
        let right = InternalMpUint::from_limbs(right_limbs.to_vec());
        assert!(
            !left.is_zero() && !right.is_zero(),
            "cyclic operands must be positive"
        );
        let mut residue = ScratchBuffer::acquire(minimum);
        let mut scratch = MulScratch::default();
        let admitted = Multiplication::try_mul_mod_bnm1::<true, false>(
            left.limbs(),
            right.limbs(),
            minimum,
            &mut residue,
            &mut scratch,
        );
        let width = if admitted { residue.len() } else { minimum };
        Self {
            left,
            right,
            minimum,
            width,
            full: InternalMpUint::zero(),
            residue,
            scratch,
        }
    }

    /// Executes the selected calculation with retained storage.
    ///
    /// Each result has `w` limbs. Both all-zero and all-maximum limbs represent
    /// zero in this ring; comparisons must accept either representation.
    pub fn run(&mut self, algorithm: CyclicProductAlgorithm) -> &[Limb] {
        let admitted = match algorithm {
            CyclicProductAlgorithm::Full => false,
            CyclicProductAlgorithm::Cyclic => Multiplication::try_mul_mod_bnm1::<true, false>(
                self.left.limbs(),
                self.right.limbs(),
                self.minimum,
                &mut self.residue,
                &mut self.scratch,
            ),
            CyclicProductAlgorithm::Production => Multiplication::try_mul_mod_bnm1::<false, false>(
                self.left.limbs(),
                self.right.limbs(),
                self.minimum,
                &mut self.residue,
                &mut self.scratch,
            ),
        };
        if !admitted {
            self.full
                .assign_product_with_scratch(&self.left, &self.right, &mut self.scratch);
            let width = self.width;
            self.residue.resize(width, 0);
            let product = self.full.limbs();
            let mut carry = false;
            for (index, destination) in self.residue.iter_mut().enumerate() {
                let low = product.get(index).copied().unwrap_or(0);
                // SAFETY: resize establishes index<width=residue.len(). A
                // materialized native-limb slice has at least two bytes per
                // element, so twice its length fits usize on every target.
                let high_index = unsafe { index.unchecked_add(width) };
                let high = product.get(high_index).copied().unwrap_or(0);
                let (partial, first) = low.overflowing_add(high);
                let (sum, second) = partial.overflowing_add(Limb::from(carry));
                *destination = sum;
                carry = first || second;
            }
            // With R=B^w, each operand is at most R-1, so the high part
            // is at most R-2. A carried sum is at most R-3; adding its
            // end-around carry cannot overflow again.
            for destination in self.residue.iter_mut() {
                if !carry {
                    break;
                }
                let (sum, overflow) = destination.overflowing_add(1);
                *destination = sum;
                carry = overflow;
            }
        }
        &self.residue
    }
}
