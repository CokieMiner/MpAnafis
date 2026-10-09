//! Greatest-common-divisor worker fixtures and family-major scoring.

use core::{
    hint::black_box,
    mem::{size_of, swap},
};

use mp_anafis::MpUint;

use super::{CellSelection, InterleavedMeasure, ProfileWorkers};

/// Public GCD-family operation scored by the compiled worker.
#[derive(Clone, Copy, Debug)]
pub enum GcdOperation {
    /// Ordinary greatest common divisor.
    Gcd,
    /// Extended Euclidean coefficients.
    ExtendedGcd,
    /// Modular inversion.
    Invert,
    /// Jacobi symbol.
    Jacobi,
}

/// Family-major order of the compiled GCD worker protocol.
pub const GCD_OPERATIONS: [GcdOperation; 4] = [
    GcdOperation::Gcd,
    GcdOperation::ExtendedGcd,
    GcdOperation::Invert,
    GcdOperation::Jacobi,
];

/// Operand-pair distribution used by one GCD score cell.
#[derive(Clone, Copy, Debug)]
pub enum GcdShape {
    /// Independent random operands.
    Random,
    /// Operands of nearly equal magnitude.
    NearEqual,
    /// One operand much shorter than the other.
    Uneven,
    /// Operands sharing a nontrivial factor.
    SharedFactor,
    /// Consecutive Fibonacci values.
    Fibonacci,
    /// One operand fits in a single limb.
    Scalar,
    /// Exact multiple of the other operand.
    ExactMultiple,
    /// Operands separated by a bit gap.
    BitGap(u32),
}

/// Limb widths around simulation and recursive-cofactor crossovers, followed
/// by distinct quotient distributions. Every cell uses four operand pairs.
pub const GCD_SCORE_CASES: [(usize, GcdShape); 65] = [
    (1, GcdShape::Random),
    (1, GcdShape::NearEqual),
    (1, GcdShape::Uneven),
    (1, GcdShape::Scalar),
    (2, GcdShape::Scalar),
    (16, GcdShape::Scalar),
    (1, GcdShape::ExactMultiple),
    (4, GcdShape::ExactMultiple),
    (64, GcdShape::ExactMultiple),
    (256, GcdShape::ExactMultiple),
    (2, GcdShape::Random),
    (4, GcdShape::Random),
    (16, GcdShape::Random),
    (32, GcdShape::Random),
    (47, GcdShape::Random),
    (48, GcdShape::Random),
    (49, GcdShape::Random),
    (63, GcdShape::Random),
    (64, GcdShape::Random),
    (65, GcdShape::Random),
    (96, GcdShape::Random),
    (123, GcdShape::Random),
    (128, GcdShape::Random),
    (192, GcdShape::Random),
    (256, GcdShape::Random),
    (512, GcdShape::Random),
    (1_024, GcdShape::Random),
    (4_096, GcdShape::Random),
    (16, GcdShape::NearEqual),
    (64, GcdShape::NearEqual),
    (256, GcdShape::NearEqual),
    (16, GcdShape::Uneven),
    (64, GcdShape::Uneven),
    (256, GcdShape::Uneven),
    (64, GcdShape::SharedFactor),
    (256, GcdShape::SharedFactor),
    (16, GcdShape::Fibonacci),
    (64, GcdShape::Fibonacci),
    (256, GcdShape::Fibonacci),
    (3, GcdShape::Random),
    (5, GcdShape::Random),
    (6, GcdShape::Random),
    (7, GcdShape::Random),
    (8, GcdShape::Random),
    (12, GcdShape::Random),
    (24, GcdShape::Random),
    (31, GcdShape::Random),
    (33, GcdShape::Random),
    (95, GcdShape::Random),
    (97, GcdShape::Random),
    (127, GcdShape::Random),
    (129, GcdShape::Random),
    (1, GcdShape::BitGap(3)),
    (1, GcdShape::BitGap(4)),
    (1, GcdShape::BitGap(5)),
    (1, GcdShape::BitGap(7)),
    (1, GcdShape::BitGap(8)),
    (1, GcdShape::BitGap(9)),
    (1, GcdShape::BitGap(15)),
    (1, GcdShape::BitGap(16)),
    (1, GcdShape::BitGap(17)),
    (1, GcdShape::BitGap(31)),
    (1, GcdShape::BitGap(32)),
    (1, GcdShape::BitGap(33)),
    (1, GcdShape::BitGap(63)),
];

/// Constructed GCD-family operands and independently checked results.
pub struct GcdFixture {
    /// First operand.
    pub left: MpUint,
    /// Second operand.
    pub right: MpUint,
    /// Greatest common divisor of the operands.
    pub gcd: MpUint,
    /// Jacobi symbol of the operand pair.
    pub jacobi: i8,
}

/// Whole-profile GCD worker domains and fixture scoring.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GcdWorker;

impl GcdWorker {
    /// Print complete GCD, extended GCD, inversion, and Jacobi timings.
    pub fn print_score(selection: &str) -> Result<(), String> {
        #[cfg(not(target_pointer_width = "16"))]
        {
            let indices =
                CellSelection::parse(selection, Self::cell_weights(&GCD_SCORE_CASES).len())?;
            let values = Self::score_cells(0, &GCD_SCORE_CASES, &indices);
            ProfileWorkers::print_encoded("MP_ANAFIS_GCD_SCORE=", &values);
        }
        #[cfg(target_pointer_width = "16")]
        println!("MP_ANAFIS_GCD_SCORE=");
        Ok(())
    }

    /// Equal weights give each size and quotient distribution the same aggregate
    /// contribution. Candidate and final-validation guards check each cell too.
    #[must_use]
    pub fn cell_weights(cases: &[(usize, GcdShape)]) -> Vec<u32> {
        GCD_OPERATIONS
            .iter()
            .flat_map(|_| cases.iter().map(|_| 1))
            .collect()
    }

    /// Verify every fixture before measuring any family. Values are median
    /// picoseconds per public call, with operand construction outside the clock.
    pub fn score_cells(
        seed_offset: u64,
        cases: &[(usize, GcdShape)],
        selection: &[usize],
    ) -> Vec<u128> {
        let fixtures: Vec<_> = cases
            .iter()
            .enumerate()
            .map(|(case, &(len, shape))| {
                if !selection.is_empty()
                    && !selection
                        .iter()
                        .any(|index| index.rem_euclid(cases.len()) == case)
                {
                    return Vec::new();
                }
                (0..4)
                    .map(|seed| GcdFixture::new(len, shape, seed_offset.wrapping_add(seed)))
                    .collect::<Vec<_>>()
            })
            .collect();
        for cell in &fixtures {
            for fixture in cell {
                fixture.verify();
            }
        }
        let mut values = Vec::new();
        for (family, operation) in GCD_OPERATIONS.into_iter().enumerate() {
            for (case, (&(len, _), pairs)) in cases.iter().zip(&fixtures).enumerate() {
                let index = family
                    .checked_mul(cases.len())
                    .and_then(|offset| offset.checked_add(case))
                    .expect("GCD catalog fits");
                if !selection.is_empty() && selection.binary_search(&index).is_err() {
                    continue;
                }
                println!(
                    "MP_ANAFIS_CELL gcd family={family} case={case} limbs={len} pairs=4 seed_offset={seed_offset}"
                );
                let iterations = match len {
                    0..=16 => 128,
                    17..=64 => 32,
                    65..=256 => 8,
                    257..=1_024 => 2,
                    _ => 1,
                };
                let elapsed = match operation {
                    GcdOperation::Gcd => InterleavedMeasure::median_batch_samples(
                        || {
                            for pair in pairs {
                                drop(black_box(black_box(&pair.left).gcd(black_box(&pair.right))));
                            }
                        },
                        iterations,
                        9,
                    ),
                    GcdOperation::ExtendedGcd => InterleavedMeasure::median_batch_samples(
                        || {
                            for pair in pairs {
                                drop(black_box(
                                    black_box(&pair.left).extended_gcd(black_box(&pair.right)),
                                ));
                            }
                        },
                        iterations,
                        9,
                    ),
                    GcdOperation::Invert => InterleavedMeasure::median_batch_samples(
                        || {
                            for pair in pairs {
                                drop(black_box(
                                    black_box(&pair.left).invert(black_box(&pair.right)),
                                ));
                            }
                        },
                        iterations,
                        9,
                    ),
                    GcdOperation::Jacobi => InterleavedMeasure::median_batch_samples(
                        || {
                            for pair in pairs {
                                let _ = black_box(
                                    black_box(&pair.left).jacobi_symbol(black_box(&pair.right)),
                                );
                            }
                        },
                        iterations,
                        9,
                    ),
                };
                values.push(elapsed.div_euclid(4));
            }
        }
        values
    }

    /// Binary Jacobi also yields the GCD for a positive odd denominator. Each
    /// subtraction preserves GCD, removes powers of two from the numerator,
    /// and records reciprocity before ordering the next subtraction.
    pub fn binary_reference(left: &MpUint, right: &MpUint) -> (MpUint, i8) {
        assert!(right.is_odd(), "binary oracle requires an odd denominator");
        let mut numerator = left.clone();
        let mut denominator = right.clone();
        let mut negative = false;
        while !numerator.is_zero() {
            let zeros = numerator.trailing_zeros();
            numerator = shift_right(&numerator, zeros);
            let denominator_low = (&denominator & MpUint::from(7_u8))
                .to_u64()
                .expect("three-bit residue");
            negative ^= zeros & 1 != 0 && matches!(denominator_low, 3 | 5);
            if numerator < denominator {
                let numerator_low = (&numerator & MpUint::from(3_u8))
                    .to_u64()
                    .expect("two-bit residue");
                negative ^= numerator_low == 3 && denominator_low & 3 == 3;
                swap(&mut numerator, &mut denominator);
            }
            numerator = numerator
                .checked_sub(&denominator)
                .expect("ordered subtraction");
        }
        let jacobi = if denominator.is_one() {
            if negative { -1 } else { 1 }
        } else {
            0
        };
        (denominator, jacobi)
    }
}

impl GcdFixture {
    pub fn new(len: usize, shape: GcdShape, seed: u64) -> Self {
        let byte_len = len
            .checked_mul(size_of::<usize>())
            .expect("fixture byte width fits");
        let mut state = seed.wrapping_add(42);
        let mut operands = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut bytes = Vec::with_capacity(byte_len);
            for _ in 0..byte_len {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                bytes.push(state.to_be_bytes()[0]);
            }
            *bytes.first_mut().expect("nonempty fixture") |= 1;
            *bytes.last_mut().expect("nonempty fixture") |= 128;
            operands.push(MpUint::from_le_bytes(&bytes));
        }
        let mut right = operands.pop().expect("second operand exists");
        let mut left = operands.pop().expect("first operand exists");
        match shape {
            GcdShape::Random => {}
            GcdShape::BitGap(gap) => {
                let shift = usize::try_from(gap.min(usize::BITS.saturating_sub(1)))
                    .expect("native bit gap fits usize");
                left = shift_right(&right, shift);
                left |= MpUint::one();
            }
            GcdShape::Scalar => left = MpUint::from(seed.wrapping_mul(2).wrapping_add(5)),
            GcdShape::ExactMultiple => {
                left = right
                    .checked_mul(&MpUint::from(3_u8))
                    .expect("exact multiple");
            }
            GcdShape::NearEqual => {
                left = right
                    .checked_add(&MpUint::from(2_u8))
                    .expect("near-equal sum");
            }
            GcdShape::Uneven => {
                right = shift_right(&right, byte_len.checked_mul(7).expect("fixture shift fits"));
                right |= MpUint::one();
            }
            GcdShape::SharedFactor => {
                let bits = byte_len.checked_mul(4).expect("half fixture width fits");
                left = shift_right(&left, bits);
                right = shift_right(&right, bits);
                left |= MpUint::one();
                right |= MpUint::one();
                let factor = shift_left(&MpUint::one(), bits)
                    .checked_add(&MpUint::one())
                    .expect("shared factor");
                left = left.checked_mul(&factor).expect("shared left");
                right = right.checked_mul(&factor).expect("shared right");
            }
            GcdShape::Fibonacci => {
                let bits = byte_len.checked_mul(8).expect("fixture width fits");
                let bound = shift_left(&MpUint::one(), bits);
                left = MpUint::one();
                right = MpUint::from(2_u8);
                while right < bound {
                    let next = left.checked_add(&right).expect("Fibonacci step");
                    left = right;
                    right = next;
                }
                // Preserve an odd Jacobi denominator and vary the initial
                // quotient while retaining the long quotient-one suffix.
                if right.is_even() {
                    swap(&mut left, &mut right);
                }
                let addend = right
                    .checked_mul(&MpUint::from(seed))
                    .expect("Fibonacci seed product");
                left = left.checked_add(&addend).expect("Fibonacci seed sum");
            }
        }
        let (gcd, jacobi) = GcdWorker::binary_reference(&left, &right);
        Self {
            left,
            right,
            gcd,
            jacobi,
        }
    }

    /// Binary subtraction and reciprocity provide references independent of
    /// the tuned Lehmer/HGCD transition and recursive coefficient policies.
    pub fn verify(&self) {
        assert_eq!(
            self.left.gcd(&self.right),
            self.gcd,
            "compiled GCD failed binary oracle"
        );
        let (gcd, x, y) = self
            .left
            .extended_gcd(&self.right)
            .expect("nonzero divisor");
        assert_eq!(gcd, self.gcd, "compiled extended GCD failed binary oracle");
        let left_x = self.left.checked_mul(&x).expect("first Bezout product");
        let right_y = self.right.checked_mul(&y).expect("second Bezout product");
        assert_eq!(
            left_x.checked_rem(&self.right).expect("nonzero right"),
            gcd.checked_rem(&self.right).expect("nonzero right"),
            "invalid first Bezout residue"
        );
        assert_eq!(
            right_y.checked_rem(&self.left).expect("nonzero left"),
            gcd.checked_rem(&self.left).expect("nonzero left"),
            "invalid second Bezout residue"
        );
        match self.left.invert(&self.right) {
            Some(inverse) => {
                assert!(self.gcd.is_one(), "inverse exists for non-coprime fixture");
                let product = self.left.checked_mul(&inverse).expect("inverse product");
                assert_eq!(
                    product.checked_rem(&self.right).expect("nonzero right"),
                    MpUint::one(),
                    "invalid inverse residue"
                );
            }
            None => assert!(!self.gcd.is_one(), "coprime fixture has no inverse"),
        }
        assert_eq!(
            self.left.jacobi_symbol(&self.right),
            Some(self.jacobi),
            "compiled Jacobi failed binary oracle"
        );
    }
}

fn shift_left(value: &MpUint, bits: usize) -> MpUint {
    value.checked_shl(bits).expect("left shift fits")
}

fn shift_right(value: &MpUint, bits: usize) -> MpUint {
    if bits == 0 {
        return value.clone();
    }
    let divisor = MpUint::one()
        .checked_shl(bits)
        .expect("right-shift divisor");
    value.checked_div(&divisor).unwrap_or_else(MpUint::zero)
}
