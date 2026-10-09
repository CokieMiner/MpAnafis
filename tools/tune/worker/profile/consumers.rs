//! Complete production formatting and modular exponentiation catalogs.

use core::hint::black_box;

use mp_anafis::{
    MpUint,
    tune_api::{FormattingAlgorithm, FormattingRunner, ModularPowAlgorithm, ModularPowRunner},
};

use super::{HASH_A, HASH_B, InterleavedMeasure, ProfileWorkers, ScoreCell};

/// Representative formatting radices shared by consumer and boundary validation.
pub const CONSUMER_FORMAT_RADICES: [u32; 7] = [3, 5, 9, 10, 11, 19, 36];

const FORMAT_LIMB_WIDTHS: [usize; 10] = [4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048];
const POW_LIMB_WIDTHS: [usize; 12] = [8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768];
const POW_EXPONENT_LIMBS: [usize; 2] = [1, 4];

/// Complete consumer vector length, derived from the two Cartesian catalogs.
pub const CONSUMER_SCORE_COUNT: usize = CONSUMER_FORMAT_RADICES
    .len()
    .checked_mul(FORMAT_LIMB_WIDTHS.len())
    .expect("finite formatting catalog")
    .checked_add(
        POW_LIMB_WIDTHS
            .len()
            .checked_mul(POW_EXPONENT_LIMBS.len())
            .expect("finite exponentiation catalog"),
    )
    .expect("finite consumer catalog");

/// Production consumer measurements with independent forced-algorithm references.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConsumerWorker;

impl ConsumerWorker {
    /// Validate formatting and modular exponentiation through production dispatch.
    /// Forced schoolbook formatting and Montgomery supply reference outputs.
    pub fn print_score() {
        let mut values = Vec::with_capacity(CONSUMER_SCORE_COUNT);
        for radix in CONSUMER_FORMAT_RADICES {
            for len in FORMAT_LIMB_WIDTHS {
                let mut reference =
                    FormattingRunner::new(FormattingAlgorithm::Schoolbook, len, radix);
                let expected = reference.output();
                let value =
                    MpUint::from_str_radix(&expected, radix).expect("valid reference digits");
                assert_eq!(
                    value.to_string_radix(radix),
                    expected,
                    "production formatting differs from schoolbook"
                );
                println!("MP_ANAFIS_CELL format radix={radix} limbs={len}");
                values.push(InterleavedMeasure::median_batch_samples(
                    || {
                        drop(black_box(
                            black_box(&value).to_string_radix(black_box(radix)),
                        ));
                    },
                    1,
                    9,
                ));
            }
        }
        for len in POW_LIMB_WIDTHS {
            for exponent_limbs in POW_EXPONENT_LIMBS {
                let base = ScoreCell::operand(len, HASH_A);
                let modulus = ScoreCell::operand(len, HASH_B);
                let exponent = vec![usize::MAX; exponent_limbs];
                let mut runner = ModularPowRunner::new(&base, &exponent, &modulus);
                assert_eq!(
                    runner.run(ModularPowAlgorithm::Production),
                    runner.run(ModularPowAlgorithm::Montgomery),
                    "production pow_mod differs from Montgomery"
                );
                println!("MP_ANAFIS_CELL pow_mod limbs={len} exponent_limbs={exponent_limbs}");
                values.push(InterleavedMeasure::median_batch_samples(
                    || {
                        drop(black_box(
                            black_box(&mut runner).run(ModularPowAlgorithm::Production),
                        ));
                    },
                    1,
                    9,
                ));
            }
        }
        assert_eq!(
            values.len(),
            CONSUMER_SCORE_COUNT,
            "consumer protocol cell count"
        );
        ProfileWorkers::print_encoded("MP_ANAFIS_CONSUMER_SCORE=", &values);
    }
}
