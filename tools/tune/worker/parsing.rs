//! Complete parsing calls scored on frozen radix/chunk dimensions.

use core::hint::black_box;

use mp_anafis::tune_api::{ParsingAlgorithm, ParsingRunner};

use super::{CellSelection, InterleavedMeasure};

/// Endpoints and an interior radix for each independently tuned group.
pub const PARSING_RADICES: [u32; 7] = [3, 5, 9, 10, 11, 19, 36];
/// Native radix-chunk widths used by production parsing probes.
pub const PARSING_CHUNK_SIZES: [usize; 24] = [
    1, 2, 4, 8, 12, 16, 17, 18, 19, 20, 24, 32, 48, 64, 96, 128, 192, 224, 256, 320, 384, 512,
    1_024, 4_096,
];

/// Production parsing worker and independent schoolbook correctness reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParsingWorker;

impl ParsingWorker {
    /// Parse explicit dimensions before allocating any input or output buffers.
    pub fn widths(specification: &str) -> Result<Vec<usize>, String> {
        let widths = if specification.is_empty() {
            PARSING_CHUNK_SIZES.to_vec()
        } else {
            specification
                .split(',')
                .map(|field| {
                    field
                        .parse::<usize>()
                        .map_err(|error| format!("invalid parsing chunk count: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        // Every non-power-of-two radix uses fewer than Limb::BITS digits per
        // native chunk. This byte bound also covers two output limb buffers.
        let digit_bound = usize::try_from(usize::BITS).map_err(|error| error.to_string())?;
        if widths.is_empty()
            || widths.iter().any(|&width| {
                width == 0
                    || width
                        .checked_mul(digit_bound)
                        .is_none_or(|bytes| isize::try_from(bytes).is_err())
            })
        {
            return Err("parsing dimensions exceed the addressable input span".to_owned());
        }
        Ok(widths)
    }

    /// Emit radix-major timings; root and leaf cutoffs come from the compiled
    /// profile. An optional `;schoolbook` or `;recursive` suffix selects a
    /// forced tier for diagnosis; coordinate searches use production calls.
    pub fn print_score(specification: &str, selection: &str) -> Result<(), String> {
        let (dimensions, algorithm, label) = match specification.split_once(';') {
            Some((dimensions, "schoolbook")) => {
                (dimensions, ParsingAlgorithm::Schoolbook, "schoolbook")
            }
            Some((dimensions, "recursive")) => {
                (dimensions, ParsingAlgorithm::Recursive, "recursive")
            }
            Some(_) => return Err("invalid forced parsing algorithm".to_owned()),
            None => (specification, ParsingAlgorithm::Production, "production"),
        };
        let widths = Self::widths(dimensions)?;
        let count = widths
            .len()
            .checked_mul(PARSING_RADICES.len())
            .expect("finite parsing catalog");
        let indices = CellSelection::parse(selection, count)?;
        let mut values = Vec::with_capacity(indices.len());
        for index in indices {
            let radix = *PARSING_RADICES
                .get(index.div_euclid(widths.len()))
                .expect("validated radix index");
            let chunks = *widths
                .get(index.rem_euclid(widths.len()))
                .expect("validated width index");
            let runner = ParsingRunner::new(algorithm, chunks, radix);
            if !runner.verify() {
                return Err(format!(
                    "{label} parsing differs from schoolbook at radix {radix}, {chunks} chunks"
                ));
            }
            println!(
                "MP_ANAFIS_CELL parsing algorithm={label} index={index} radix={radix} chunks={chunks} bytes={}",
                runner.input().len()
            );
            values.push(InterleavedMeasure::median_batch_samples(
                || {
                    drop(black_box(runner.run()));
                },
                1,
                9,
            ));
        }
        println!(
            "MP_ANAFIS_PARSING_SCORE={}",
            values
                .iter()
                .map(u128::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        Ok(())
    }

    /// Logarithmic chunk weights preserve the exact radix-major worker order.
    #[must_use]
    pub fn cell_weights(widths: &[usize]) -> Vec<u32> {
        PARSING_RADICES
            .iter()
            .flat_map(|_| widths.iter().map(|&width| width.ilog2().max(1)))
            .collect()
    }
}
