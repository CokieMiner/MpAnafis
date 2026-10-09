//! Interleaved A/B/B/A and odd-median batch sampling kernels.

use core::time::Duration;
use std::time::Instant;

use super::PairedStatistics;

/// Interleaved A/B/B/A and odd-median batch sampling kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterleavedMeasure;

impl InterleavedMeasure {
    /// Return a configurable odd-sample median in picoseconds per invocation.
    ///
    /// Compile-time SSA candidates include RAM-sized cells whose individual runs
    /// take seconds. Their inner medians screen candidates; fresh outer
    /// paired slots establish confirmation bounds under the inference assumptions.
    ///
    /// # Panics
    ///
    /// Panics if `sample_count` is 0 or even, or if `iterations` is 0.
    pub fn median_batch_samples<F: FnMut()>(
        mut operation: F,
        iterations: u32,
        sample_count: usize,
    ) -> u128 {
        assert!(
            sample_count > 0 && !sample_count.is_multiple_of(2),
            "median sample count must be positive and odd"
        );
        assert!(iterations > 0, "measurement iterations must be positive");
        let warmup_iterations = iterations.div_euclid(4).max(1);
        for _ in 0..warmup_iterations {
            operation();
        }

        let calibrated = calibrate_batch(&mut operation, iterations);
        let mut samples = vec![Duration::MAX; sample_count];
        for sample in &mut samples {
            *sample = time_batch(&mut operation, calibrated);
            println!(
                "MP_ANAFIS_SAMPLE iterations={calibrated} total_ns={}",
                sample.as_nanos()
            );
        }
        Self::median_of(samples)
            .as_nanos()
            .checked_mul(1_000)
            .expect("measurement picoseconds fit u128")
            .div_euclid(u128::from(calibrated))
    }

    /// Compare two operations with paired, symmetric interleaved sampling.
    ///
    /// Even slots run A/B/B/A and odd slots B/A/A/B. Summing each operation's two
    /// batches inside the same slot reduces slow frequency drift; alternating the
    /// outer operation prevents a persistent first/last-position advantage. The
    /// returned durations belong to the slot with the median paired ratio.
    /// A nonzero `confidence_bits` requests fresh confirmation with at most
    /// three scheduled looks; zero returns a directional screening ratio only.
    /// `maximum_ratio` is fixed before measuring the pilot.
    ///
    /// # Panics
    ///
    /// Panics if `sample_count` is 0 or even, or if `iterations` is 0.
    pub fn paired_batches(
        mut baseline: impl FnMut(),
        mut candidate: impl FnMut(),
        iterations: u32,
        sample_count: usize,
        confidence_bits: u32,
        maximum_ratio: u128,
    ) -> (Duration, Duration, u128) {
        assert!(
            sample_count > 0 && !sample_count.is_multiple_of(2),
            "median sample count must be positive and odd"
        );
        assert!(iterations > 0, "measurement iterations must be positive");
        let warmup_iterations = iterations.div_euclid(4).max(1);
        for _ in 0..warmup_iterations {
            baseline();
            candidate();
        }

        let calibrated = calibrate_batch(&mut baseline, iterations)
            .max(calibrate_batch(&mut candidate, iterations));
        let mut count = sample_count;
        let mut pilot = true;
        let mut samples = Vec::new();
        loop {
            for slot in samples.len()..count {
                samples.push(time_pair(
                    &mut baseline,
                    &mut candidate,
                    calibrated,
                    pilot,
                    count,
                    slot,
                ));
            }
            samples.sort_unstable_by_key(|&(baseline_time, candidate_time)| {
                candidate_time
                    .as_nanos()
                    .checked_mul(1_000_000)
                    .expect("paired measurement scale fits u128")
                    .div_ceil(baseline_time.as_nanos().max(1))
            });
            let mut ratios: Vec<_> = samples
                .iter()
                .map(|&(baseline_time, candidate_time)| {
                    candidate_time
                        .as_nanos()
                        .checked_mul(1_000_000)
                        .expect("paired ratio fits u128")
                        .div_ceil(baseline_time.as_nanos().max(1))
                })
                .collect();
            if pilot && confidence_bits > 0 {
                count = PairedStatistics::confirmation_samples(&ratios, confidence_bits);
                pilot = false;
                samples.clear();
                continue;
            }
            let &(baseline_time, candidate_time) = samples
                .get(count.div_euclid(2))
                .expect("positive sample count has a median");
            let upper = if confidence_bits == 0 {
                *ratios
                    .get(count.div_euclid(2))
                    .expect("screening ratio median")
            } else {
                let estimate = PairedStatistics::median_bounds(&mut ratios, confidence_bits);
                println!(
                    "MP_ANAFIS_CONFIRM slots={count} confidence_bits={confidence_bits} median_ppm={} lower_ppm={} upper_ppm={}",
                    estimate.median, estimate.lower, estimate.upper
                );
                if estimate.upper > maximum_ratio && estimate.lower <= maximum_ratio && count < 127
                {
                    count = if count == 31 { 63 } else { 127 };
                    continue;
                }
                estimate.upper
            };
            return (baseline_time, candidate_time, upper);
        }
    }

    /// Return the median of an odd-length vector of durations.
    ///
    /// # Panics
    ///
    /// Panics if `samples` is empty.
    #[must_use]
    pub fn median_of(mut samples: Vec<Duration>) -> Duration {
        samples.sort_unstable();
        samples
            .get(samples.len().div_euclid(2))
            .copied()
            .expect("positive odd sample count has a median")
    }
}

/// One fresh slot retains its four raw batches and symmetric order.
fn time_pair(
    baseline: &mut impl FnMut(),
    candidate: &mut impl FnMut(),
    iterations: u32,
    pilot: bool,
    count: usize,
    slot: usize,
) -> (Duration, Duration) {
    let (a1, b1, b2, a2) = if slot.is_multiple_of(2) {
        (
            time_batch(baseline, iterations),
            time_batch(candidate, iterations),
            time_batch(candidate, iterations),
            time_batch(baseline, iterations),
        )
    } else {
        let b1 = time_batch(candidate, iterations);
        let a1 = time_batch(baseline, iterations);
        let a2 = time_batch(baseline, iterations);
        let b2 = time_batch(candidate, iterations);
        (a1, b1, b2, a2)
    };
    println!(
        "MP_ANAFIS_PAIR pilot={pilot} slots={count} slot={slot} iterations={iterations} a1_ns={} b1_ns={} b2_ns={} a2_ns={}",
        a1.as_nanos(),
        b1.as_nanos(),
        b2.as_nanos(),
        a2.as_nanos()
    );
    (a1.saturating_add(a2), b1.saturating_add(b2))
}

/// Establish a two-millisecond batch floor separately for each measured cell.
/// Large operations keep their original batch; doubling is bounded by u32.
fn calibrate_batch(operation: &mut impl FnMut(), minimum: u32) -> u32 {
    let mut iterations = minimum;
    while time_batch(operation, iterations) < Duration::from_millis(2) {
        let Some(next) = iterations.checked_mul(2) else {
            break;
        };
        iterations = next;
    }
    iterations
}

fn time_batch(operation: &mut impl FnMut(), iterations: u32) -> Duration {
    let started = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    started.elapsed()
}
