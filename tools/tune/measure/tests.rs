//! Tests for sustained crossover measurement.

use core::{cell::RefCell, time::Duration};
use std::thread::sleep;

use super::{CrossoverMeasure, InterleavedMeasure, PairedStatistics, ProbeQuality};

#[test]
fn invalid_durations_cannot_establish_a_crossover() {
    for (candidate, baseline) in [(0, 1), (1, 0), (1, u128::MAX), (u128::MAX, u128::MAX)] {
        assert!(!CrossoverMeasure::confidently_faster_nanos(
            candidate, baseline, 7_500
        ));
    }
    assert!(CrossoverMeasure::confidently_faster_nanos(90, 100, 7_500));
    assert!(!CrossoverMeasure::confidently_faster_nanos(100, 100, 7_500));
}

#[test]
fn single_iteration_samples_warm_both_operations_and_alternate_slots() {
    let calls = RefCell::new(Vec::new());
    let _ = InterleavedMeasure::paired_batches(
        || {
            calls.borrow_mut().push('A');
            sleep(Duration::from_millis(3));
        },
        || {
            calls.borrow_mut().push('B');
            sleep(Duration::from_millis(3));
        },
        1,
        3,
        0,
        992_499,
    );
    assert_eq!(
        calls.into_inner(),
        "ABABABBABAABABBA".chars().collect::<Vec<_>>()
    );
    let mut count = 0;
    let _ = InterleavedMeasure::median_batch_samples(
        || {
            count += 1;
            sleep(Duration::from_millis(3));
        },
        1,
        3,
    );
    assert_eq!(count, 5);
}

#[test]
#[should_panic(expected = "measurement iterations must be positive")]
fn empty_measurements_are_rejected() {
    let _ = InterleavedMeasure::median_batch_samples(|| {}, 0, 3);
}

#[test]
fn sustained_crossover_finds_the_boundary_with_two_qualities() {
    let mut coarse_probes = 0;
    let mut precise_probes = 0;
    let result = CrossoverMeasure::sustained_crossover(
        8,
        &[8, 16, 32, 64, 128, 256, 512, 1_024],
        "test",
        |len, quality| {
            match quality {
                ProbeQuality::Coarse => coarse_probes += 1,
                ProbeQuality::Precise => precise_probes += 1,
            }
            len >= 200
        },
    );
    assert_eq!(result, Some(200));
    // Nine neighbouring widths, the guard and three remaining ladder points.
    assert_eq!(precise_probes, 13);
    assert!(coarse_probes > 0);
}

#[test]
fn sustained_crossover_rejects_an_isolated_noisy_win() {
    // A winning island that includes the nearby guard, then the real boundary.
    let result = CrossoverMeasure::sustained_crossover(
        8,
        &[8, 16, 32, 64, 128, 256, 512],
        "test",
        |len, _quality| (64..=80).contains(&len) || len >= 300,
    );
    // The spike survives its own guard only if it is sustained; it is not,
    // so the sweep continues and lands on the real boundary.
    assert_eq!(result, Some(300));
}

#[test]
fn sustained_crossover_checks_ladder_points_before_the_nearby_guard() {
    let probes = RefCell::new(Vec::new());
    let result = CrossoverMeasure::sustained_crossover(
        16,
        &[8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96],
        "nonmonotone tail",
        |len, quality| {
            if quality == ProbeQuality::Precise {
                probes.borrow_mut().push(len);
            }
            len >= 18 && (quality == ProbeQuality::Coarse || len != 20)
        },
    );
    assert_eq!(result, Some(21));
    assert!(probes.borrow().starts_with(&[16, 17, 18, 19, 20, 21, 22]));
}

#[test]
fn sustained_crossover_does_not_measure_a_duplicate_guard() {
    let probes = RefCell::new(Vec::new());
    let result =
        CrossoverMeasure::sustained_crossover(0, &[10, 12, 14, 26, 30], "tail", |len, quality| {
            if quality == ProbeQuality::Precise {
                probes.borrow_mut().push(len);
            }
            len >= 10
        });
    assert_eq!(result, Some(10));
    assert_eq!(*probes.borrow(), [6, 7, 8, 9, 10, 11, 12, 13, 14, 26, 30]);
}

#[test]
fn precise_neighbours_can_recover_an_earlier_boundary_missed_by_screening() {
    let result = CrossoverMeasure::sustained_crossover(
        128,
        &[128, 256, 512],
        "local recovery",
        |len, quality| {
            len >= if quality == ProbeQuality::Coarse {
                202
            } else {
                198
            }
        },
    );
    assert_eq!(result, Some(198));
}

#[test]
fn a_regression_between_ladder_points_moves_the_boundary_past_the_hole() {
    let result = CrossoverMeasure::sustained_crossover(
        128,
        &[128, 256, 512],
        "local hole",
        |len, quality| len >= 200 && (quality == ProbeQuality::Coarse || len != 202),
    );
    assert_eq!(result, Some(203));
}

#[test]
fn immediate_successors_are_checked_when_the_local_suffix_starts_at_its_edge() {
    let result = CrossoverMeasure::sustained_crossover(
        128,
        &[128, 256, 512, 1_024],
        "edge hole",
        |len, quality| {
            if quality == ProbeQuality::Coarse {
                len >= 200
            } else {
                len >= 204 && len != 206
            }
        },
    );
    assert_eq!(result, Some(207));
}

#[test]
fn exact_median_bounds_use_binomial_order_statistics() {
    let mut samples: Vec<_> = (1_u128..=31).collect();
    let estimate = PairedStatistics::median_bounds(&mut samples, 12);
    assert_eq!(
        (estimate.median, estimate.lower, estimate.upper),
        (16, 6, 26)
    );
    let inconclusive = PairedStatistics::median_bounds(&mut samples, 32);
    assert_eq!((inconclusive.lower, inconclusive.upper), (0, u128::MAX));
}

#[test]
fn median_bounds_respect_exact_tails_for_every_supported_odd_sample_count() {
    for count in (1_usize..=127).step_by(2) {
        let count_bits = u32::try_from(count).expect("bounded count");
        // Compute the binomial row independently by the multiplicative
        // recurrence C(n,k)=C(n,k-1)*(n-k+1)/k. Divide the predecessor first
        // into quotient and remainder to keep intermediate products in u128.
        let mut tails = vec![1_u128];
        let mut coefficient = 1_u128;
        let mut tail = coefficient;
        for index in 1..count.div_euclid(2) {
            let divisor = u128::try_from(index).expect("bounded divisor");
            let factor = u128::try_from(count - index + 1).expect("bounded factor");
            coefficient = coefficient.div_euclid(divisor) * factor
                + (coefficient % divisor * factor).div_euclid(divisor);
            tail += coefficient;
            tails.push(tail);
        }
        for bits in 1..=count_bits + 1 {
            let mut samples: Vec<_> = (1..=count)
                .map(|value| u128::try_from(value).expect("bounded sample"))
                .collect();
            let estimate = PairedStatistics::median_bounds(&mut samples, bits);
            assert_eq!(
                estimate.median,
                u128::try_from(count.div_euclid(2) + 1).expect("median")
            );
            if estimate.lower == 0 {
                assert_eq!(estimate.upper, u128::MAX);
                continue;
            }
            assert!(bits <= count_bits);
            let rank = usize::try_from(estimate.lower - 1).expect("bounded rank");
            let allowed_mass = 1_u128 << (count_bits - bits);
            assert!(*tails.get(rank).expect("finite lower rank") <= allowed_mass);
            assert_eq!(
                estimate.upper,
                u128::try_from(count - rank).expect("upper rank")
            );
            if let Some(&next_tail) = tails.get(rank + 1) {
                assert!(
                    next_tail > allowed_mass,
                    "next rank exceeds the tail budget"
                );
            }
        }
    }
}

#[test]
fn confirmation_budget_increases_with_dispersion_and_comparison_count() {
    assert_eq!(
        PairedStatistics::confirmation_samples(&[1_000_000; 9], 12),
        31
    );
    assert_eq!(
        PairedStatistics::confirmation_samples(&[1_000_000; 9], 32),
        63
    );
    assert_eq!(
        PairedStatistics::confirmation_samples(&[800_000, 1_200_000], 12),
        127
    );
    assert_eq!(PairedStatistics::confidence_bits(1, 1), 15);
    assert_eq!(PairedStatistics::confidence_bits(2, 5), 20);
    let mut samples = vec![1_000_000; 127];
    assert_eq!(
        PairedStatistics::median_bounds(&mut samples, 100).upper,
        1_000_000
    );
}

#[test]
fn noisy_confirmation_cannot_pass_on_its_median_alone() {
    let mut samples = vec![900_000; 16];
    samples.extend_from_slice(&[1_100_000; 15]);
    let estimate = PairedStatistics::median_bounds(&mut samples, 12);
    assert_eq!(estimate.median, 900_000);
    assert_eq!(estimate.upper, 1_100_000);
    assert!(!CrossoverMeasure::confidently_faster_nanos(
        estimate.upper,
        1_000_000,
        7_500
    ));
}

#[test]
fn scheduled_looks_cover_both_tails_with_one_simultaneous_budget() {
    // Per-comparison allocation is bounded by 6 * 2^-15 / j². The integral
    // bound sum(j^-2) < 2 gives 12 * 2^-15 < 0.001 for all comparisons.
    assert!(12 * 1_000 < (1_u64 << PairedStatistics::confidence_bits(1, 1)));
    for comparison in [1_u64, 2, 3, 32, 100, 10_000] {
        for cells in [1_usize, 2, 3, 16, 100_000] {
            let exponent = PairedStatistics::confidence_bits(comparison, cells);
            let allocated = 1_u128 << exponent;
            assert!(
                allocated
                    >= 32_768
                        * u128::from(comparison).pow(2)
                        * u128::try_from(cells).expect("cell count fits u128")
            );
            for count in [31_usize, 63, 127] {
                let mut samples: Vec<_> = (1..=count)
                    .map(|value| u128::try_from(value).expect("bounded sample count"))
                    .collect();
                let estimate = PairedStatistics::median_bounds(&mut samples, exponent);
                let count_bits = u32::try_from(count).expect("at most 127 observations");
                if exponent > count_bits {
                    assert_eq!((estimate.lower, estimate.upper), (0, u128::MAX));
                    continue;
                }
                let rank = usize::try_from(estimate.lower).expect("bounded order statistic") - 1;
                // Sum Pascal's row using an independent multiplicative
                // recurrence for each coefficient; the division is exact.
                let mut coefficient = 1_u128;
                let mut tail = coefficient;
                for index in 1..=rank {
                    let numerator = count - index + 1;
                    let denominator = index;
                    let divisor = u128::try_from(denominator).expect("bounded index");
                    let factor = u128::try_from(numerator).expect("bounded index");
                    let quotient = coefficient.div_euclid(divisor);
                    let remainder = coefficient % divisor;
                    coefficient = quotient * factor + (remainder * factor).div_euclid(divisor);
                    tail += coefficient;
                }
                assert!(tail <= (1_u128 << (count_bits - exponent)));
                assert_eq!(
                    estimate.upper,
                    u128::try_from(count - rank).expect("bounded rank")
                );
            }
        }
    }
}

#[test]
fn inconclusive_early_look_requires_additional_fresh_observations() {
    let bits = PairedStatistics::confidence_bits(1, 1);
    let mut samples = vec![900_000; 16];
    samples.extend([1_100_000; 15]);
    let first = PairedStatistics::median_bounds(&mut samples, bits);
    assert!(first.lower <= 992_499 && first.upper > 992_499);
    samples.extend([900_000; 32]);
    let second = PairedStatistics::median_bounds(&mut samples, bits);
    assert!(second.upper <= 992_499);
    assert_eq!(samples.len(), 63);
}
