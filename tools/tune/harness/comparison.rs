//! Fresh compiled-profile confirmation using whole-worker outer paired slots.

use std::path::PathBuf;

use crate::measure::PairedStatistics;

use super::{CandidateHarness, ComparisonDecision, ComparisonRule, ScoreDomain, TuningProfile};

impl CandidateHarness {
    /// Upper bounds on median paired ratios, in ppm, for each worker cell.
    ///
    /// Both binaries are frozen before timing. Nine outer A/B/B/A slots form
    /// a pilot; fresh slots support scheduled looks at 31, 63 and 127. Inner
    /// batch repetitions are never counted as independent profile comparisons.
    /// The rule is frozen before the pilot. Both tails at all three looks share
    /// the error budget with every cell and every earlier fresh comparison.
    /// Captures retain raw workers, sample counts, measured medians and bounds.
    pub fn compare_profiles(
        &mut self,
        baseline: &TuningProfile,
        candidate: &TuningProfile,
        domain: ScoreDomain,
        rule: &ComparisonRule,
    ) -> Option<Vec<u128>> {
        if !self.check_context() {
            return None;
        }
        let (flag, prefix, features) = domain.protocol(self);
        let execution = (|| {
            let paths = [
                self.executables.prepare(baseline, features)?,
                self.executables.prepare(candidate, features)?,
            ];
            let mut count = 9;
            let mut pilot = true;
            let mut confidence_bits = 0;
            let mut blocks = Vec::new();
            loop {
                for slot in blocks.len()..count {
                    let ratios = self.profile_slot(&paths, &flag, prefix, slot)?;
                    if blocks
                        .first()
                        .is_some_and(|first: &Vec<u128>| first.len() != ratios.len())
                    {
                        return Err("outer slot cell counts disagree".to_owned());
                    }
                    println!(
                        "MP_ANAFIS_PROFILE_PAIR pilot={pilot} slots={count} slot={slot} ratios_ppm={}",
                        ratios
                            .iter()
                            .map(u128::to_string)
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    blocks.push(ratios);
                }
                let cells = blocks.first().expect("positive outer slot count").len();
                if rule.scores.iter().any(|score| score.weights.len() != cells) {
                    return Err("confirmation rule and worker cell counts disagree".to_owned());
                }
                if pilot {
                    self.confirmations = self
                        .confirmations
                        .checked_add(1)
                        .expect("finite tuning comparisons");
                    confidence_bits = PairedStatistics::confidence_bits(self.confirmations, cells);
                    count = (0..cells)
                        .map(|cell| {
                            let ratios: Vec<_> = blocks
                                .iter()
                                .map(|block| *block.get(cell).expect("equal cell counts"))
                                .collect();
                            PairedStatistics::confirmation_samples(&ratios, confidence_bits)
                        })
                        .max()
                        .expect("nonempty worker grid");
                    pilot = false;
                    blocks.clear();
                    continue;
                }
                let mut estimates = Vec::with_capacity(cells);
                for cell in 0..cells {
                    let mut ratios: Vec<_> = blocks
                        .iter()
                        .map(|block| *block.get(cell).expect("equal cell counts"))
                        .collect();
                    let estimate = PairedStatistics::median_bounds(&mut ratios, confidence_bits);
                    println!(
                        "MP_ANAFIS_PROFILE_CONFIRM comparison={} cell={cell} slots={count} confidence_bits={confidence_bits} median_ppm={} lower_ppm={} upper_ppm={}",
                        self.confirmations, estimate.median, estimate.lower, estimate.upper
                    );
                    estimates.push(estimate);
                }
                let decision = rule.decision(&estimates);
                println!(
                    "MP_ANAFIS_PROFILE_LOOK comparison={} slots={count} decision={}",
                    self.confirmations,
                    match decision {
                        ComparisonDecision::Accept => "accept",
                        ComparisonDecision::Reject => "reject",
                        ComparisonDecision::Unresolved => "unresolved",
                    }
                );
                if decision != ComparisonDecision::Unresolved || count == 127 {
                    return Ok(estimates.iter().map(|estimate| estimate.upper).collect());
                }
                count = if count == 31 { 63 } else { 127 };
            }
        })();
        if !self.check_context() {
            return None;
        }
        match execution {
            Ok(bounds) => Some(bounds),
            Err(error) => {
                self.failed = true;
                eprintln!("paired measurement failed: {error}");
                None
            }
        }
    }

    /// Execute one symmetric outer slot using frozen binary identities and the
    /// same catalog mask. Worker repetitions form one observation per cell.
    fn profile_slot(
        &mut self,
        paths: &[PathBuf; 2],
        flag: &str,
        prefix: &str,
        slot: usize,
    ) -> Result<Vec<u128>, String> {
        let order = if slot.is_multiple_of(2) {
            [0, 1, 1, 0]
        } else {
            [1, 0, 0, 1]
        };
        let mut samples = [Vec::new(), Vec::new()];
        for index in order {
            let path = paths.get(index).expect("paired order is binary");
            samples
                .get_mut(index)
                .expect("paired order is binary")
                .push(
                    self.executables
                        .run(path, flag, prefix, &self.selected_cells)?,
                );
        }
        let [baseline, candidate] = samples;
        Self::paired_ratios(&baseline, &candidate)
            .ok_or_else(|| "paired worker vectors disagree".to_owned())
    }
}
