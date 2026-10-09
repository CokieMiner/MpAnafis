//! Bounded expansion of coordinate grids at an incumbent boundary.

use super::{Knob, TuningProfile};

/// Two geometric extensions bound each coordinate's extra compilation work.
const MAX_EXPANSIONS: usize = 2;

/// Search values retained across passes, with a fixed expansion envelope.
#[derive(Clone, Debug)]
pub struct CoordinateGrid {
    /// Declared, incumbent, and expanded values used by screening and refinement.
    pub values: Vec<usize>,
    /// Frozen lower bound of geometric exploration.
    pub minimum: usize,
    /// Frozen upper bound of geometric exploration.
    pub maximum: usize,
    /// Extensions already used across all passes in this phase.
    pub expansions: usize,
}

impl CoordinateGrid {
    /// Freeze the envelope before searching. Each extension halves or doubles
    /// an enabled boundary; zero remains an explicitly declared strategy switch.
    #[must_use]
    pub fn new(knob: &Knob<'_>, profile: &TuningProfile) -> Self {
        let mut values = knob.candidates.to_vec();
        values.push((knob.get)(*profile));
        values.sort_unstable();
        values.dedup();
        let first = values.iter().copied().find(|&value| value > 0).unwrap_or(1);
        let last = values.last().copied().unwrap_or(1);
        let mut minimum = first;
        let mut maximum = last;
        for _ in 0..MAX_EXPANSIONS {
            minimum = minimum.div_euclid(2).max(1);
            maximum = maximum.checked_mul(2).unwrap_or(usize::MAX - 2);
        }
        Self {
            values,
            minimum,
            maximum,
            expansions: 0,
        }
    }

    /// Extend only an enabled incumbent at an admissible grid edge. Complete
    /// profile validation enforces domain bounds and parameter relationships.
    /// The envelope and shared budget stay fixed through all coordinate passes.
    pub fn expand(&mut self, knob: &Knob<'_>, profile: &TuningProfile) -> bool {
        let current = (knob.get)(*profile);
        if current == 0 || self.expansions == MAX_EXPANSIONS {
            return false;
        }
        let admissible: Vec<_> = self
            .values
            .iter()
            .copied()
            .filter(|&value| {
                let mut trial = *profile;
                (knob.set)(&mut trial, value);
                value > 0 && trial.validate().is_ok()
            })
            .collect();
        let mut additions = Vec::new();
        if admissible.first() == Some(&current) {
            additions.push(current.div_euclid(2).max(self.minimum));
        }
        if admissible.last() == Some(&current) {
            additions.push(
                current
                    .checked_mul(2)
                    .unwrap_or(self.maximum)
                    .min(self.maximum),
            );
        }
        additions.retain(|&value| {
            let mut trial = *profile;
            (knob.set)(&mut trial, value);
            !self.values.contains(&value) && trial.validate().is_ok()
        });
        if additions.is_empty() {
            return false;
        }
        self.values.extend(additions);
        self.values.sort_unstable();
        self.values.dedup();
        self.expansions = self.expansions.checked_add(1).expect("bounded expansions");
        true
    }
}
