//! Search phases in arithmetic dependency order.

use crate::{
    arguments::Mode,
    compiled::{
        CompiledTuner, CoordinateSearch, GCD_KNOBS, PARSING_KNOBS, PRODUCT_KNOBS, TOOM_KNOBS,
        TRANSFORM_SHAPE_KNOBS,
    },
    tiers::{FormattingTuner, TierTuner},
    validation::Validation,
};

use super::TuneSession;

impl TuneSession {
    /// Execute the selected phases; the session driver gates installation.
    pub fn run_phases(&mut self, mode: &Mode) {
        #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
        if let Mode::ParallelOnly(specification) = mode {
            CompiledTuner::tune_parallel(self, specification);
            return;
        }
        self.run_product_phases(mode);
        if matches!(
            mode,
            Mode::All | Mode::CompiledOnly | Mode::TiersOnly | Mode::DivisionOnly
        ) {
            CompiledTuner::tune_division(self);
        }
        if matches!(mode, Mode::All | Mode::TiersOnly | Mode::ModularOnly) {
            TierTuner::modular_pow(self);
        }
        if matches!(mode, Mode::All | Mode::TiersOnly | Mode::GcdOnly) {
            CoordinateSearch::tune_coordinates(self, "GCD", &GCD_KNOBS);
        }
        if matches!(mode, Mode::All | Mode::TiersOnly | Mode::FormattingOnly) {
            FormattingTuner::tune_formatting(self);
        }
        if matches!(mode, Mode::All | Mode::TiersOnly | Mode::ParsingOnly) {
            CoordinateSearch::tune_coordinates(self, "Radix parsing", PARSING_KNOBS);
        }
        if matches!(mode, Mode::All | Mode::FormattingOnly | Mode::ValidateOnly) {
            self.formatting_validated = Validation::formatting_boundaries(self);
        }
    }

    /// Reconciles primitive choices before tuning their division and modular consumers.
    fn run_product_phases(&mut self, mode: &Mode) {
        if matches!(mode, Mode::All | Mode::CompiledOnly | Mode::ToomOnly) {
            CoordinateSearch::tune_coordinates(self, "Toom reconstruction", &TOOM_KNOBS);
        }
        if matches!(mode, Mode::All | Mode::TiersOnly) {
            TierTuner::multiplication(self);
            TierTuner::low_product(self);
            TierTuner::squaring(self);
        }
        if matches!(mode, Mode::All) {
            // Reconstruction depends on lower tiers. Reconcile kernel choices
            // and entry thresholds after the conventional towers are selected.
            let before = self.profile;
            CoordinateSearch::tune_coordinates(self, "Toom reconstruction", &TOOM_KNOBS);
            if self.profile != before {
                TierTuner::multiplication(self);
                TierTuner::low_product(self);
                TierTuner::squaring(self);
            }
        }
        if matches!(mode, Mode::All | Mode::CompiledOnly) {
            CompiledTuner::tune_ssa(self);
        }
        if matches!(mode, Mode::All | Mode::TiersOnly) {
            TierTuner::transforms(self);
            CoordinateSearch::tune_coordinates(self, "Transform shapes", &TRANSFORM_SHAPE_KNOBS);
            // Full multiplication is one low-product candidate. Reconcile its
            // crossover once after final transform and shape selection.
            TierTuner::low_product(self);
        }
        if matches!(
            mode,
            Mode::All
                | Mode::CompiledOnly
                | Mode::TiersOnly
                | Mode::DivisionOnly
                | Mode::ModularOnly
        ) {
            CoordinateSearch::tune_coordinates(self, "Product policies", &PRODUCT_KNOBS);
        }
    }
}
