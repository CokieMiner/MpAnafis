//! Profile catalog and selection for multi-modulus floating-point NTT.

use super::{MAX_PRIMES, PRIME_CATALOG};

mod catalog;
mod planner;

pub use catalog::{NttProfile, STANDARD_PROFILES};
pub use planner::PlannedProfile;
