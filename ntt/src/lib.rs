//! Exact multi-prime number-theoretic transforms and native-limb multiplication.
//!
//! Plans bound every convolution coefficient by the product of the selected
//! primes. Workspaces retain transform tables and buffers across operations.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod arch;
mod conversion;
mod crt;
mod plan;
mod prime;
mod profile;
mod support;
mod transform;

pub use plan::{Ntt, NttPlan, NttWorkspace};
pub(crate) use prime::{MAX_PRIMES, PRIME_CATALOG, Prime};
pub(crate) use profile::PlannedProfile;
pub(crate) use support::{DefaultExecutor, LIMB_BITS};
#[cfg(feature = "rayon")]
pub use support::{FixedParallelismExecutor, RayonExecutor};
pub use support::{Limb, ParallelExecutor, SequentialExecutor};

#[cfg(test)]
mod tests;
