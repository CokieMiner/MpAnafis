//! Products, transforms, ring arithmetic, and workspace contracts for SSA.

use core::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use crate::{
    int::logic::unsigned::math::mul::{Schoolbook, ssa::SsaPointwise},
    parallel::{ParallelExecutor, SequentialExecutor},
};

use super::*;

mod boundaries;
mod coefficients;
mod convolution;
#[cfg(feature = "_internal-tune")]
mod direct;
mod executors;
mod fermat;
mod mersenne;
mod nested;
mod oracle;
mod parallel_policy;
mod products;
mod reconstruction;
mod scratch;
mod strategies;
mod truncated_products;
mod two_by_one;

pub use executors::{CountingExecutor, OneSlotCountingExecutor};
pub use oracle::{oracle_add_mod, oracle_shift};
pub use strategies::{operands, transform_operands};
