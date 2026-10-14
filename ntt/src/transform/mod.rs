//! Contiguous twiddle tables and cache-blocked radix-4 transforms.

use crate::arch::{NttFloatKernels, Radix4Kernel, Scalar};

use super::{ParallelExecutor, Prime};

mod forward;
mod inverse;
mod stage;
mod twiddle;

pub use forward::Transform;
pub use stage::Stage;
pub use twiddle::{Tables, Twiddles};

#[cfg(test)]
mod tests;
