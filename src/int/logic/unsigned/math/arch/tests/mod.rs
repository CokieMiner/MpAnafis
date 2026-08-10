//! Architecture-kernel properties grouped by operation.

#[cfg(all(
    feature = "std",
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(miri)
))]
use support::exact_limb_vec;
use support::{
    equal_length_limb_vecs, equal_length_odd_limb_vecs, limb_vec, montgomery_inverse,
    reference_add_multiply_limb, reference_add_multiply_two, reference_multiply_two,
};

mod addition;
mod division;
mod fused_multiply;
mod montgomery;
mod propagation;
mod shifts;
mod support;
#[cfg(all(
    feature = "std",
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(miri)
))]
mod x86_backends;
