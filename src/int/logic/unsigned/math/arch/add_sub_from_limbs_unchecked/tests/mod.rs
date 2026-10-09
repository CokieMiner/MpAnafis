//! Shared-source arithmetic contracts and the four-lane carry model.

use super::Limb;

mod contracts;
mod prefix;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64"
))]
mod vector;
