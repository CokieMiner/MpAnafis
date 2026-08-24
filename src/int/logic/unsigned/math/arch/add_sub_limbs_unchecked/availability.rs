//! Selected add/subtract carry-chain capability.

use super::ArchKernels;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "adx")
))]
use super::runtime_fast_add_sub_limbs_available;

#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "adx")
))]
impl ArchKernels {
    /// Returns whether the selected backend has independent carry chains.
    #[inline]
    pub fn fast_add_sub_limbs_available() -> bool {
        runtime_fast_add_sub_limbs_available()
    }
}

#[cfg(not(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "adx")
)))]
impl ArchKernels {
    /// Returns whether the selected backend has independent carry chains.
    #[inline]
    pub const fn fast_add_sub_limbs_available() -> bool {
        cfg!(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "adx"
        ))
    }
}
