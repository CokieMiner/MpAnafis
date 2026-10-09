//! Shared selection contracts and independent limb-arithmetic oracles.

pub mod cases;
mod limb_product;
pub mod oracle;
#[cfg(all(
    feature = "std",
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(miri),
))]
mod runtime;
