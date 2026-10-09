//! Complete products, fixed geometries, and specialized row contracts.

mod fixed;
mod products;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(target_feature = "adx", target_feature = "bmi2")),
))]
mod rows;
