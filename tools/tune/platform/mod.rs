//! Host discovery and measurement affinity.

mod affinity;
mod host;

pub use host::{AffinityIdentity, Platform};

#[cfg(test)]
mod tests;
