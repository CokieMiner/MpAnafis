//! Leaf arithmetic used inside the multiplication tower.
//!
//! These rows do not select production thresholds. They justify architecture
//! kernels against a portable implementation or an equivalent external `mpn`
//! primitive, with a bias-control row where benchmark ordering can matter.

pub mod addition;
pub mod paired_add_sub;
