//! How fast each multiplication tier is at each width.
//!
//! Every benchmark here forces one algorithm rather than going through the
//! dispatcher, so a tier can be measured outside the range the tower would
//! actually select it for. `gmp_reference` holds the matching GMP tier at the
//! same widths, which is what makes a forced measurement interpretable: on its
//! own, a Toom-6 timing says nothing about whether Toom-6 is any good.

pub mod algorithms;
pub mod gmp_reference;
pub mod squares;
