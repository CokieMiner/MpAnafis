//! Primality testing for [`InternalMpUint`].

#[cfg(test)]
use core::cmp::Ordering;

use super::{InternalMpUint, Limb, MontgomeryDomain, MulScratch};

use self::operations::Primality;
#[cfg(test)]
use self::operations::SIEVE_PRIMES;

mod miller_rabin;
mod operations;

#[cfg(test)]
mod tests;
