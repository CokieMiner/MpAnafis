//! Number theory.
//!
//! - [`common_divisors`]: the GCD family, including the extended form modular
//!   inversion sits on.
//! - [`primality`]: Miller-Rabin and prime search, on the three input shapes
//!   that reach different amounts of the algorithm.
//! - [`roots`]: integer roots and the perfect-square test.
//! - [`special`]: factorial, Jacobi symbol, Euler's totient.

mod common_divisors;
mod primality;
mod roots;
mod special;
