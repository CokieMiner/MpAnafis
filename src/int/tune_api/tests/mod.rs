//! Tuning runner contracts through the exported facade and public integer API.

mod division;
mod formatting;
mod gcd;
mod low_product;
#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
mod low_product_parallel;
mod modular;
mod multiplication;
mod parsing;
mod products;
mod squaring;
mod strategies;
mod tier;
