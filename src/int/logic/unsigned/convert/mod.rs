//! Native integer, floating-point, byte, and radix conversions.

use super::{
    Addition, BarrettDomain, BarrettScratch, Division, DoubleLimb, INLINE_LIMBS, InternalMpUint,
    LIMB_BITS, LIMB_BYTES, Limb, MulScratch, Multiplication,
    RADIX_FORMAT_DECIMAL_RECURSIVE_THRESHOLD, RADIX_FORMAT_LARGE_RECURSIVE_THRESHOLD,
    RADIX_FORMAT_SMALL_RECURSIVE_THRESHOLD, RADIX_PARSE_DECIMAL_RECURSIVE_THRESHOLD,
    RADIX_PARSE_LARGE_RECURSIVE_THRESHOLD, RADIX_PARSE_LEAF_CHUNKS,
    RADIX_PARSE_SMALL_RECURSIVE_THRESHOLD,
};

mod bytes;
mod decimal;
mod digits;
mod float;
mod format;
mod native;
mod parse;
mod radix;
mod reconstruct;
mod recursive;

pub use radix::{
    BASE4_BYTE_DIGITS, BASE8_DIGITS, BASE32_DIGITS, BINARY_BYTE_DIGITS, Convert, HEX_BYTE_DIGITS,
    RADIX_CHUNK_RECIPROCALS, RadixParameters,
};
pub use recursive::FormatCache;

#[cfg(test)]
mod tests;
