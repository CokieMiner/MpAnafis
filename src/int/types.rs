//! Platform-adaptive limb types.
//!
//! `Limb` is `usize` so that native arithmetic is used on every target.
//! `DoubleLimb` is twice the width of `Limb` and serves as the accumulator
//! for limb multiplication, carry propagation, and division intermediates.
//!
//! # 16-bit targets
//!
//! All bit-counting casts from `u32` to `Limb` (`usize`) are safe because
//! the maximum bit count for a 64-bit limb is 64, which fits in `u16` (65 535).

/// Native unsigned integer type used to represent a single limb of an arbitrary-precision integer.
pub type Limb = usize;

/// Unsigned integer type with twice the bit width of [`Limb`], used for intermediate products and dividends.
#[cfg(target_pointer_width = "64")]
pub type DoubleLimb = u128;
/// Unsigned integer type with twice the bit width of [`Limb`], used for intermediate products and dividends.
#[cfg(target_pointer_width = "32")]
pub type DoubleLimb = u64;
/// Unsigned integer type with twice the bit width of [`Limb`], used for intermediate products and dividends.
#[cfg(target_pointer_width = "16")]
pub type DoubleLimb = u32;

/// Number of bits in a single limb.
#[allow(
    clippy::as_conversions,
    reason = "Limb::BITS is u32, fits in usize even on 16-bit targets (where Limb::BITS is 16 and usize is u16); casting avoids runtime range checks and compiles to a branchless compile-time constant."
)]
pub const LIMB_BITS: usize = Limb::BITS as usize;

/// Number of bytes in a single limb.
pub const LIMB_BYTES: usize = LIMB_BITS.wrapping_div(8);

/// Number of limbs stored inline before spilling to heap.
/// Fixed at 4 limbs — capacity scales with pointer width:
/// 256 bits (64-bit), 128 bits (32-bit), 64 bits (16-bit).
pub const INLINE_LIMBS: usize = 4;
