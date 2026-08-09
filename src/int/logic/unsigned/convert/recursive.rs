//! Recursive Barrett divide-and-conquer radix formatting.

#![allow(
    unsafe_code,
    reason = "Domain chains and recursion frames are indexed below proved bounds (index < domains.len(), one frame per divide node), unwrapped conversions are proved in-range, and leaf bytes are ASCII."
)]

use core::{
    cmp::Ordering,
    fmt::{Result as FmtResult, Write},
    str::from_utf8_unchecked,
};

use alloc::vec::Vec;

use super::{
    BarrettDomain, BarrettScratch, InternalMpUint, Limb, MulScratch, RadixParameters,
    byte_from_digit, div_rem_small, write_decimal_chunks,
};

/// Reusable working state for recursive radix formatting.
///
/// Holds the cached Barrett domains per radix plus the digit scratch, leaf
/// value, multiplication and Barrett scratches, and recursion frames that the
/// recursive formatter would otherwise allocate on every call. All buffers
/// keep their capacities across formatting calls, so repeated formatting of
/// similar-sized values does not reallocate.
#[derive(Debug)]
pub struct FormatCache {
    /// Cached domains for each radix from 2..=36.
    /// Index `r - 2` corresponds to radix `r`.
    domains: [Vec<BarrettDomain>; 35],
    /// Reverse-order digit scratch shared by the recursion leaves.
    digit_scratch: Vec<u8>,
    /// Multiplication scratch shared by the Barrett quotients.
    mul_scratch: MulScratch,
    /// Barrett reduction scratch shared by every divide node.
    barrett_scratch: BarrettScratch,
    /// Base-case value reused by the recursion leaves.
    leaf_value: InternalMpUint,
    /// One quotient/remainder frame per active recursion depth.
    frames: Vec<FormatFrame>,
}

/// Quotient and remainder of one recursive divide node, reused across
/// formatting calls and recursion levels so deep formatting never allocates
/// integer objects per node.
#[derive(Debug)]
struct FormatFrame {
    quotient: InternalMpUint,
    remainder: InternalMpUint,
}

impl Default for FormatFrame {
    fn default() -> Self {
        Self {
            quotient: InternalMpUint::from_limb(0),
            remainder: InternalMpUint::from_limb(0),
        }
    }
}

impl Default for FormatCache {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatCache {
    /// Creates a new, empty format cache.
    #[must_use]
    pub fn new() -> Self {
        const EMPTY_VEC: Vec<BarrettDomain> = Vec::new();
        Self {
            domains: [EMPTY_VEC; 35],
            digit_scratch: Vec::new(),
            mul_scratch: MulScratch::default(),
            barrett_scratch: BarrettScratch::default(),
            leaf_value: InternalMpUint::from_limb(0),
            frames: Vec::new(),
        }
    }
}

impl InternalMpUint {
    /// Formats the integer in the given radix with the recursive Barrett
    /// divide-and-conquer path, reusing the working state in `cache`.
    ///
    /// # Preconditions
    ///
    /// The caller must have validated that `radix` is a non-power-of-two
    /// radix in `2..=36`; the public dispatch paths ([`Self::format_radix_writer`]
    /// and [`Self::to_string_radix`]) do so before calling. The digit scratch,
    /// leaf value, multiplication and Barrett scratches, and recursion frames
    /// in `cache` are reused across calls and keep their capacities.
    pub fn format_recursive_writer_with_cache<W: Write + ?Sized>(
        &self,
        radix: u32,
        w: &mut W,
        cache: &mut FormatCache,
    ) -> FmtResult {
        let RadixParameters {
            max_digits,
            max_power,
        } = RadixParameters::for_limb(radix);

        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "radix is checked to be in 2..=36 and therefore fits in Limb"
        )]
        let radix_limb = radix as Limb;

        let domains = get_domains(&mut cache.domains, radix, max_power, self);
        let frames = &mut cache.frames;
        if frames.len() < domains.len() {
            frames.resize_with(domains.len(), FormatFrame::default);
        }

        format_recursive_into(
            self,
            domains,
            radix_limb,
            max_digits,
            None,
            w,
            &mut cache.digit_scratch,
            &mut cache.mul_scratch,
            &mut cache.barrett_scratch,
            &mut cache.leaf_value,
            frames,
        )
    }
}

/// Gets or builds domains for the given radix, up to the required depth to
/// format `value`.
///
/// The per-radix chain grows by squaring the previous modulus, so repeated
/// formatting of larger values reuses every already-built domain.
fn get_domains<'domains>(
    domains: &'domains mut [Vec<BarrettDomain>; 35],
    radix: u32,
    max_power: Limb,
    value: &InternalMpUint,
) -> &'domains [BarrettDomain] {
    #[allow(clippy::as_conversions, reason = "radix is in 2..=36")]
    // SAFETY: radix is bounded in 2..=36, so `radix as usize - 2` is in 0..=34.
    let entry = unsafe { domains.get_unchecked_mut((radix as usize).wrapping_sub(2)) };
    if entry.is_empty() {
        entry.push(BarrettDomain::new(&InternalMpUint::from_limb(max_power)));
    }

    // SAFETY: We just ensured the vec is not empty.
    while value.cmp(&unsafe { entry.last().unwrap_unchecked() }.modulus) == Ordering::Greater {
        // SAFETY: We just ensured the vec is not empty.
        let last = unsafe { entry.last().unwrap_unchecked() };
        entry.push(BarrettDomain::new(&last.modulus.mul(&last.modulus)));
    }

    let mut active_domains = 0_usize;
    for domain in entry.iter() {
        active_domains = active_domains.wrapping_add(1);
        if value.cmp(&domain.modulus) != Ordering::Greater {
            break;
        }
    }
    // SAFETY: active_domains is guaranteed to be <= entry.len()
    unsafe { entry.get_unchecked(..active_domains) }
}

#[allow(
    clippy::too_many_arguments,
    reason = "divide and conquer recursion requires passing multiple scratches, frames, and thresholds"
)]
fn format_recursive_into<W: Write + ?Sized>(
    value: &InternalMpUint,
    domains: &[BarrettDomain],
    radix: Limb,
    max_digits: usize,
    pad_to: Option<usize>,
    w: &mut W,
    scratch: &mut Vec<u8>,
    mul_scratch: &mut MulScratch,
    barrett_scratch: &mut BarrettScratch,
    leaf_value: &mut InternalMpUint,
    frames: &mut [FormatFrame],
) -> FmtResult {
    if domains.is_empty() {
        scratch.clear();
        leaf_value.clone_from(value);
        if radix == 10 {
            // Decimal blocks split on powers of ten, so each base-case value
            // is at most `DECIMAL_CHUNK_DIGITS` digits wide; one division by
            // the decimal chunk divisor replaces the per-digit loop below.
            if !leaf_value.is_zero() {
                write_decimal_chunks(leaf_value, scratch);
            }
        } else {
            while !leaf_value.is_zero() {
                let remainder = div_rem_small(leaf_value, radix);
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "remainder is less than radix, which is at most 36"
                )]
                scratch.push(byte_from_digit(remainder as u8));
            }
        }
        if pad_to.is_none() && scratch.is_empty() {
            scratch.push(b'0');
        }
        if let Some(pad) = pad_to {
            while scratch.len() < pad {
                scratch.push(b'0');
            }
        }
        scratch.reverse();
        // SAFETY: `write_decimal_chunks` and `byte_from_digit` append only
        // valid ASCII digits and lowercase letters.
        return w.write_str(unsafe { from_utf8_unchecked(scratch) });
    }

    let index = domains.len().wrapping_sub(1);
    // `domains` is non-empty in this branch, so `index < domains.len()`.
    // SAFETY: the preceding branch proves the index is in bounds.
    let domain = unsafe { domains.get_unchecked(index) };
    // `domains[0]` has `max_digits` digits in radix, and every next entry squares the
    // previous one. Thus `domains[index]` has more than `2^index` significant
    // bits. A materialized integer has a `usize` limb/bit length, proving
    // `index < usize::BITS <= 64 <= u32::MAX` on every supported target.
    // SAFETY: the proved range makes this conversion infallible.
    let shift = unsafe { u32::try_from(index).unwrap_unchecked() };
    let block_digits = max_digits.wrapping_mul(1_usize.wrapping_shl(shift));

    if value.cmp(&domain.modulus) == Ordering::Less {
        // The value fits entirely in a narrower slice of this block, so it is
        // passed down unchanged. Propagating `pad_to` (instead of forcing the
        // child to a full `block_digits`) makes the child emit exactly the
        // requested width, so the parent never needs to shift memory to insert
        // leading zeros: the leaf pads to its received width directly.
        return format_recursive_into(
            value,
            // SAFETY: `index < domains.len()` was proved above.
            unsafe { domains.get_unchecked(..index) },
            radix,
            max_digits,
            pad_to,
            w,
            scratch,
            mul_scratch,
            barrett_scratch,
            leaf_value,
            frames,
        );
    }

    // One frame serves each recursion depth: the top-level caller sized
    // `frames` to the initial domain chain length, every divide node consumes
    // exactly one frame and passes the tail on to both children, and the
    // "value fits narrower" branch consumes none, so `depth + domains.len()`
    // stays constant and every divide node still finds a frame in front.
    // SAFETY: the frame budget proof above guarantees at least one frame
    // remains at every divide node, so the split always succeeds.
    let (frame, rest) = unsafe { frames.split_first_mut().unwrap_unchecked() };
    domain.div_rem_into_with_barrett_scratch(
        value,
        &mut frame.quotient,
        &mut frame.remainder,
        mul_scratch,
        barrett_scratch,
    );
    let quotient_pad = pad_to.map(|pad| pad.wrapping_sub(block_digits));
    // SAFETY: `index < domains.len()` was proved above.
    let lower_domains = unsafe { domains.get_unchecked(..index) };
    format_recursive_into(
        &frame.quotient,
        lower_domains,
        radix,
        max_digits,
        quotient_pad,
        w,
        scratch,
        mul_scratch,
        barrett_scratch,
        leaf_value,
        rest,
    )?;
    format_recursive_into(
        &frame.remainder,
        lower_domains,
        radix,
        max_digits,
        Some(block_digits),
        w,
        scratch,
        mul_scratch,
        barrett_scratch,
        leaf_value,
        rest,
    )
}
