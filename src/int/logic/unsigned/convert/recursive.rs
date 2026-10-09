//! Recursive Barrett divide-and-conquer radix formatting.
//!
//! Reference: R. P. Brent and P. Zimmermann, "Modern Computer Arithmetic",
//! Cambridge University Press, 2011, Section 1.7.2 (subquadratic base conversion).

#![expect(
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
    BarrettDomain, BarrettScratch, Convert, Division, InternalMpUint, Limb, MulScratch,
    RADIX_CHUNK_RECIPROCALS, RadixParameters,
};

/// Reusable working state for recursive radix formatting.
///
/// Holds the cached Barrett domains per radix plus the digit scratch,
/// multiplication and Barrett scratches, and recursion frames that the
/// recursive formatter would otherwise allocate on every call. All buffers
/// keep their capacities across formatting calls, so repeated formatting of
/// similar-sized values does not reallocate.
#[derive(Debug)]
pub struct FormatCache {
    /// Cached domains for each radix from 2..=36.
    /// Index `r - 2` corresponds to radix `r`.
    domains: [RadixDomains; 35],
    /// Forward-order digit scratch shared by the recursion leaves.
    digit_scratch: Vec<u8>,
    /// Multiplication scratch shared by the Barrett quotients.
    mul_scratch: MulScratch,
    /// Barrett reduction scratch shared by every divide node.
    barrett_scratch: BarrettScratch,
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

/// Constructed divisors and the next power, whose reciprocal is deferred
/// until an input actually reaches it.
#[derive(Debug)]
struct RadixDomains {
    ready: Vec<BarrettDomain>,
    next_power: InternalMpUint,
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
        Self {
            domains: [const {
                RadixDomains {
                    ready: Vec::new(),
                    next_power: InternalMpUint::zero(),
                }
            }; 35],
            digit_scratch: Vec::new(),
            mul_scratch: MulScratch::default(),
            barrett_scratch: BarrettScratch::default(),
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
    /// radix in `2..=36`; `format_radix_writer` and [`Self::to_string_radix`]
    /// validate that domain before calling. The caller must
    /// also have validated that the significant bit count fits `usize`.
    /// The digit scratch, multiplication and Barrett scratches, and recursion frames
    /// in `cache` are reused across calls and keep their capacities.
    pub fn format_recursive_writer_with_cache<W: Write + ?Sized>(
        &self,
        radix: u32,
        w: &mut W,
        cache: &mut FormatCache,
    ) -> FmtResult {
        debug_assert!(
            (2..=36).contains(&radix) && !radix.is_power_of_two(),
            "recursive formatting requires a validated non-power-of-two radix"
        );
        debug_assert!(
            self.is_zero() || self.significant_bits() > 0,
            "recursive formatting requires a representable significant bit count"
        );
        let parameters = RadixParameters::for_limb(radix);

        #[expect(
            clippy::as_conversions,
            reason = "radix is checked to be in 2..=36 and therefore fits in Limb"
        )]
        let radix_limb = radix as Limb;

        let domains = get_domains(&mut cache.domains, radix, parameters.max_power, self);
        // Active domains satisfy radix^(max_digits * 2^(index+1)) <= self.
        // Since radix >= 3, every block width is below the caller-validated
        // significant bit count and fits usize throughout the recursion.
        let frames = &mut cache.frames;
        if frames.len() < domains.len() {
            frames.resize_with(domains.len(), FormatFrame::default);
        }

        format_recursive_into(
            self,
            domains,
            radix_limb,
            parameters,
            None,
            w,
            &mut cache.digit_scratch,
            &mut cache.mul_scratch,
            &mut cache.barrett_scratch,
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
    domains: &'domains mut [RadixDomains; 35],
    radix: u32,
    max_power: Limb,
    value: &InternalMpUint,
) -> &'domains [BarrettDomain] {
    #[expect(clippy::as_conversions, reason = "radix is in 2..=36")]
    // SAFETY: radix is bounded in 2..=36, so `radix as usize - 2` is in 0..=34.
    let entry = unsafe { domains.get_unchecked_mut((radix as usize).unchecked_sub(2)) };
    if entry.ready.last().is_none_or(|last| value > &last.modulus) {
        if entry.next_power.is_zero() {
            entry.next_power = entry.ready.last().map_or_else(
                || InternalMpUint::from_limb(max_power).square(),
                |last| last.modulus.square(),
            );
        }
        while value >= &entry.next_power {
            entry.ready.push(BarrettDomain::new(&entry.next_power));
            if value == &entry.next_power {
                // Equality needs this divisor, but no larger power. A later
                // larger input computes the successor from the cached modulus.
                entry.next_power.clear();
                break;
            }
            entry.next_power = entry.next_power.square();
        }
    }
    let active_domains = entry
        .ready
        .partition_point(|domain| &domain.modulus <= value);
    // SAFETY: partition_point returns a boundary within ready.
    unsafe { entry.ready.get_unchecked(..active_domains) }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Recursive formatting shares radix parameters, padding, writer, frames, and scratch buffers across nodes"
)]
fn format_recursive_into<W: Write + ?Sized>(
    value: &InternalMpUint,
    domains: &[BarrettDomain],
    radix: Limb,
    parameters: RadixParameters,
    pad_to: Option<usize>,
    w: &mut W,
    scratch: &mut Vec<u8>,
    mul_scratch: &mut MulScratch,
    barrett_scratch: &mut BarrettScratch,
    frames: &mut [FormatFrame],
) -> FmtResult {
    if domains.is_empty() {
        // The first domain is B^2, where B = radix^max_digits < 2^W.
        // Children and skipped levels preserve value < B^2, so a leaf
        // occupies at most two native limbs.
        return format_native_blocks(value, radix, parameters, pad_to, w, scratch);
    }

    // SAFETY: the preceding `if domains.is_empty()` guard returned early, proving
    // `domains.len() >= 1`. `split_last()` is guaranteed to return `Some`.
    let (domain, lower_domains) = unsafe { domains.split_last().unwrap_unchecked() };
    let index = lower_domains.len();
    // SAFETY: the active exponent max_digits * 2^(index+1) is below the
    // validated significant bit count. Thus index+1 < usize::BITS <= 64
    // and fits u32 across 16-, 32-, and 64-bit targets.
    let shift = unsafe { u32::try_from(index.unchecked_add(1)).unwrap_unchecked() };
    // SAFETY: radix^(max_digits * 2^(index+1)) <= the original magnitude, whose
    // significant bit count fits usize. The complete exponent therefore
    // fits usize and this in-range shift cannot discard significant bits.
    let block_digits = unsafe { parameters.max_digits.unchecked_shl(shift) };

    if value.cmp(&domain.modulus) == Ordering::Less {
        // The value fits entirely in a narrower slice of this block, so it is
        // passed down unchanged. Propagating `pad_to` (instead of forcing the
        // child to a full `block_digits`) makes the child emit exactly the
        // requested width, so the parent never needs to shift memory to insert
        // leading zeros: the leaf pads to its received width directly.
        return format_recursive_into(
            value,
            lower_domains,
            radix,
            parameters,
            pad_to,
            w,
            scratch,
            mul_scratch,
            barrett_scratch,
            frames,
        );
    }

    let quotient_pad = pad_to.map(|pad| {
        debug_assert!(
            pad >= block_digits,
            "inherited padding covers the divided block"
        );
        // SAFETY: the inherited radix block contains this division's modulus.
        unsafe { pad.unchecked_sub(block_digits) }
    });

    // One frame serves each recursion depth: the top-level caller sized
    // `frames` to the initial domain chain length, every divide node consumes
    // exactly one frame and passes the tail on to both children, and the
    // "value fits narrower" branch consumes none. Thus frames.len() remains
    // at least domains.len(), and each divide node has an available frame.
    // SAFETY: the frame budget proof above guarantees at least one frame
    // remains at every divide node, so the split always succeeds.
    let (frame, rest) = unsafe { frames.split_first_mut().unwrap_unchecked() };
    debug_assert!(
        value.limbs().len() <= domain.k.saturating_mul(2),
        "the active radix domain bounds its input below B^(2k)"
    );
    // SAFETY: the comparison above establishes value >= D. The root's
    // next power is D^2 > value. Each divide produces q,r < D, while the
    // lower domain is sqrt(D); skipped levels also preserve that bound.
    // Thus value < D^2 < B^(2k). Frames and scratch are distinct owners.
    unsafe {
        domain.div_rem_bounded_unchecked(
            value,
            &mut frame.quotient,
            &mut frame.remainder,
            mul_scratch,
            barrett_scratch,
        );
    }
    debug_assert!(
        frame.quotient < domain.modulus && frame.remainder < domain.modulus,
        "both children are below the divided radix power"
    );
    format_recursive_into(
        &frame.quotient,
        lower_domains,
        radix,
        parameters,
        quotient_pad,
        w,
        scratch,
        mul_scratch,
        barrett_scratch,
        rest,
    )?;
    format_recursive_into(
        &frame.remainder,
        lower_domains,
        radix,
        parameters,
        Some(block_digits),
        w,
        scratch,
        mul_scratch,
        barrett_scratch,
        rest,
    )
}

/// Formats a value below the square of the native radix block divisor.
fn format_native_blocks<W: Write + ?Sized>(
    value: &InternalMpUint,
    radix: Limb,
    parameters: RadixParameters,
    pad_to: Option<usize>,
    w: &mut W,
    scratch: &mut Vec<u8>,
) -> FmtResult {
    let limbs = value.limbs();
    debug_assert!(
        limbs.len() <= 2,
        "a value below B^2 occupies at most two limbs"
    );
    if limbs.len() <= 1 {
        // SAFETY: the length check proves zero or one initialized native limb.
        let scalar = unsafe { value.to_usize().unwrap_unchecked() };
        if scalar < parameters.max_power {
            return write_radix_limb(scalar, radix, parameters, pad_to, w, scratch);
        }
    }
    // SAFETY: the preceding branch returned for every value below B > 0;
    // the entering nonzero magnitude therefore has an initialized low limb.
    let low = unsafe { *limbs.get_unchecked(0) };
    let high = if limbs.len() == 2 {
        // SAFETY: this branch proves the second limb exists.
        unsafe { *limbs.get_unchecked(1) }
    } else {
        0
    };
    let shift = parameters.max_power.leading_zeros();
    // SAFETY: the native radix power is nonzero and its normalization fits.
    let divisor = unsafe { parameters.max_power.unchecked_shl(shift) };
    // SAFETY: the validated radix selects its normalized reciprocal.
    let reciprocal = unsafe { *RADIX_CHUNK_RECIPROCALS.get_unchecked(radix) };
    let (quotient, normalized_remainder) = if shift == 0 {
        Division::divrem_2by1_reciprocal(high, low, divisor, reciprocal)
    } else {
        // SAFETY: 0 < shift < W bounds both shifts. Since high < B, the
        // normalized high limb is below B << shift and fits Limb.
        let (normalized_high, normalized_low) = unsafe {
            let complement = Limb::BITS.unchecked_sub(shift);
            (
                high.unchecked_shl(shift) | (low >> complement),
                low.unchecked_shl(shift),
            )
        };
        Division::divrem_2by1_reciprocal(normalized_high, normalized_low, divisor, reciprocal)
    };
    let remainder = normalized_remainder >> shift;
    debug_assert!(
        quotient < parameters.max_power && remainder < parameters.max_power,
        "division below B^2 produces two native radix blocks"
    );
    let quotient_pad = pad_to.map(|pad| {
        debug_assert!(
            pad > parameters.max_digits,
            "the inherited block covers a nonzero quotient"
        );
        // SAFETY: inherited padding contains both native radix blocks.
        unsafe { pad.unchecked_sub(parameters.max_digits) }
    });
    write_radix_limb(quotient, radix, parameters, quotient_pad, w, scratch)?;
    write_radix_limb(
        remainder,
        radix,
        parameters,
        Some(parameters.max_digits),
        w,
        scratch,
    )
}

/// Writes a native radix block with its inherited leading zero padding.
fn write_radix_limb<W: Write + ?Sized>(
    scalar: Limb,
    radix: Limb,
    parameters: RadixParameters,
    pad_to: Option<usize>,
    w: &mut W,
    scratch: &mut Vec<u8>,
) -> FmtResult {
    scratch.clear();
    if let Some(pad) = pad_to {
        // Inherited blocks contain this value and have widths that are positive
        // multiples of max_digits. A native block therefore fits the full pad.
        debug_assert!(
            parameters.max_digits <= pad,
            "radix block padding covers its native digit width"
        );
        // SAFETY: inherited padding contains at least one complete native block.
        let zeros = unsafe { pad.unchecked_sub(parameters.max_digits) };
        if zeros != 0 {
            scratch.reserve(pad);
            scratch.resize(zeros, b'0');
        }
    }
    Convert::write_radix_chunk::<false>(scalar, radix, parameters, pad_to.is_some(), scratch);
    // SAFETY: native digit extraction and padding append only ASCII bytes.
    w.write_str(unsafe { from_utf8_unchecked(scratch) })
}
