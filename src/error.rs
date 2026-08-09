#[cfg(feature = "std")]
use core::error::Error;
use core::fmt::{Display, Formatter, Result as FmtResult};
/// The central error type for the arbitrary precision library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MpError {
    /// Operation would exceed the maximum permitted precision bound.
    Overflow,
    /// Unsigned subtraction below zero, or precision underflow.
    Underflow,
    /// Attempted to divide by zero.
    DivisionByZero,
    /// Attempted to calculate an even root of a negative number.
    NegativeRoot,
    /// The algorithm does not support even moduli.
    EvenModulusUnsupported,
    /// Modulus cannot be zero in modular arithmetic.
    ModulusZero,
    /// The number has no modular inverse.
    NoInverse,
    /// The modulus has no primitive root.
    NoPrimitiveRoot,
    /// The values are not coprime.
    NotCoprime,
    /// An operation mathematically requires an explicit width, but unlimited precision was active.
    WidthRequired,
    /// The operation requires an explicit precision setting.
    PrecisionRequired,
    /// The requested bit shift exceeds limits.
    ShiftTooLarge,
    /// An allocation is required, but the library is in `no_alloc` mode.
    AllocationRequired,
    /// The provided radix is not supported.
    InvalidRadix,
    /// A string parsing operation encountered an invalid digit for the radix.
    InvalidDigit,
    /// Operation requires integer factorization but the functionality is unavailable or failed.
    FactorizationRequired,
    /// Memory representation is not in canonical form.
    NonCanonical,
    /// Precision configurations do not match where required.
    PrecisionMismatch,
    /// Bounded target precision is insufficient to store the magnitude.
    PrecisionExceeded,
    /// The input provided was empty when data was required.
    EmptyInput,
    /// A positive input was required, but a non-positive one was provided.
    NonPositiveInput,
    /// A non-negative input was required, but a negative one was provided.
    NegativeInput,
    /// The provided modulus is invalid for the operation.
    InvalidModulus,
    /// The group is not cyclic.
    NonCyclicGroup,
    /// The element is not in the generated subgroup.
    NotInGeneratedSubgroup,
    /// Generic input error for parsing/construction failures not covered above.
    InvalidInput,
    /// Conversion between numerical types would result in a loss of precision or information.
    IntegerConversionLoss,
    /// An empty slice was provided where a non-empty one was required.
    EmptySlice,
}

impl Display for MpError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match *self {
            Self::Overflow => write!(f, "overflow"),
            Self::Underflow => write!(f, "underflow"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::NegativeRoot => write!(f, "negative root"),
            Self::EvenModulusUnsupported => write!(f, "even modulus unsupported"),
            Self::ModulusZero => write!(f, "modulus is zero"),
            Self::NoInverse => write!(f, "no modular inverse"),
            Self::NoPrimitiveRoot => write!(f, "no primitive root"),
            Self::NotCoprime => write!(f, "not coprime"),
            Self::WidthRequired => write!(f, "explicit width required for this operation"),
            Self::PrecisionRequired => write!(f, "explicit precision required"),
            Self::ShiftTooLarge => write!(f, "shift too large"),
            Self::AllocationRequired => write!(f, "allocation required in no_alloc mode"),
            Self::InvalidRadix => write!(f, "invalid radix"),
            Self::InvalidDigit => write!(f, "invalid digit"),
            Self::FactorizationRequired => write!(f, "factorization required"),
            Self::NonCanonical => write!(f, "non-canonical representation"),
            Self::PrecisionMismatch => write!(f, "precision mismatch"),
            Self::PrecisionExceeded => write!(f, "precision exceeded"),
            Self::EmptyInput => write!(f, "empty input"),
            Self::NonPositiveInput => write!(f, "non-positive input"),
            Self::NegativeInput => write!(f, "negative input"),
            Self::InvalidModulus => write!(f, "invalid modulus"),
            Self::NonCyclicGroup => write!(f, "non-cyclic group"),
            Self::NotInGeneratedSubgroup => write!(f, "not in generated subgroup"),
            Self::InvalidInput => write!(f, "invalid input"),
            Self::IntegerConversionLoss => write!(f, "integer conversion loss"),
            Self::EmptySlice => write!(f, "empty slice"),
        }
    }
}

#[cfg(feature = "std")]
impl Error for MpError {}

/// Error type for parsing an `MpInt` from a string or bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseMpIntError {
    pub(crate) kind: ParseMpIntErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseMpIntErrorKind {
    Empty,
    InvalidDigit,
    InvalidRadix,
    TooLarge,
}

impl Display for ParseMpIntError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self.kind {
            ParseMpIntErrorKind::Empty => write!(f, "cannot parse integer from empty string"),
            ParseMpIntErrorKind::InvalidDigit => write!(f, "invalid digit found in string"),
            ParseMpIntErrorKind::InvalidRadix => write!(f, "invalid radix"),
            ParseMpIntErrorKind::TooLarge => write!(f, "value too large for parsing"),
        }
    }
}

#[cfg(feature = "std")]
impl Error for ParseMpIntError {}

/// Error type for parsing an `MpUint` from a string or bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseMpUintError {
    pub(crate) kind: ParseMpUintErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseMpUintErrorKind {
    Empty,
    InvalidRadix,
    InvalidDigit,
    Negative,
    TooLarge,
}

impl Display for ParseMpUintError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self.kind {
            ParseMpUintErrorKind::Empty => write!(f, "cannot parse integer from empty string"),
            ParseMpUintErrorKind::InvalidDigit => write!(f, "invalid digit found in string"),
            ParseMpUintErrorKind::InvalidRadix => write!(f, "invalid radix"),
            ParseMpUintErrorKind::Negative => {
                write!(f, "cannot parse unsigned integer from negative value")
            }
            ParseMpUintErrorKind::TooLarge => write!(f, "value too large for parsing"),
        }
    }
}

#[cfg(feature = "std")]
impl Error for ParseMpUintError {}
