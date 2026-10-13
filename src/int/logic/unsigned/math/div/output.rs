//! Compile-time output policies and division certificates.
//!
//! Ordinary division returns `()`. A truncated prefix may instead prove that
//! its remainder exceeds the quotient without materializing that remainder.
//! Known divisibility applies only to quotient-only complete operands.

use super::LIMB_BITS;

/// A completed division or an operand shape requiring reusable scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivisionAttempt {
    /// Every requested output has been written.
    Complete,
    /// The operands require a different tier or a larger workspace.
    NeedsScratch,
}

/// The remainder information retained by a truncated quotient prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemainderCertificate {
    /// The exact prefix remainder has been written.
    Written,
    /// The unnormalized prefix remainder exceeds the quotient.
    ExceedsQuotient,
}

/// Exactness of an upper quotient estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum QuotientAccuracy {
    /// The returned digits equal the exact floor quotient.
    Exact = 0,
    /// The estimate is at least exact and its excess is below `2*Limb::BITS`.
    UpperBound = 2 * LIMB_BITS,
}

/// Output and certificate policy specialized at compilation.
pub trait DivisionMode {
    /// Whether the quotient is requested.
    const WRITE_QUOTIENT: bool;
    /// Whether the remainder is requested.
    const WRITE_REMAINDER: bool;
    /// Whether a prefix may omit a remainder proved larger than its quotient.
    const CERTIFY_REMAINDER: bool;
    /// Whether complete operands are known to divide exactly.
    const KNOWN_EXACT: bool;
    /// Ordinary completion or a truncated-prefix certificate.
    type Outcome: PartialEq;
    /// Complete outputs, retaining known divisibility of the complete input.
    type Primitive: CompleteDivisionMode;
    /// Quotient-producing policy for the short-division kernel.
    /// Its completion has the same certificate type as this policy.
    type Quotient: QuotientDivisionMode<Outcome = Self::Outcome>;
    /// Completion with every requested remainder written.
    const WRITTEN: Self::Outcome;
    /// Completion with a proved remainder bound.
    const CERTIFIED: Self::Outcome;
}

/// A complete division whose outputs require no prefix certificate.
pub trait CompleteDivisionMode: DivisionMode<Outcome = ()> {}

/// A quotient-only output or a quotient with a truncated-prefix certificate.
/// Complete quotient-and-remainder output uses the full division kernel.
pub trait QuotientDivisionMode: DivisionMode {}

/// Quotient-only division, optionally with known divisibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotientOnly<const EXACT: bool = false>;

/// Remainder-only division.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemainderOnly;

/// Complete quotient and remainder division.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotientAndRemainder;

/// Quotient and remainder certificate for a one-guard truncated prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncatedPrefix;

impl<const EXACT: bool> DivisionMode for QuotientOnly<EXACT> {
    const WRITE_QUOTIENT: bool = true;
    const WRITE_REMAINDER: bool = false;
    const CERTIFY_REMAINDER: bool = false;
    const KNOWN_EXACT: bool = EXACT;
    type Outcome = ();
    type Primitive = Self;
    type Quotient = Self;
    const WRITTEN: () = ();
    const CERTIFIED: () = ();
}

impl<const EXACT: bool> CompleteDivisionMode for QuotientOnly<EXACT> {}

impl<const EXACT: bool> QuotientDivisionMode for QuotientOnly<EXACT> {}

impl DivisionMode for RemainderOnly {
    const WRITE_QUOTIENT: bool = false;
    const WRITE_REMAINDER: bool = true;
    const CERTIFY_REMAINDER: bool = false;
    const KNOWN_EXACT: bool = false;
    type Outcome = ();
    type Primitive = Self;
    type Quotient = QuotientOnly;
    const WRITTEN: () = ();
    const CERTIFIED: () = ();
}

impl CompleteDivisionMode for RemainderOnly {}

impl DivisionMode for QuotientAndRemainder {
    const WRITE_QUOTIENT: bool = true;
    const WRITE_REMAINDER: bool = true;
    const CERTIFY_REMAINDER: bool = false;
    const KNOWN_EXACT: bool = false;
    type Outcome = ();
    type Primitive = Self;
    type Quotient = QuotientOnly;
    const WRITTEN: () = ();
    const CERTIFIED: () = ();
}

impl CompleteDivisionMode for QuotientAndRemainder {}

impl DivisionMode for TruncatedPrefix {
    const WRITE_QUOTIENT: bool = true;
    const WRITE_REMAINDER: bool = true;
    const CERTIFY_REMAINDER: bool = true;
    const KNOWN_EXACT: bool = false;
    type Outcome = RemainderCertificate;
    type Primitive = QuotientAndRemainder;
    type Quotient = Self;
    const WRITTEN: RemainderCertificate = RemainderCertificate::Written;
    const CERTIFIED: RemainderCertificate = RemainderCertificate::ExceedsQuotient;
}

impl QuotientDivisionMode for TruncatedPrefix {}
