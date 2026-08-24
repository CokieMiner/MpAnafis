//! 50-bit floating-point prime field parameters for Harvey NTT.

/// 50-bit NTT prime: `p1 = 1108307720798209`.
pub const FLOAT_PRIME_1: f64 = 1_108_307_720_798_209.0;
/// Integer representation of `p1`.
pub const FLOAT_PRIME_1_INT: u64 = 1_108_307_720_798_209;
/// Inverse of `p1`: `1.0 / p1`.
pub const FLOAT_PINV_1: f64 = 1.0 / FLOAT_PRIME_1;
/// Primitive root modulo `p1`.
pub const FLOAT_ROOT_1: u64 = 11;

/// 50-bit NTT prime: `p2 = 1086317488242689`.
pub const FLOAT_PRIME_2: f64 = 1_086_317_488_242_689.0;
/// Integer representation of `p2`.
pub const FLOAT_PRIME_2_INT: u64 = 1_086_317_488_242_689;
/// Inverse of `p2`: `1.0 / p2`.
pub const FLOAT_PINV_2: f64 = 1.0 / FLOAT_PRIME_2;
/// Primitive root modulo `p2`.
pub const FLOAT_ROOT_2: u64 = 3;

/// 50-bit NTT prime: `p3 = 910395627798529`.
pub const FLOAT_PRIME_3: f64 = 910_395_627_798_529.0;
/// Integer representation of `p3`.
pub const FLOAT_PRIME_3_INT: u64 = 910_395_627_798_529;
/// Inverse of `p3`: `1.0 / p3`.
pub const FLOAT_PINV_3: f64 = 1.0 / FLOAT_PRIME_3;
/// Primitive root modulo `p3`.
pub const FLOAT_ROOT_3: u64 = 7;
