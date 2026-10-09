//! Function-pointer contracts for selected limb kernels, with base `B = 2^LIMB_BITS`.

use super::Limb;

/// Computes `dst_new + carry * B^len = dst_old + src * scalar`.
pub type AddMulKernel = unsafe fn(*mut Limb, *const Limb, usize, Limb) -> Limb;

/// Computes `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
pub type SubMulKernel = unsafe fn(*mut Limb, *const Limb, usize, Limb) -> (Limb, Limb);

with_direct_basecase_components! {
    /// Accumulates `src * s0 + src * s1 * B` into `len + 1` destination limbs.
    ///
    /// Returns the separate carries of the low and shifted rows.
    pub type AddMul2Kernel = unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb) -> (Limb, Limb);

    /// Writes the complete `len + 2` limbs of `src * (s0 + s1 * B)`.
    pub type Mul2Kernel = unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb);
}

/// Computes `sum_new + c * B^len = sum_old + src` and
/// `diff_new - b * B^len = sum_old - src`.
#[cfg(not(target_pointer_width = "16"))]
pub type AddSubFromKernel = unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb);

/// Computes `(T + x_i * Y + m * N) / B`, where `m = (T_0 + x_i * Y_0) * mu mod B`.
pub type MontyKernel = unsafe fn(*mut Limb, *const Limb, *const Limb, usize, Limb, Limb) -> Limb;

/// Left-shifts an initialized limb span and returns its shifted-out high bits.
pub type LshiftKernel = unsafe fn(*mut Limb, usize, u32) -> Limb;

/// Left-shifts a source into a disjoint destination and returns its shifted-out high bits.
pub type LshiftIntoKernel = unsafe fn(*mut Limb, *const Limb, usize, u32) -> Limb;

/// Left-shifts `limbs[0..len]` into `limbs[offset..offset + len]`.
#[cfg(not(target_pointer_width = "16"))]
pub type LshiftOverlappingKernel = unsafe fn(*mut Limb, usize, usize, u32) -> Limb;

/// Right-shifts an initialized limb span and returns its shifted-out low bits.
pub type RshiftKernel = unsafe fn(*mut Limb, usize, u32) -> Limb;

/// Right-shifts a source into a disjoint destination and returns its shifted-out low bits.
pub type RshiftIntoKernel = unsafe fn(*mut Limb, *const Limb, usize, u32) -> Limb;

/// Computes `dst_new - b * B^len = dst_old - (src >> (Limb::BITS - shift)) - b_in`.
#[cfg(not(target_pointer_width = "16"))]
pub type SubShiftedHighKernel = unsafe fn(*mut Limb, *const Limb, usize, u32, Limb) -> Limb;
