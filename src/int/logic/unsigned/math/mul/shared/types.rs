//! Shared evaluation namespace and scalar-product kernel type.

use super::Limb;

/// Namespace for guarded evaluation, interpolation, and exact division.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SharedEval;

/// Adds `scalar*src` to `dst` and returns the escaping limb.
///
/// Evaluators select this architecture backend once and reuse it across points.
///
/// # Safety
/// Both pointer spans contain `len` initialized limbs. The destination is
/// writable, and the spans are disjoint or start at the same address.
pub type AddMulKernel = unsafe fn(*mut Limb, *const Limb, usize, Limb) -> Limb;
