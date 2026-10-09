//! Conventional multiplication and squaring workspace recurrences.
//!
//! Each tier's `*_mul_scratch_len` and `*_sqr_scratch_len` size its explicit
//! driver, including required shape fallbacks and recursive children. The
//! `*_dispatch_mul_scratch_len` and `*_dispatch_sqr_scratch_len` variants also
//! apply the Toom-3 crossover. Karatsuba child sizes apply their schoolbook
//! crossover directly at each recurrence.

#![expect(
    unsafe_code,
    reason = "Karatsuba halves and admitted Toom chunk offsets are bounded by their widths; arbitrary-width workspace totals retain checked validation"
)]

use core::cmp::{max, min};
#[cfg(target_has_atomic = "ptr")]
use core::sync::atomic::{AtomicUsize, Ordering};

use super::{
    KARATSUBA_THRESHOLD, Karatsuba, MulShape, Multiplication, SQR_KARATSUBA_THRESHOLD,
    SQR_TOOM_COOK_THRESHOLD, TOOM_COOK_THRESHOLD, TierCeiling, Toom6, Toom8, Widths,
};

/// Widths covered by the balanced sizing memo tables.
#[cfg(target_has_atomic = "ptr")]
const BALANCED_MEMO_LEN: usize = 512;

/// Memo of balanced-width scratch lengths for one tier and operation.
///
/// Each length depends only on its width and build-time crossovers. Concurrent
/// writers store the same value, so relaxed atomics suffice. Zero marks an
/// empty slot; every memoized layout has positive length.
#[cfg(target_has_atomic = "ptr")]
struct BalancedMemo([AtomicUsize; BALANCED_MEMO_LEN]);

#[cfg(target_has_atomic = "ptr")]
impl BalancedMemo {
    const fn new() -> Self {
        Self([const { AtomicUsize::new(0) }; BALANCED_MEMO_LEN])
    }

    fn get(&self, len: usize) -> Option<usize> {
        self.0
            .get(len)
            .map(|slot| slot.load(Ordering::Relaxed))
            .filter(|memoized| *memoized != 0)
    }

    fn store(&self, len: usize, scratch_len: usize) {
        if let Some(slot) = self.0.get(len) {
            slot.store(scratch_len, Ordering::Relaxed);
        }
    }
}

#[cfg(target_has_atomic = "ptr")]
static KARATSUBA_MUL_MEMO: BalancedMemo = BalancedMemo::new();
#[cfg(target_has_atomic = "ptr")]
static KARATSUBA_SQR_MEMO: BalancedMemo = BalancedMemo::new();
#[cfg(target_has_atomic = "ptr")]
static TOOM3_MUL_MEMO: BalancedMemo = BalancedMemo::new();
#[cfg(target_has_atomic = "ptr")]
static TOOM3_SQR_MEMO: BalancedMemo = BalancedMemo::new();

impl Multiplication {
    /// Sizes [`Karatsuba::mul`], which executes a root level below the crossover
    /// when the operand shape admits a Karatsuba split.
    pub fn karatsuba_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        if len_a < 2 || len_b < 2 || min(len_a, len_b) <= max(len_a, len_b).div_ceil(2) {
            return 0;
        }
        if len_a == len_b {
            match len_a {
                20 => return Karatsuba::BALANCED_20_SCRATCH_LIMBS,
                24 => return Karatsuba::BALANCED_24_SCRATCH_LIMBS,
                32 => return Karatsuba::BALANCED_32_SCRATCH_LIMBS,
                48 => return Karatsuba::BALANCED_48_SCRATCH_LIMBS,
                _ => {}
            }
            #[cfg(target_has_atomic = "ptr")]
            if let Some(memoized) = KARATSUBA_MUL_MEMO.get(len_a) {
                return memoized;
            }
            let split_len = Karatsuba::balanced_split_len(len_a);
            // SAFETY: len_a >= 2 and balanced_split_len retains a nonempty
            // high block, so split_len < len_a even for arbitrary sizing widths.
            let high_len = unsafe { len_a.unchecked_sub(split_len) };
            // Two split-limb differences and one guarded 2*split-limb product.
            let local_space = split_len
                .checked_mul(4)
                .and_then(|local| local.checked_add(1))
                .expect("Karatsuba local workspace overflows usize");
            let low_scratch = if split_len < KARATSUBA_THRESHOLD {
                0
            } else {
                Self::karatsuba_mul_scratch_len(split_len, split_len)
            };
            let high_scratch = if high_len < KARATSUBA_THRESHOLD {
                0
            } else {
                Self::karatsuba_mul_scratch_len(high_len, high_len)
            };
            let recursive_space = max(low_scratch, high_scratch);
            let scratch_len = local_space
                .checked_add(recursive_space)
                .expect("Karatsuba recursive workspace overflows usize");
            #[cfg(target_has_atomic = "ptr")]
            KARATSUBA_MUL_MEMO.store(len_a, scratch_len);
            return scratch_len;
        }
        let split_len = max(len_a, len_b).div_ceil(2);
        // SAFETY: split_len = ceil(max(len_a,len_b)/2), so split_len+1 fits
        // usize on every supported pointer width, including arbitrary inputs.
        let sum_space = unsafe { split_len.unchecked_add(1) };
        let local_space = sum_space
            .checked_mul(4)
            .expect("Karatsuba local workspace overflows usize");
        // Both children with a split_len-limb operand share this crossover.
        let endpoint_scratch = if split_len < KARATSUBA_THRESHOLD {
            0
        } else {
            max(
                Self::karatsuba_mul_scratch_len(sum_space, split_len),
                Self::karatsuba_mul_scratch_len(split_len, split_len),
            )
        };
        let sum_scratch = if sum_space < KARATSUBA_THRESHOLD {
            0
        } else {
            Self::karatsuba_mul_scratch_len(sum_space, sum_space)
        };
        let recursive_space = max(endpoint_scratch, sum_scratch);
        local_space
            .checked_add(recursive_space)
            .expect("Karatsuba recursive workspace overflows usize")
    }

    pub fn karatsuba_sqr_scratch_len(len: usize) -> usize {
        if len < 2 {
            return 0;
        }
        #[cfg(target_has_atomic = "ptr")]
        if let Some(memoized) = KARATSUBA_SQR_MEMO.get(len) {
            return memoized;
        }
        let split_len = len.div_ceil(2);
        // SAFETY: len >= 2 and split_len = ceil(len/2) < len.
        let high_len = unsafe { len.unchecked_sub(split_len) };
        // One split-limb difference and its guarded 2*split-limb square.
        let local_space = split_len
            .checked_mul(3)
            .and_then(|local| local.checked_add(1))
            .expect("Karatsuba local square workspace overflows usize");
        let low_scratch = if split_len < SQR_KARATSUBA_THRESHOLD {
            0
        } else {
            Self::karatsuba_sqr_scratch_len(split_len)
        };
        let high_scratch = if high_len < SQR_KARATSUBA_THRESHOLD {
            0
        } else {
            Self::karatsuba_sqr_scratch_len(high_len)
        };
        let recursive_space = max(low_scratch, high_scratch);
        let scratch_len = local_space
            .checked_add(recursive_space)
            .expect("Karatsuba recursive square workspace overflows usize");
        #[cfg(target_has_atomic = "ptr")]
        KARATSUBA_SQR_MEMO.store(len, scratch_len);
        scratch_len
    }

    pub fn toom3_dispatch_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        let smaller = min(len_a, len_b);
        if smaller < TOOM_COOK_THRESHOLD {
            return if smaller < KARATSUBA_THRESHOLD {
                0
            } else {
                Self::karatsuba_mul_scratch_len(len_a, len_b)
            };
        }
        Self::toom3_mul_scratch_len(len_a, len_b)
    }

    pub fn toom3_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        if len_a < 3 || len_b < 3 {
            return if min(len_a, len_b) < KARATSUBA_THRESHOLD {
                0
            } else {
                Self::karatsuba_mul_scratch_len(len_a, len_b)
            };
        }
        #[cfg(target_has_atomic = "ptr")]
        let balanced = len_a == len_b;
        #[cfg(target_has_atomic = "ptr")]
        if balanced && let Some(memoized) = TOOM3_MUL_MEMO.get(len_a) {
            return memoized;
        }
        let split_len = max(len_a, len_b).div_ceil(3);
        let low_len_a = min(len_a, split_len);
        let low_len_b = min(len_b, split_len);
        // SAFETY: on 16/32/64-bit targets usize::MAX=2^w-1 is divisible by
        // three. Thus m=ceil(max_len/3)<=usize::MAX/3 and 2m<=usize::MAX.
        let high_offset = unsafe { split_len.unchecked_mul(2) };
        let high_len_a = len_a.saturating_sub(high_offset);
        let high_len_b = len_b.saturating_sub(high_offset);
        // Guard multiplication is linear; only the split-width low product
        // recurses. Size that child exactly rather than the full evaluation.
        let evaluation_inner = Self::toom3_dispatch_mul_scratch_len(split_len, split_len);
        let low_inner = if low_len_a == split_len && low_len_b == split_len {
            evaluation_inner
        } else {
            Self::toom3_dispatch_mul_scratch_len(low_len_a, low_len_b)
        };
        let high_inner = Self::toom3_dispatch_mul_scratch_len(high_len_a, high_len_b);
        let inner_space = max(evaluation_inner, max(low_inner, high_inner));
        // Three (2m+1)-limb products and two (m+1)-limb evaluations.
        let scratch_len = split_len
            .checked_mul(8)
            .and_then(|width| width.checked_add(5))
            .and_then(|width| width.checked_add(inner_space))
            .expect("Toom-3 workspace exceeds usize");
        #[cfg(target_has_atomic = "ptr")]
        if balanced {
            TOOM3_MUL_MEMO.store(len_a, scratch_len);
        }
        scratch_len
    }

    pub fn toom3_dispatch_sqr_scratch_len(len: usize) -> usize {
        if len < SQR_TOOM_COOK_THRESHOLD {
            return if len < SQR_KARATSUBA_THRESHOLD {
                0
            } else {
                Self::karatsuba_sqr_scratch_len(len)
            };
        }
        Self::toom3_sqr_scratch_len(len)
    }

    pub fn toom3_sqr_scratch_len(len: usize) -> usize {
        if len < 3 {
            return if len < SQR_KARATSUBA_THRESHOLD {
                0
            } else {
                Self::karatsuba_sqr_scratch_len(len)
            };
        }
        #[cfg(target_has_atomic = "ptr")]
        if let Some(memoized) = TOOM3_SQR_MEMO.get(len) {
            return memoized;
        }
        let split_len = len.div_ceil(3);
        // SAFETY: usize::MAX is divisible by three on every supported width,
        // bounding ceil(len/3)*2 by two thirds of usize::MAX.
        let high_offset = unsafe { split_len.unchecked_mul(2) };
        let high_len = len.saturating_sub(high_offset);
        let evaluation_inner = Self::toom3_dispatch_sqr_scratch_len(split_len);
        let high_inner = Self::toom3_dispatch_sqr_scratch_len(high_len);
        // len>=3 gives a full split-width low block, identical to the guarded
        // evaluation child. Only the generally shorter high square differs.
        let inner_space = max(evaluation_inner, high_inner);
        // Three (2m+1)-limb products and one (m+1)-limb evaluation.
        let scratch_len = split_len
            .checked_mul(7)
            .and_then(|width| width.checked_add(4))
            .and_then(|width| width.checked_add(inner_space))
            .expect("Toom-3 workspace exceeds usize");
        #[cfg(target_has_atomic = "ptr")]
        TOOM3_SQR_MEMO.store(len, scratch_len);
        scratch_len
    }

    pub fn toom4_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        if len_a < 4 || len_b < 4 {
            return Self::toom3_dispatch_mul_scratch_len(len_a, len_b);
        }
        let split_len = max(len_a, len_b).div_ceil(4);
        if !Self::operand_has_four_parts(len_a, split_len)
            || !Self::operand_has_four_parts(len_b, split_len)
        {
            return Self::toom3_dispatch_mul_scratch_len(len_a, len_b);
        }
        // SAFETY: ceil(max_len/4)<=2^(w-2); multiplying by three stays below 2^w.
        let high_offset = unsafe { split_len.unchecked_mul(3) };
        let high_len_a = len_a.saturating_sub(high_offset);
        let high_len_b = len_b.saturating_sub(high_offset);
        let plan_eval = Self::select_plan(split_len, split_len, TierCeiling::Toom4);
        let evaluation_inner = Self::scratch_len(plan_eval, split_len, split_len);
        let low_inner = Self::toom3_dispatch_mul_scratch_len(split_len, split_len);
        let high_inner = Self::toom3_dispatch_mul_scratch_len(high_len_a, high_len_b);
        let inner_space = max(evaluation_inner, max(low_inner, high_inner));
        // Four 2(m+1)-limb products and two (m+1)-limb evaluations.
        split_len
            .checked_mul(10)
            .and_then(|width| width.checked_add(10))
            .and_then(|width| width.checked_add(inner_space))
            .expect("Toom-4 workspace exceeds usize")
    }

    pub fn toom4_sqr_scratch_len(len: usize) -> usize {
        if len < 4 {
            return Self::toom3_dispatch_sqr_scratch_len(len);
        }
        let split_len = len.div_ceil(4);
        if !Self::operand_has_four_parts(len, split_len) {
            return Self::toom3_dispatch_sqr_scratch_len(len);
        }
        // SAFETY: ceil(len/4)<=2^(w-2), so three times the split fits usize.
        let high_offset = unsafe { split_len.unchecked_mul(3) };
        let high_len = len.saturating_sub(high_offset);
        let plan_eval = Self::select_square_plan(split_len, TierCeiling::Toom3);
        let evaluation_inner = Self::square_scratch_len(plan_eval, split_len);
        let low_inner = Self::toom3_dispatch_sqr_scratch_len(split_len);
        let high_inner = Self::toom3_dispatch_sqr_scratch_len(high_len);
        let inner_space = max(evaluation_inner, max(low_inner, high_inner));
        // Four 2(m+1)-limb products and one (m+1)-limb evaluation.
        split_len
            .checked_mul(9)
            .and_then(|width| width.checked_add(9))
            .and_then(|width| width.checked_add(inner_space))
            .expect("Toom-4 workspace exceeds usize")
    }

    /// Workspace for one three-by-two level.
    ///
    /// The selector establishes a valid three-by-two split. Point products
    /// and `W(0)` use equal split widths; `W(inf)` uses the high parts. Their
    /// selected layouts can differ, so reserve the maximum child workspace.
    pub fn toom32_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        let widths = Widths::new(len_a, len_b);
        debug_assert!(
            widths.toom32_suitable(),
            "three-by-two scratch asked for a shape the tier cannot split"
        );
        let split_len = widths.larger.div_ceil(3);
        // SAFETY: ceil(larger/3)<=usize::MAX/3 on the even supported widths.
        let high_offset = unsafe { split_len.unchecked_mul(2) };
        let high_len_a = widths.larger.saturating_sub(high_offset);
        let high_len_b = widths.smaller.saturating_sub(split_len);

        let balanced_child = Self::select_plan(split_len, split_len, TierCeiling::Full);
        let balanced_inner = Self::scratch_len(balanced_child, split_len, split_len);
        let infinity_child = Self::select_plan(high_len_a, high_len_b, TierCeiling::Full);
        let infinity_inner = Self::scratch_len(infinity_child, high_len_a, high_len_b);
        let inner_space = max(balanced_inner, infinity_inner);
        // Two 2(m+1)-limb products and two (m+1)-limb positive evaluations;
        // their negative evaluations reuse the positive-product slot.
        split_len
            .checked_mul(6)
            .and_then(|width| width.checked_add(6))
            .and_then(|width| width.checked_add(inner_space))
            .expect("Toom-3-by-2 workspace exceeds usize")
    }

    /// Workspace for one four-by-three level.
    ///
    /// Point products and `W(0)` use equal split widths; `W(inf)` uses the high
    /// parts. Reserve the maximum of their independently selected layouts.
    pub fn toom43_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        let widths = Widths::new(len_a, len_b);
        debug_assert!(
            widths.toom43_suitable(),
            "four-by-three scratch asked for a shape the tier cannot split"
        );
        let split_len = widths.larger.div_ceil(4);
        // SAFETY: m=ceil(larger/4)<=2^(w-2), bounding both 3m and 2m by usize::MAX.
        let (large_offset, small_offset) =
            unsafe { (split_len.unchecked_mul(3), split_len.unchecked_mul(2)) };
        let high_len_a = widths.larger.saturating_sub(large_offset);
        let high_len_b = widths.smaller.saturating_sub(small_offset);

        let balanced_child = Self::select_plan(split_len, split_len, TierCeiling::Full);
        let balanced_inner = Self::scratch_len(balanced_child, split_len, split_len);
        let infinity_child = Self::select_plan(high_len_a, high_len_b, TierCeiling::Full);
        let infinity_inner = Self::scratch_len(infinity_child, high_len_a, high_len_b);
        let inner_space = max(balanced_inner, infinity_inner);
        // Four 2(m+1)-limb products and two (m+1)-limb positive evaluations;
        // each positive-product slot first holds its negative evaluations.
        split_len
            .checked_mul(10)
            .and_then(|width| width.checked_add(10))
            .and_then(|width| width.checked_add(inner_space))
            .expect("Toom-4-by-3 workspace exceeds usize")
    }

    pub fn toom6_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        if len_a < 6 || len_b < 6 {
            return Self::toom4_mul_scratch_len(len_a, len_b);
        }
        let Some(shape) = Widths::new(len_a, len_b).toom6_shape() else {
            // Mirror `toom6::mul`, which hands an unsuitable shape to
            // `recursive_mul` under a Toom-4 ceiling. The full ceiling names a
            // blocked or fractional plan for every lopsided shape, whose layout
            // differs from the capped plan the driver actually runs.
            let plan = Self::select_plan(len_a, len_b, TierCeiling::Toom4);
            return Self::scratch_len(plan, len_a, len_b);
        };
        if matches!(shape, MulShape::Half) {
            return Toom6::mul_half_scratch_len(len_a, len_b);
        }
        let split_len = max(len_a, len_b).div_ceil(6);
        // A power-of-two split needs no evaluation guard limb, so the widest
        // evaluated product is the split itself rather than split+1.
        let plan_low = Self::select_plan(split_len, split_len, TierCeiling::Toom4);
        let low_inner = Self::scratch_len(plan_low, split_len, split_len);
        let evaluation_inner = if split_len.is_power_of_two() {
            low_inner
        } else {
            // SAFETY: m=ceil(max_len/6)<=usize::MAX/6+1, so m+1 fits w>=16.
            let eval_len = unsafe { split_len.unchecked_add(1) };
            let plan_eval = Self::select_plan(eval_len, eval_len, TierCeiling::Toom4);
            Self::scratch_len(plan_eval, eval_len, eval_len)
        };
        let inner_space = max(evaluation_inner, low_inner);
        let points_are_placed = Toom6::destination_points_fit(
            len_a
                .checked_add(len_b)
                .expect("Toom-6 product width overflows usize"),
            split_len,
        );
        Toom6::local_scratch_len::<false>(split_len, inner_space, points_are_placed)
    }

    pub fn toom6_sqr_scratch_len(len: usize) -> usize {
        if len < 6 {
            return Self::toom4_sqr_scratch_len(len);
        }
        let split_len = len.div_ceil(6);
        let plan_low = Self::select_square_plan(split_len, TierCeiling::Toom4);
        let low_inner = Self::square_scratch_len(plan_low, split_len);
        let evaluation_inner = if split_len.is_power_of_two() {
            low_inner
        } else {
            // SAFETY: m=ceil(len/6)<=usize::MAX/6+1, leaving room for its guard.
            let eval_len = unsafe { split_len.unchecked_add(1) };
            let plan_eval = Self::select_square_plan(eval_len, TierCeiling::Toom4);
            Self::square_scratch_len(plan_eval, eval_len)
        };
        let inner_space = max(evaluation_inner, low_inner);
        let points_are_placed = Toom6::destination_points_fit(
            len.checked_mul(2)
                .expect("Toom-6 square width overflows usize"),
            split_len,
        );
        Toom6::local_scratch_len::<true>(split_len, inner_space, points_are_placed)
    }

    pub fn toom8_mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        let Some(shape) = Widths::new(len_a, len_b).toom8_shape() else {
            let plan = Self::select_plan(len_a, len_b, TierCeiling::Toom6);
            return Self::scratch_len(plan, len_a, len_b);
        };
        let split_width = Toom8::multiplication_split_len(shape, len_a, len_b);
        Toom8::local_mul_scratch_len(shape, split_width, len_a, len_b)
    }

    pub fn toom8_sqr_scratch_len(len: usize) -> usize {
        if !Self::operand_has_eight_parts(len) {
            let plan = Self::select_square_plan(len, TierCeiling::Toom6);
            return Self::square_scratch_len(plan, len);
        }
        Toom8::local_sqr_scratch_len(len)
    }
}
