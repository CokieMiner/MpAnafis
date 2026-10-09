//! Power-of-two paired evaluation for Toom-8 and Toom-8.5.

#![expect(
    unsafe_code,
    reason = "Admitted nonempty parts fit guarded evaluations; fixed point exponents bound word weights and Horner shifts"
)]

use core::num::NonZeroUsize;

use super::{AddMulKernel, ArchKernels, Limb, SharedEval, Toom8};

#[derive(Clone, Copy, Debug)]
pub enum EvaluationDirection {
    Direct,
    Reciprocal,
}

/// The fixed point exponents; every direct or reciprocal shift is at most three.
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum PointShift {
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
}

/// Non-unit points after the evaluator has handled `k=1` once.
#[derive(Clone, Copy)]
#[repr(u32)]
enum EvaluationShift {
    One = 1,
    Two = 2,
    Three = 3,
}

#[derive(Clone, Copy, Debug)]
pub struct EvaluationPoint {
    pub direction: EvaluationDirection,
    pub shift: PointShift,
}

/// Seven or eight complete low blocks and one nonempty high block of at most m
/// limbs. Only an admitted eight-/nine-part operand can construct this view.
#[derive(Clone, Copy)]
struct Parts<'value> {
    low: &'value [Limb],
    high: &'value [Limb],
    split_width: NonZeroUsize,
    nine: bool,
}

impl Toom8 {
    pub fn evaluate_even_odd(
        even: &mut [Limb],
        odd: &mut [Limb],
        operand: &[Limb],
        split_width: NonZeroUsize,
        point: EvaluationPoint,
        kernel: AddMulKernel,
    ) -> bool {
        let split_len = split_width.get();
        debug_assert_eq!(even.len(), odd.len(), "evaluation widths must match");
        debug_assert!(
            split_len < even.len(),
            "each Toom-8 part must fit below the evaluation guard"
        );
        debug_assert!(
            (8..=9).contains(&operand.len().div_ceil(split_len)),
            "Toom-8 admission requires eight or nine nonempty parts"
        );
        // SAFETY: admission gives 7m<|A|<=9m. Actual limb-slice byte bounds
        // leave room for 8m. Selecting degree seven/eight therefore places the
        // split inside A and leaves seven/eight complete low blocks plus 1..=m
        // high limbs. Immutable splitting preserves initialization and lifetimes.
        let parts = unsafe {
            let eight_parts = split_len.unchecked_mul(8);
            let nine = operand.len() > eight_parts;
            let high_offset = if nine {
                eight_parts
            } else {
                split_len.unchecked_mul(7)
            };
            let (low, high) = operand.split_at_unchecked(high_offset);
            Parts {
                low,
                high,
                split_width,
                nine,
            }
        };
        let shift = match point.shift {
            PointShift::Zero => {
                // At unit denominator, direct and reciprocal evaluation both
                // give A(+/-1); no coefficient shift or scalar product is needed.
                evaluate_direct_one(even, odd, parts);
                return even.iter().rev().cmp(odd.iter().rev()).is_lt();
            }
            PointShift::One => EvaluationShift::One,
            PointShift::Two => EvaluationShift::Two,
            PointShift::Three => EvaluationShift::Three,
        };
        match (point.direction, Limb::BITS >= 32) {
            (EvaluationDirection::Direct, true) => {
                evaluate_direct_word_weights(even, odd, parts, shift, kernel);
            }
            (EvaluationDirection::Reciprocal, true) => {
                evaluate_reciprocal_word_weights(even, odd, parts, shift, kernel);
            }
            (EvaluationDirection::Direct, false) => {
                evaluate_direct_horner(even, odd, parts, shift);
            }
            (EvaluationDirection::Reciprocal, false) => {
                evaluate_reciprocal_horner(even, odd, parts, shift);
            }
        }
        even.iter().rev().cmp(odd.iter().rev()).is_lt()
    }
}

fn evaluate_direct_horner(
    even: &mut [Limb],
    odd: &mut [Limb],
    parts: Parts<'_>,
    point: EvaluationShift,
) {
    let eval_len = even.len();
    let split_len = parts.split_width.get();
    #[expect(
        clippy::as_conversions,
        reason = "EvaluationShift's u32 discriminants are 1..=3 on every pointer width"
    )]
    let point_shift = point as u32;
    // SAFETY: EvaluationShift gives 1<=point_shift<=3, hence 2<=squared_shift<=6.
    let squared_shift = unsafe { point_shift.unchecked_mul(2) };
    let (leading, trailing) = if parts.nine {
        (&mut *even, &mut *odd)
    } else {
        (&mut *odd, &mut *even)
    };
    // SAFETY: low contains seven/eight full blocks. Removing its last
    // m-limb block leaves six/seven complete blocks for the three pairs below.
    let (mut remaining, next) = unsafe {
        parts
            .low
            .split_at_unchecked(parts.low.len().unchecked_sub(split_len))
    };
    SharedEval::copy_part(leading, parts.high);
    SharedEval::copy_part(trailing, next);
    for _ in 0..3 {
        // SAFETY: remaining begins with six/seven full blocks; each of
        // three iterations removes exactly two, leaving zero/one. Every
        // subtraction and split stays in the original initialized low span.
        let (lower, trailing_part, leading_part) = unsafe {
            let pair_len = split_len.unchecked_mul(2);
            let (lower, pair) =
                remaining.split_at_unchecked(remaining.len().unchecked_sub(pair_len));
            let (trailing_part, leading_part) = pair.split_at_unchecked(split_len);
            (lower, trailing_part, leading_part)
        };
        // SAFETY: both views retain eval_len initialized, disjoint limbs.
        // 2<=squared_shift<=6<Limb::BITS, and the 25-bit guard bound absorbs carry.
        unsafe {
            let leading_carry =
                ArchKernels::lshift_unchecked(leading.as_mut_ptr(), eval_len, squared_shift);
            debug_assert_eq!(leading_carry, 0, "leading evaluation exceeds its guard");
            let trailing_carry =
                ArchKernels::lshift_unchecked(trailing.as_mut_ptr(), eval_len, squared_shift);
            debug_assert_eq!(trailing_carry, 0, "trailing evaluation exceeds its guard");
        }
        // SAFETY: both full blocks fit below their initialized guards.
        unsafe {
            SharedEval::add_part(leading, leading_part);
            SharedEval::add_part(trailing, trailing_part);
        }
        remaining = lower;
    }
    if parts.nine {
        // SAFETY: leading retains eval_len initialized limbs and enough guards
        // for the next bounded evaluation; 2<=squared_shift<=6<Limb::BITS.
        let carry =
            unsafe { ArchKernels::lshift_unchecked(leading.as_mut_ptr(), eval_len, squared_shift) };
        debug_assert_eq!(carry, 0, "degree-eight evaluation exceeds its guard");
        // SAFETY: degree eight leaves exactly the full constant block.
        unsafe {
            SharedEval::add_part(leading, remaining);
        }
    }
    // SAFETY: odd retains eval_len initialized limbs; 1<=point_shift<=3<Limb::BITS
    // and the complete polynomial bound fits its retained evaluation guards.
    let carry = unsafe { ArchKernels::lshift_unchecked(odd.as_mut_ptr(), eval_len, point_shift) };
    debug_assert_eq!(carry, 0, "odd evaluation exceeds its guard");
}

fn evaluate_reciprocal_horner(
    even: &mut [Limb],
    odd: &mut [Limb],
    parts: Parts<'_>,
    point: EvaluationShift,
) {
    let eval_len = even.len();
    let split_len = parts.split_width.get();
    #[expect(
        clippy::as_conversions,
        reason = "EvaluationShift's u32 discriminants are 1..=3 on every pointer width"
    )]
    let denominator_shift = point as u32;
    // SAFETY: EvaluationShift gives 1<=denominator_shift<=3 and 2<=squared_shift<=6.
    let squared_shift = unsafe { denominator_shift.unchecked_mul(2) };
    // SAFETY: low has seven/eight full blocks, including constant and linear.
    let (constant, linear, mut remaining) = unsafe {
        let (constant, after_constant) = parts.low.split_at_unchecked(split_len);
        let (linear, remaining) = after_constant.split_at_unchecked(split_len);
        (constant, linear, remaining)
    };
    SharedEval::copy_part(even, constant);
    SharedEval::copy_part(odd, linear);
    for _ in 0..2 {
        // SAFETY: the five/six full remaining blocks supply two complete
        // pairs and the full sextic block. Consecutive immutable splits fit.
        let (even_part, odd_part, after_pair) = unsafe {
            let (even_part, rest) = remaining.split_at_unchecked(split_len);
            let (odd_part, after_pair) = rest.split_at_unchecked(split_len);
            (even_part, odd_part, after_pair)
        };
        // SAFETY: even/odd retain eval_len initialized disjoint limbs.
        // 2<=squared_shift<=6<Limb::BITS; the guarded Horner prefixes fit.
        unsafe {
            let even_carry =
                ArchKernels::lshift_unchecked(even.as_mut_ptr(), eval_len, squared_shift);
            debug_assert_eq!(even_carry, 0, "even reciprocal prefix exceeds its guard");
            let odd_carry =
                ArchKernels::lshift_unchecked(odd.as_mut_ptr(), eval_len, squared_shift);
            debug_assert_eq!(odd_carry, 0, "odd reciprocal prefix exceeds its guard");
        }
        // SAFETY: the full blocks fit below both initialized guards.
        unsafe {
            SharedEval::add_part(even, even_part);
            SharedEval::add_part(odd, odd_part);
        }
        remaining = after_pair;
    }
    // SAFETY: after four blocks, remaining contains the full sextic block
    // and, for degree eight, a full septic block. The high block is <=m.
    let (sextic, optional_odd) = unsafe { remaining.split_at_unchecked(split_len) };
    // SAFETY: the same initialized guarded views and positive shift bound hold;
    // the next prefixes remain below the complete 25-bit evaluation bound.
    unsafe {
        let even_carry = ArchKernels::lshift_unchecked(even.as_mut_ptr(), eval_len, squared_shift);
        debug_assert_eq!(even_carry, 0, "even reciprocal prefix exceeds its guard");
        let odd_carry = ArchKernels::lshift_unchecked(odd.as_mut_ptr(), eval_len, squared_shift);
        debug_assert_eq!(odd_carry, 0, "odd reciprocal prefix exceeds its guard");
    }
    // SAFETY: each source has at most m limbs below initialized guards.
    unsafe {
        SharedEval::add_part(even, sextic);
        SharedEval::add_part(odd, if parts.nine { optional_odd } else { parts.high });
    }
    if parts.nine {
        // SAFETY: even retains eval_len initialized limbs with guards for the
        // degree-eight prefix; 2<=squared_shift<=6<Limb::BITS.
        let carry =
            unsafe { ArchKernels::lshift_unchecked(even.as_mut_ptr(), eval_len, squared_shift) };
        debug_assert_eq!(carry, 0, "degree-eight reciprocal exceeds its guard");
        // SAFETY: the degree-eight high block fits below the evaluation guard.
        unsafe {
            SharedEval::add_part(even, parts.high);
        }
    }
    let scaled = if parts.nine { odd } else { even };
    // SAFETY: scaled retains eval_len initialized limbs. EvaluationShift bounds
    // 1<=denominator_shift<=3<Limb::BITS, and the complete evaluation fits its guards.
    let carry =
        unsafe { ArchKernels::lshift_unchecked(scaled.as_mut_ptr(), eval_len, denominator_shift) };
    debug_assert_eq!(carry, 0, "scaled reciprocal evaluation exceeds its guard");
}

fn evaluate_direct_word_weights(
    even: &mut [Limb],
    odd: &mut [Limb],
    parts: Parts<'_>,
    point: EvaluationShift,
    kernel: AddMulKernel,
) {
    let split_len = parts.split_width.get();
    // SAFETY: Parts contains seven/eight full low blocks. The first two
    // splits leave exactly five/six complete low blocks.
    let (constant, linear, mut remaining) = unsafe {
        let (constant, rest) = parts.low.split_at_unchecked(split_len);
        let (linear, remaining) = rest.split_at_unchecked(split_len);
        (constant, linear, remaining)
    };
    SharedEval::copy_part(even, constant);
    SharedEval::copy_part(odd, linear);

    #[expect(
        clippy::as_conversions,
        reason = "EvaluationShift has u32 discriminants from one through three on all supported targets"
    )]
    let point_shift = point as u32;
    // SAFETY: EvaluationShift bounds 1<=point_shift<=3, hence 2<=squared_shift<=6.
    let squared_shift = unsafe { point_shift.unchecked_mul(2) };
    // SAFETY: squared_shift<=6 is below each supported limb width, including 16.
    let squared_scalar = unsafe { 1_usize.unchecked_shl(squared_shift) };
    let mut scalar = 1_usize;
    for _ in 0..2 {
        // SAFETY: the two advances reach weights z and z^2, with z<=64.
        scalar = unsafe { scalar.unchecked_mul(squared_scalar) };
        // SAFETY: remaining starts with five/six complete low blocks;
        // two iterations consume four, leaving the sextic and optional septic.
        let (even_part, odd_part, after_pair) = unsafe {
            let (even_part, rest) = remaining.split_at_unchecked(split_len);
            let (odd_part, after_pair) = rest.split_at_unchecked(split_len);
            (even_part, odd_part, after_pair)
        };
        SharedEval::add_mul_word_with_kernel_in_place(even, even_part, scalar, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(odd, odd_part, scalar, kernel);
        remaining = after_pair;
    }
    // SAFETY: the remaining one/two full blocks contain sextic and optional
    // septic. Advancing z^2 to z^3 fits a >=32-bit limb since z<=64.
    let (sextic, optional_odd, sixth_scalar) = unsafe {
        let (sextic, optional_odd) = remaining.split_at_unchecked(split_len);
        (sextic, optional_odd, scalar.unchecked_mul(squared_scalar))
    };
    SharedEval::add_mul_word_with_kernel_in_place(even, sextic, sixth_scalar, kernel);
    SharedEval::add_mul_word_with_kernel_in_place(
        odd,
        if parts.nine { optional_odd } else { parts.high },
        sixth_scalar,
        kernel,
    );
    if parts.nine {
        // SAFETY: z^4<=64^4=2^24 fits the >=32-bit scalar path.
        let highest_scalar = unsafe { sixth_scalar.unchecked_mul(squared_scalar) };
        SharedEval::add_mul_word_with_kernel_in_place(even, parts.high, highest_scalar, kernel);
    }
    // SAFETY: odd retains even.len() initialized limbs; 1<=point_shift<=3<Limb::BITS
    // and the complete evaluation bound fits its guards without escaping carry.
    let carry = unsafe { ArchKernels::lshift_unchecked(odd.as_mut_ptr(), even.len(), point_shift) };
    debug_assert_eq!(carry, 0, "odd weighted evaluation exceeds its guard");
}

fn evaluate_direct_one(even: &mut [Limb], odd: &mut [Limb], parts: Parts<'_>) {
    let split_len = parts.split_width.get();
    // SAFETY: Parts contains seven/eight complete initialized low blocks.
    // Consecutive splits expose the first seven and an optional eighth, all
    // exactly m limbs; no chunk-count or remainder calculation is needed.
    let (constant, linear, quadratic, cubic, quartic, quintic, sextic, optional_odd) = unsafe {
        let (constant, after_constant) = parts.low.split_at_unchecked(split_len);
        let (linear, after_linear) = after_constant.split_at_unchecked(split_len);
        let (quadratic, after_quadratic) = after_linear.split_at_unchecked(split_len);
        let (cubic, after_cubic) = after_quadratic.split_at_unchecked(split_len);
        let (quartic, after_quartic) = after_cubic.split_at_unchecked(split_len);
        let (quintic, after_quintic) = after_quartic.split_at_unchecked(split_len);
        let (sextic, optional_odd) = after_quintic.split_at_unchecked(split_len);
        (
            constant,
            linear,
            quadratic,
            cubic,
            quartic,
            quintic,
            sextic,
            optional_odd,
        )
    };

    sum_initial_parts(even, constant, quadratic);
    sum_initial_parts(odd, linear, cubic);

    // At x=1 every part has unit weight. The fixed schedule avoids a chunk
    // iterator and scalar multiplication for every remaining addition.
    // SAFETY: both sums initialized their complete guarded destinations;
    // all full and high parts have <=m limbs and remain disjoint from scratch.
    unsafe {
        SharedEval::add_part(even, quartic);
        SharedEval::add_part(odd, quintic);
        SharedEval::add_part(even, sextic);
        SharedEval::add_part(odd, if parts.nine { optional_odd } else { parts.high });
        if parts.nine {
            SharedEval::add_part(even, parts.high);
        }
    }
}

fn sum_initial_parts(dst: &mut [Limb], left: &[Limb], right: &[Limb]) {
    debug_assert!(left.len() < dst.len(), "each chunk fits below the guard");
    debug_assert_eq!(
        left.len(),
        right.len(),
        "the two initial parts must have equal widths"
    );
    debug_assert_ne!(left.len(), 0, "initial evaluation parts must be nonempty");
    // SAFETY: evaluate_direct_one supplies full, nonempty first/fourth m-limb
    // chunks. Its disjoint destination has m+ceil(25/LIMB_BITS) limbs.
    let (body, guard) = unsafe { dst.split_at_mut_unchecked(left.len()) };
    // SAFETY: both sources and the destination body span m initialized,
    // mutually disjoint limbs. The primitive overwrites the complete body.
    let carry = unsafe {
        ArchKernels::add_limbs_3_unchecked(
            body.as_mut_ptr(),
            left.as_ptr(),
            right.as_ptr(),
            body.len(),
        )
    };
    // SAFETY: ceil(25/LIMB_BITS) >= 1 leaves a nonempty guard suffix.
    let (first_guard, remaining_guards) = unsafe { guard.split_first_mut().unwrap_unchecked() };
    *first_guard = carry;
    remaining_guards.fill(0);
}

fn evaluate_reciprocal_word_weights(
    even: &mut [Limb],
    odd: &mut [Limb],
    parts: Parts<'_>,
    point: EvaluationShift,
    kernel: AddMulKernel,
) {
    let split_len = parts.split_width.get();
    let (leading, trailing) = if parts.nine {
        (&mut *even, &mut *odd)
    } else {
        (&mut *odd, &mut *even)
    };
    // SAFETY: seven/eight complete low blocks have a final full block;
    // removing it leaves exactly six/seven full blocks for the paired solve.
    let (mut remaining, next) = unsafe {
        parts
            .low
            .split_at_unchecked(parts.low.len().unchecked_sub(split_len))
    };
    SharedEval::copy_part(leading, parts.high);
    SharedEval::copy_part(trailing, next);

    #[expect(
        clippy::as_conversions,
        reason = "EvaluationShift has u32 discriminants from one through three on all supported targets"
    )]
    let denominator_shift = point as u32;
    // SAFETY: EvaluationShift bounds denominator_shift in 1..=3, hence squared_shift in 2..=6.
    let squared_shift = unsafe { denominator_shift.unchecked_mul(2) };
    // SAFETY: squared_shift<=6 is below each supported limb width, including 16.
    let squared_scalar = unsafe { 1_usize.unchecked_shl(squared_shift) };
    let mut scalar = 1_usize;
    for _ in 0..3 {
        // SAFETY: three advances reach z^3<=64^3=2^18, within a >=32-bit limb.
        scalar = unsafe { scalar.unchecked_mul(squared_scalar) };
        // SAFETY: the six/seven full remaining blocks supply exactly three
        // pairs. Each removal leaves four/five, two/three, then zero/one blocks.
        let (lower, trailing_part, leading_part) = unsafe {
            let pair_len = split_len.unchecked_mul(2);
            let (lower, pair) =
                remaining.split_at_unchecked(remaining.len().unchecked_sub(pair_len));
            let (trailing_part, leading_part) = pair.split_at_unchecked(split_len);
            (lower, trailing_part, leading_part)
        };
        SharedEval::add_mul_word_with_kernel_in_place(leading, leading_part, scalar, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(trailing, trailing_part, scalar, kernel);
        remaining = lower;
    }
    if parts.nine {
        // SAFETY: degree eight leaves the full constant block and needs
        // z^4<=2^24, which fits this >=32-bit scalar path.
        let highest_scalar = unsafe { scalar.unchecked_mul(squared_scalar) };
        SharedEval::add_mul_word_with_kernel_in_place(leading, remaining, highest_scalar, kernel);
    }
    // SAFETY: trailing retains its original initialized guarded view; the enum
    // gives 1<=denominator_shift<=3<Limb::BITS. The full evaluation fits its guards.
    let carry = unsafe {
        ArchKernels::lshift_unchecked(trailing.as_mut_ptr(), trailing.len(), denominator_shift)
    };
    debug_assert_eq!(carry, 0, "weighted reciprocal evaluation exceeds its guard");
}
