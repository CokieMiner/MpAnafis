//! Paired power-of-two interpolation for Toom-8 and Toom-8.5.

use core::{cmp::min, mem::swap};

use super::{
    AddMulKernel, Addition, ArchKernels, LIMB_BITS, Limb, ScaledSource, SharedEval,
    TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS, Toom8,
};

pub struct Values<'buffer> {
    pub one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub four: &'buffer mut [Limb],
    pub eight: &'buffer mut [Limb],
    pub half: &'buffer mut [Limb],
    pub quarter: &'buffer mut [Limb],
    pub eighth: &'buffer mut [Limb],
}

pub struct CouplingContext<'value> {
    pub zero: &'value [Limb],
    pub infinity: &'value [Limb],
    pub split_len: usize,
    pub degree: usize,
}

impl Toom8 {
    pub fn couple_direct(
        packed: &mut [Limb],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        context: &CouplingContext<'_>,
        point_shift: usize,
    ) {
        let value_len = negative.len();
        let (_, positive_and_guard) = packed.split_at_mut(context.split_len);
        let (positive, _) = positive_and_guard.split_at_mut(value_len);
        let mut pair = PairValues {
            positive,
            negative,
            negative_product_is_negative,
        };
        recover_scaled_even_odd(&mut pair);
        SharedEval::sub_full_slices_in_place(pair.positive, context.zero);
        Self::exact_signed_div2_repeated(pair.positive, point_shift.wrapping_mul(2));
        Self::exact_signed_div2_repeated(pair.negative, point_shift);
        if context.degree == 15 {
            subtract_shifted_bits(
                pair.negative,
                context.infinity,
                point_shift.wrapping_mul(14),
            );
        }
        pack_even_odd(
            packed,
            negative,
            context.split_len,
            negative_product_is_negative,
        );
    }

    pub fn couple_reciprocal(
        packed: &mut [Limb],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        context: &CouplingContext<'_>,
        denominator_shift: usize,
    ) {
        let value_len = negative.len();
        let (_, positive_and_guard) = packed.split_at_mut(context.split_len);
        let (positive, _) = positive_and_guard.split_at_mut(value_len);
        let mut pair = PairValues {
            positive,
            negative,
            negative_product_is_negative,
        };
        recover_scaled_even_odd(&mut pair);
        if context.degree == 14 {
            subtract_shifted_bits(
                pair.positive,
                context.zero,
                denominator_shift.wrapping_mul(14),
            );
            Self::exact_signed_div2_repeated(pair.negative, denominator_shift);
        } else {
            Self::exact_signed_div2_repeated(pair.positive, denominator_shift);
            subtract_shifted_bits(
                pair.positive,
                context.zero,
                denominator_shift.wrapping_mul(14),
            );
            SharedEval::sub_full_slices_in_place(pair.negative, context.infinity);
            Self::exact_signed_div2_repeated(pair.negative, denominator_shift.wrapping_mul(2));
        }
        pack_even_odd(
            packed,
            negative,
            context.split_len,
            negative_product_is_negative,
        );
    }
}

impl Toom8 {
    pub fn interpolate_and_reconstruct(
        dst: &mut [Limb],
        split_len: usize,
        values: Values<'_>,
        temporary: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        let Values {
            one,
            two,
            four,
            eight,
            half,
            quarter,
            eighth,
        } = values;

        Self::interpolate_values(
            Values {
                one: &mut *one,
                two: &mut *two,
                four: &mut *four,
                eight: &mut *eight,
                half: &mut *half,
                quarter: &mut *quarter,
                eighth: &mut *eighth,
            },
            temporary,
            add_mul_kernel,
        );

        SharedEval::add_coefficient_in_place(dst, eighth, split_len);
        SharedEval::add_coefficient_in_place(dst, half, split_len.wrapping_mul(3));
        SharedEval::add_coefficient_in_place(dst, quarter, split_len.wrapping_mul(5));
        SharedEval::add_coefficient_in_place(dst, one, split_len.wrapping_mul(7));
        SharedEval::add_coefficient_in_place(dst, two, split_len.wrapping_mul(9));
        SharedEval::add_coefficient_in_place(dst, four, split_len.wrapping_mul(11));
        SharedEval::add_coefficient_in_place(dst, eight, split_len.wrapping_mul(13));
    }

    /// Interpolate seven packed point pairs in place.
    ///
    /// On return, `eighth`, `half`, `quarter`, `one`, `two`, `four`, and `eight`
    /// respectively hold coefficient pairs beginning at shifts 1 through 13.
    pub fn interpolate_values(
        values: Values<'_>,
        temporary: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        let Values {
            one,
            two,
            four,
            eight,
            half,
            quarter,
            eighth,
        } = values;

        // Solve the antisymmetric rows. Every division follows from eliminating
        // the reciprocal-minus-direct Vandermonde system at z=4,16,64, so all
        // quotients are exact fixed-width two's-complement values.
        Self::linear_combination(quarter, 1, half, -1_028, temporary, add_mul_kernel);
        Self::exact_sub_mul_two_u64_odd_in_place(
            eighth,
            [
                ScaledSource {
                    value: quarter,
                    scalar: 1_300,
                },
                ScaledSource {
                    value: half,
                    scalar: 1_052_688,
                },
            ],
            48_070_897_875,
            temporary,
            add_mul_kernel,
        );
        Self::exact_sub_mul_u64_odd_in_place(
            quarter,
            eighth,
            12_567_555,
            2_835,
            temporary,
            add_mul_kernel,
        );
        SharedEval::exact_signed_div_power_of_two_in_place(quarter, 6);
        Self::linear_combination(half, 1, eighth, -4_095, temporary, add_mul_kernel);
        Self::linear_combination(half, 1, quarter, 240, temporary, add_mul_kernel);
        Self::exact_signed_div_u64(half, 1_020);

        // Solve the symmetric rows after removing the central packed coefficient.
        Self::linear_combination(two, 1, one, -128, temporary, add_mul_kernel);
        Self::linear_combination(four, 1, one, -8_192, temporary, add_mul_kernel);
        Self::linear_combination(four, 1, two, -400, temporary, add_mul_kernel);
        Self::linear_combination(eight, 1, one, -524_288, temporary, add_mul_kernel);
        Self::linear_combination(eight, 1, four, -1_428, temporary, add_mul_kernel);
        Self::exact_sub_mul_u64_odd_in_place(
            eight,
            two,
            112_896,
            46_591_793_325,
            temporary,
            add_mul_kernel,
        );
        Self::exact_sub_mul_u64_odd_in_place(
            four,
            eight,
            15_181_425,
            42_525,
            temporary,
            add_mul_kernel,
        );
        SharedEval::exact_signed_div_power_of_two_in_place(four, 4);
        Self::linear_combination(two, 1, eight, -3_969, temporary, add_mul_kernel);
        Self::exact_sub_mul_u64_odd_in_place(two, four, 900, 9, temporary, add_mul_kernel);
        SharedEval::exact_signed_div_power_of_two_in_place(two, 4);
        SharedEval::sub_full_slices_in_place(one, eight);
        SharedEval::sub_full_slices_in_place(one, two);
        SharedEval::sub_full_slices_in_place(one, four);

        recover_pair(four, half);
        recover_reverse_pair(two, quarter);
        recover_pair(eight, eighth);
    }

    /// Add coefficient pairs not already placed at shifts 3, 7, and 11.
    pub fn reconstruct_alternating(
        dst: &mut [Limb],
        split_len: usize,
        coefficients: [&[Limb]; 4],
        fast_paired_add: bool,
    ) {
        let [first, fifth, ninth, thirteenth] = coefficients;
        // The existing single-add backend is eight-way unrolled and wins on short
        // spans. Once a packed coefficient reaches 256 limbs, the independent ADX
        // carry chains repay their loop setup and reduce the two passes to one.
        if fast_paired_add && first.len() >= TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS {
            reconstruct_alternating_paired(dst, split_len, first, fifth, ninth, thirteenth);
            return;
        }
        SharedEval::add_coefficient_in_place(dst, first, split_len);
        SharedEval::add_coefficient_in_place(dst, fifth, split_len.wrapping_mul(5));
        SharedEval::add_coefficient_in_place(dst, ninth, split_len.wrapping_mul(9));
        SharedEval::add_coefficient_in_place(dst, thirteenth, split_len.wrapping_mul(13));
    }
}

fn reconstruct_alternating_paired(
    dst: &mut [Limb],
    split_len: usize,
    first: &[Limb],
    fifth: &[Limb],
    ninth: &[Limb],
    thirteenth: &[Limb],
) {
    let packed_len = first.len();
    debug_assert_eq!(fifth.len(), packed_len, "coefficient widths differ");
    debug_assert_eq!(ninth.len(), packed_len, "coefficient widths differ");
    debug_assert_eq!(thirteenth.len(), packed_len, "coefficient widths differ");
    let first_shift = split_len;
    let fifth_shift = split_len.wrapping_mul(5);
    let ninth_shift = split_len.wrapping_mul(9);
    let thirteenth_shift = split_len.wrapping_mul(13);
    debug_assert!(
        ninth_shift.wrapping_add(packed_len) <= dst.len(),
        "paired coefficient exceeds destination"
    );

    let dst_ptr = dst.as_mut_ptr();
    // SAFETY: the first and ninth destinations are disjoint spans of
    // packed_len limbs inside dst. All four sources occupy separate scratch
    // buffers and cannot overlap either destination or one another.
    let (first_carry, ninth_carry) = unsafe {
        ArchKernels::add_two_limbs_unchecked(
            dst_ptr.add(first_shift),
            first.as_ptr(),
            dst_ptr.add(ninth_shift),
            ninth.as_ptr(),
            packed_len,
        )
    };
    propagate_coefficient_carry(dst, first_shift.wrapping_add(packed_len), first_carry);
    propagate_coefficient_carry(dst, ninth_shift.wrapping_add(packed_len), ninth_carry);

    let thirteenth_len = dst.len().saturating_sub(thirteenth_shift);
    let paired_len = min(packed_len, thirteenth_len);
    debug_assert!(
        thirteenth
            .get(paired_len..)
            .is_none_or(|tail| tail.iter().all(|limb| *limb == 0)),
        "highest coefficient exceeds destination"
    );
    // SAFETY: the fifth and thirteenth spans are disjoint, both cover
    // paired_len limbs, and their source buffers are mutually disjoint scratch.
    let (fifth_carry, thirteenth_carry) = unsafe {
        ArchKernels::add_two_limbs_unchecked(
            dst_ptr.add(fifth_shift),
            fifth.as_ptr(),
            dst_ptr.add(thirteenth_shift),
            thirteenth.as_ptr(),
            paired_len,
        )
    };
    let (_, fifth_tail) = fifth.split_at(paired_len);
    let fifth_tail_shift = fifth_shift.wrapping_add(paired_len);
    let tail_carry = if fifth_tail.is_empty() {
        0
    } else {
        let (_, tail_dst) = dst.split_at_mut(fifth_tail_shift);
        Addition::add_slice_in_place(tail_dst, fifth_tail)
    };
    propagate_coefficient_carry(
        dst,
        fifth_tail_shift.wrapping_add(fifth_tail.len()),
        tail_carry,
    );
    propagate_coefficient_carry(dst, fifth_tail_shift, fifth_carry);
    debug_assert_eq!(
        thirteenth_carry, 0,
        "highest coefficient carried beyond destination"
    );
}

fn propagate_coefficient_carry(dst: &mut [Limb], start: usize, mut carry: Limb) {
    let (_, suffix) = dst.split_at_mut(start);
    for limb in suffix {
        if carry == 0 {
            break;
        }
        let (sum, overflow) = limb.overflowing_add(carry);
        *limb = sum;
        carry = Limb::from(overflow);
    }
    debug_assert_eq!(carry, 0, "coefficient carry exceeded destination");
}

struct PairValues<'buffer> {
    positive: &'buffer mut [Limb],
    negative: &'buffer mut [Limb],
    negative_product_is_negative: bool,
}

fn recover_scaled_even_odd(pair: &mut PairValues<'_>) {
    SharedEval::exact_half_reverse_difference_in_place(pair.negative, pair.positive);
    SharedEval::sub_full_slices_in_place(pair.positive, pair.negative);
    if pair.negative_product_is_negative {
        swap(&mut pair.positive, &mut pair.negative);
    }
}

fn pack_even_odd(packed: &mut [Limb], other: &[Limb], split_len: usize, even_is_in_other: bool) {
    let value_len = other.len();
    if even_is_in_other {
        assert!(
            split_len <= packed.len() && value_len <= packed.len().saturating_sub(split_len),
            "coupled Toom-8 window cannot contain the shifted point product"
        );
        packed.copy_within(split_len.., 0);
        let (_, high_guard) = packed.split_at_mut(value_len);
        high_guard.fill(0);
        // SAFETY: the release check expresses the required non-wrapping shifted
        // bound directly.
        let _ = unsafe { SharedEval::fused_add_shifted_in_place(packed, other, split_len) };
        return;
    }

    let low_len = min(split_len, other.len());
    let (other_low, other_high) = other.split_at(low_len);
    let (packed_low, _) = packed.split_at_mut(low_len);
    packed_low.copy_from_slice(other_low);
    assert!(
        other_high.is_empty() || (split_len <= packed.len() && other.len() <= packed.len()),
        "coupled Toom-8 window cannot contain the shifted point-product tail"
    );
    // SAFETY: an empty tail returns before inspecting the shift. Otherwise the
    // release check gives `split_len <= packed.len()` and, because
    // `low_len == split_len`, `other_high.len() <= packed.len() - split_len`.
    let _ = unsafe { SharedEval::fused_add_shifted_in_place(packed, other_high, split_len) };
}

fn recover_pair(low: &mut [Limb], signed_difference: &mut [Limb]) {
    SharedEval::exact_half_modular_sum_in_place(signed_difference, low);
    SharedEval::sub_full_slices_in_place(low, signed_difference);
}

fn recover_reverse_pair(low: &mut [Limb], signed_difference: &mut [Limb]) {
    // The middle antisymmetric row has the opposite sign: recover
    // low=(sum-difference)/2 first, then high=sum-low.
    SharedEval::reverse_difference_in_place(signed_difference, low);
    SharedEval::exact_signed_div_power_of_two_in_place(signed_difference, 1);
    SharedEval::sub_full_slices_in_place(low, signed_difference);
}

fn subtract_shifted_bits(dst: &mut [Limb], src: &[Limb], shift_bits: usize) {
    let source_len = src
        .iter()
        .rposition(|limb| *limb != 0)
        .map_or(0, |index| index.wrapping_add(1));
    if source_len == 0 {
        return;
    }
    let limb_shift = shift_bits.div_euclid(LIMB_BITS);
    let inner_shift = shift_bits.rem_euclid(LIMB_BITS);
    let (active_src, _) = src.split_at(source_len);
    let (_, shifted_dst) = dst.split_at_mut(limb_shift);
    // SAFETY: inner_shift < LIMB_BITS ≤ 64, always fits in u32.
    let inner_shift_u32 = unsafe { u32::try_from(inner_shift).unwrap_unchecked() };
    let scalar = 1_usize.wrapping_shl(inner_shift_u32);
    SharedEval::sub_mul_word_in_place(shifted_dst, active_src, scalar);
}

impl Toom8 {
    pub fn sum_and_reverse_difference(forward: &mut [Limb], reversed: &mut [Limb]) {
        debug_assert_eq!(forward.len(), reversed.len(), "paired widths differ");
        // SAFETY: both slices are disjoint interpolation buffers with equal
        // lengths. The kernel reads each pair before writing either destination.
        let (carry, borrow) = unsafe {
            ArchKernels::add_reverse_sub_limbs_unchecked(
                forward.as_mut_ptr(),
                reversed.as_mut_ptr(),
                forward.len(),
            )
        };
        let _ = (carry, borrow);
    }
}
