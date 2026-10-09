//! Paired power-of-two interpolation for Toom-8 and Toom-8.5.
//!
//! # References
//!
//! - Bodrato, M., & Zanoni, A. (2007). *Integer and Polynomial Multiplication:
//!   Towards Optimal Toom-Cook Matrices*. ISSAC 2007, 17–24.
//!   <https://doi.org/10.1145/1277548.1277552>.

#![expect(
    unsafe_code,
    reason = "Complete packed products retain signed matrix guards; admitted placement bounds all coefficient spans and carry frontiers"
)]

use core::{cmp::min, num::NonZeroUsize, ptr::copy_nonoverlapping};

use super::{
    AddMulKernel, Addition, ArchKernels, Limb, LimbOutput, PointShift, SharedEval,
    TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS, Toom8,
};

pub struct Values<'buffer, Output = Limb> {
    pub one: &'buffer mut [Output],
    pub two: &'buffer mut [Output],
    pub four: &'buffer mut [Output],
    pub eight: &'buffer mut [Output],
    pub half: &'buffer mut [Output],
    pub quarter: &'buffer mut [Output],
    pub eighth: &'buffer mut [Output],
}

pub struct CouplingContext<'value> {
    pub zero: &'value [Limb],
    pub infinity: &'value [Limb],
    pub split_len: NonZeroUsize,
    pub degree: usize,
}

impl<'buffer, Output: LimbOutput> Values<'buffer, Output> {
    /// Promotes all coupled tables after their point products and low-block writes.
    ///
    /// # Safety
    /// Every element of each table must contain an initialized limb.
    pub unsafe fn assume_init(self) -> Values<'buffer> {
        // SAFETY: the caller proves complete initialization of each disjoint
        // table; the new views preserve all exclusive borrows and lifetimes.
        unsafe {
            Values {
                one: Output::assume_init_mut(self.one),
                two: Output::assume_init_mut(self.two),
                four: Output::assume_init_mut(self.four),
                eight: Output::assume_init_mut(self.eight),
                half: Output::assume_init_mut(self.half),
                quarter: Output::assume_init_mut(self.quarter),
                eighth: Output::assume_init_mut(self.eighth),
            }
        }
    }
}

impl Toom8 {
    pub fn couple_direct(
        packed: &mut [impl LimbOutput],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        context: &CouplingContext<'_>,
        point: PointShift,
    ) {
        #[expect(
            clippy::as_conversions,
            reason = "PointShift's u32 discriminants are exactly 0..=3 on every target"
        )]
        let point_shift = point as u32;
        // SAFETY: packed spans 3m+g limbs; its m-limb prefix leaves exactly
        // the initialized 2m+g-limb point product, disjoint from negative.
        let positive = unsafe {
            LimbOutput::assume_init_mut(packed.split_at_mut_unchecked(context.split_len.get()).1)
        };
        let mut pair = PairValues {
            positive,
            negative,
            negative_product_is_negative,
        };
        recover_scaled_even_odd(&mut pair);
        SharedEval::sub_full_slices_in_place(pair.positive, context.zero);
        // SAFETY: the direct schedule has point_shift<=3, giving twice_shift<=6.
        let twice_shift = unsafe { point_shift.unchecked_mul(2) };
        SharedEval::exact_signed_div_power_of_two_in_place(pair.positive, twice_shift);
        SharedEval::exact_signed_div_power_of_two_in_place(pair.negative, point_shift);
        if context.degree == 15 {
            subtract_endpoint(pair.negative, context.infinity, point);
        }
        pack_even_odd(packed, negative, context.split_len.get());
    }

    pub fn couple_reciprocal(
        packed: &mut [impl LimbOutput],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        context: &CouplingContext<'_>,
        point: PointShift,
    ) {
        #[expect(
            clippy::as_conversions,
            reason = "PointShift's u32 discriminants are exactly 0..=3 on every target"
        )]
        let denominator_shift = point as u32;
        // SAFETY: the m-limb prefix leaves the exact initialized 2m+g-limb
        // point-product tail of a 3m+g-limb packed window.
        let positive = unsafe {
            LimbOutput::assume_init_mut(packed.split_at_mut_unchecked(context.split_len.get()).1)
        };
        let mut pair = PairValues {
            positive,
            negative,
            negative_product_is_negative,
        };
        recover_scaled_even_odd(&mut pair);
        if context.degree == 14 {
            subtract_endpoint(pair.positive, context.zero, point);
            SharedEval::exact_signed_div_power_of_two_in_place(pair.negative, denominator_shift);
        } else {
            SharedEval::exact_signed_div_power_of_two_in_place(pair.positive, denominator_shift);
            subtract_endpoint(pair.positive, context.zero, point);
            SharedEval::sub_full_slices_in_place(pair.negative, context.infinity);
            // SAFETY: denominator_shift<=3 gives twice_shift<=6.
            let twice_shift = unsafe { denominator_shift.unchecked_mul(2) };
            SharedEval::exact_signed_div_power_of_two_in_place(pair.negative, twice_shift);
        }
        pack_even_odd(packed, negative, context.split_len.get());
    }
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

        // SAFETY: m<=ceil(maximum actual operand length/8). The slice byte
        // bound and >=2-byte limbs put every offset through 13m below usize::MAX.
        let (third, fifth, seventh, ninth, eleventh, thirteenth) = unsafe {
            (
                split_len.unchecked_mul(3),
                split_len.unchecked_mul(5),
                split_len.unchecked_mul(7),
                split_len.unchecked_mul(9),
                split_len.unchecked_mul(11),
                split_len.unchecked_mul(13),
            )
        };
        SharedEval::add_coefficient_in_place(dst, eighth, split_len);
        SharedEval::add_coefficient_in_place(dst, half, third);
        SharedEval::add_coefficient_in_place(dst, quarter, fifth);
        SharedEval::add_coefficient_in_place(dst, one, seventh);
        SharedEval::add_coefficient_in_place(dst, two, ninth);
        SharedEval::add_coefficient_in_place(dst, four, eleventh);
        SharedEval::add_coefficient_in_place(dst, eight, thirteenth);
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
        Self::add_mul_signed_in_place::<1_028, true>(quarter, half, temporary, add_mul_kernel);
        Self::exact_sub_mul_two_u64_odd_in_place::<1_300, 1_052_688, 48_070_897_875>(
            eighth,
            quarter,
            half,
            temporary,
            add_mul_kernel,
        );
        Self::exact_sub_mul_u64_odd_in_place::<12_567_555, 2_835>(
            quarter,
            eighth,
            temporary,
            add_mul_kernel,
        );
        SharedEval::exact_signed_div_power_of_two_in_place(quarter, 6);
        Self::add_mul_signed_in_place::<4_095, true>(half, eighth, temporary, add_mul_kernel);
        Self::add_mul_signed_in_place::<240, false>(half, quarter, temporary, add_mul_kernel);
        Self::exact_signed_div_u64::<1_020>(half);

        // Solve the symmetric rows after removing the central packed coefficient.
        Self::add_mul_signed_in_place::<128, true>(two, one, temporary, add_mul_kernel);
        Self::add_mul_signed_in_place::<8_192, true>(four, one, temporary, add_mul_kernel);
        Self::add_mul_signed_in_place::<400, true>(four, two, temporary, add_mul_kernel);
        Self::add_mul_signed_in_place::<524_288, true>(eight, one, temporary, add_mul_kernel);
        Self::add_mul_signed_in_place::<1_428, true>(eight, four, temporary, add_mul_kernel);
        Self::exact_sub_mul_u64_odd_in_place::<112_896, 46_591_793_325>(
            eight,
            two,
            temporary,
            add_mul_kernel,
        );
        Self::exact_sub_mul_u64_odd_in_place::<15_181_425, 42_525>(
            four,
            eight,
            temporary,
            add_mul_kernel,
        );
        SharedEval::exact_signed_div_power_of_two_in_place(four, 4);
        Self::add_mul_signed_in_place::<3_969, true>(two, eight, temporary, add_mul_kernel);
        Self::exact_sub_mul_u64_odd_in_place::<900, 9>(two, four, temporary, add_mul_kernel);
        SharedEval::exact_signed_div_power_of_two_in_place(two, 4);
        SharedEval::sub_three_full_slices_in_place(one, eight, two, four);

        SharedEval::exact_half_modular_sum_in_place(half, four);
        SharedEval::sub_full_slices_in_place(four, half);
        // The middle antisymmetric row has the opposite sign: recover
        // low=(sum-difference)/2 first, then high=sum-low.
        SharedEval::reverse_difference_in_place(quarter, two);
        SharedEval::exact_signed_div_power_of_two_in_place(quarter, 1);
        SharedEval::sub_full_slices_in_place(two, quarter);
        SharedEval::exact_half_modular_sum_in_place(eighth, eight);
        SharedEval::sub_full_slices_in_place(eight, eighth);
    }

    /// Add coefficient pairs not already placed at shifts 3, 7, and 11.
    pub fn reconstruct_alternating(
        dst: &mut [Limb],
        split_len: usize,
        coefficients: [&[Limb]; 4],
        fast_paired_add: bool,
    ) {
        let [first, fifth, ninth, thirteenth] = coefficients;
        // For coefficient widths meeting TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS,
        // dual-stream addition kernels fuse paired coefficient additions into a single pass.
        if fast_paired_add && first.len() >= TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS {
            reconstruct_alternating_paired(dst, split_len, first, fifth, ninth, thirteenth);
            return;
        }
        // SAFETY: actual admitted limb slices bound m by ceil(maximum length/8);
        // the byte-length limit with >=2-byte limbs leaves room for 13m in usize.
        let (fifth_offset, ninth_offset, thirteenth_offset) = unsafe {
            (
                split_len.unchecked_mul(5),
                split_len.unchecked_mul(9),
                split_len.unchecked_mul(13),
            )
        };
        SharedEval::add_coefficient_in_place(dst, first, split_len);
        SharedEval::add_coefficient_in_place(dst, fifth, fifth_offset);
        SharedEval::add_coefficient_in_place(dst, ninth, ninth_offset);
        SharedEval::add_coefficient_in_place(dst, thirteenth, thirteenth_offset);
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
    // SAFETY: placement requires packed_len=3m+g<=4m, hence g<=m. Admitted
    // operands each exceed 7m, so dst.len()>=14m+2. Therefore the complete
    // ninth span ends at 12m+g<=13m<dst.len(), and all three offsets fit.
    let (fifth_shift, ninth_shift, thirteenth_shift, after_first, after_ninth, thirteenth_len) = unsafe {
        let ninth_offset = split_len.unchecked_mul(9);
        let thirteenth_offset = split_len.unchecked_mul(13);
        (
            split_len.unchecked_mul(5),
            ninth_offset,
            thirteenth_offset,
            split_len.unchecked_add(packed_len),
            ninth_offset.unchecked_add(packed_len),
            dst.len().unchecked_sub(thirteenth_offset),
        )
    };
    debug_assert!(
        after_ninth <= dst.len(),
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
    propagate_coefficient_carry(dst, after_first, first_carry);
    propagate_coefficient_carry(dst, after_ninth, ninth_carry);

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
    // SAFETY: paired_len<=packed_len=fifth.len(); 5m+packed_len<=9m fits
    // inside dst. The source tail and its destination are therefore bounded.
    let (fifth_tail, fifth_tail_shift, after_fifth, after_thirteenth) = unsafe {
        (
            fifth.split_at_unchecked(paired_len).1,
            fifth_shift.unchecked_add(paired_len),
            fifth_shift.unchecked_add(packed_len),
            thirteenth_shift.unchecked_add(paired_len),
        )
    };
    let tail_carry = if fifth_tail.is_empty() {
        0
    } else {
        // SAFETY: fifth_tail_shift<=5m+packed_len<=9m<dst.len(). The
        // remaining destination contains fifth_tail and its carry frontier.
        let tail_dst = unsafe { dst.split_at_mut_unchecked(fifth_tail_shift).1 };
        Addition::add_slice_in_place(tail_dst, fifth_tail)
    };
    propagate_coefficient_carry(dst, after_fifth, tail_carry);
    propagate_coefficient_carry(dst, fifth_tail_shift, fifth_carry);
    propagate_coefficient_carry(dst, after_thirteenth, thirteenth_carry);
}

fn propagate_coefficient_carry(dst: &mut [Limb], start: usize, mut carry: Limb) {
    // SAFETY: every caller supplies the end of a bounded coefficient span,
    // proven <=dst.len() by placement or by its min-clipped highest span.
    let suffix = unsafe { dst.split_at_mut_unchecked(start).1 };
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
    if pair.negative_product_is_negative {
        // Evaluation routed (N,P) into (packed,temporary). E=(P-N)/2
        // remains in packed, and O=P-E=(P+N)/2 remains in temporary.
        SharedEval::exact_half_reverse_difference_in_place(pair.positive, pair.negative);
        SharedEval::sub_full_slices_in_place(pair.negative, pair.positive);
    } else {
        // The conventional (P,N) route yields temporary O=(P-N)/2,
        // followed by packed E=P-O=(P+N)/2. Both are nonnegative.
        SharedEval::exact_half_reverse_difference_in_place(pair.negative, pair.positive);
        SharedEval::sub_full_slices_in_place(pair.positive, pair.negative);
    }
}

fn pack_even_odd(packed: &mut [impl LimbOutput], other: &[Limb], split_len: usize) {
    // E already occupies packed[m..]; write O's disjoint low block and then
    // add O's remaining tail. No sign requires a full product move or zero fill.
    // SAFETY: other.len()=2m+g and packed.len()=3m+g, with m>=1. Both
    // disjoint buffers contain their m-limb prefixes; only the source must
    // already be initialized because the destination receives its first write.
    let ((other_low, other_high), (packed_low, _)) = unsafe {
        (
            other.split_at_unchecked(split_len),
            packed.split_at_mut_unchecked(split_len),
        )
    };
    // SAFETY: the source contains m initialized limbs, the destination has the
    // exact limb layout, and the packed/temporary prefixes cannot overlap.
    unsafe {
        copy_nonoverlapping(
            other_low.as_ptr(),
            packed_low.as_mut_ptr().cast(),
            split_len,
        );
    }
    debug_assert!(
        other_high.is_empty() || (split_len <= packed.len() && other.len() <= packed.len()),
        "coupled Toom-8 window cannot contain the shifted point-product tail"
    );
    // SAFETY: other_high.len()=m+g<=packed.len()-m=2m+g. Its addition at m
    // reads the initialized E window, disjoint from every source limb. The copy
    // above completed initialization of the entire packed destination.
    let _ = unsafe {
        SharedEval::fused_add_shifted_in_place(
            LimbOutput::assume_init_mut(packed),
            other_high,
            split_len,
        )
    };
}

fn subtract_endpoint(dst: &mut [Limb], src: &[Limb], point: PointShift) {
    #[expect(
        clippy::as_conversions,
        reason = "PointShift has u32 discriminants 0..=3 on every pointer width"
    )]
    let exponent = point as u32;
    // SAFETY: PointShift bounds exponent<=3, hence 14*exponent<=42 fits u32.
    let shift_bits = unsafe { exponent.unchecked_mul(14) };
    // The complete endpoint occupies <=2m limbs, and shift_bits<=42 consumes
    // at most two whole limbs at B>=2^16. The product's guard width is
    // two evaluation guards: four limbs at 16 bits and two otherwise.
    // It therefore contains the full shifted endpoint and a borrow guard.
    #[expect(
        clippy::as_conversions,
        reason = "shift_bits<=42 and Limb::BITS>=16 give a quotient at most two on every target"
    )]
    let limb_shift = shift_bits.div_euclid(Limb::BITS) as usize;
    let inner_shift = shift_bits.rem_euclid(Limb::BITS);
    // SAFETY: shift_bits<=42 gives limb_shift<=2<dst.len()=2m+g. The
    // remainder is <LIMB_BITS and fits u32; its unit shift fits one limb.
    let (shifted_dst, scalar) = unsafe {
        (
            dst.split_at_mut_unchecked(limb_shift).1,
            1_usize.unchecked_shl(inner_shift),
        )
    };
    SharedEval::sub_mul_word_in_place(shifted_dst, src, scalar);
}
