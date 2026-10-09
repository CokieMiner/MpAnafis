//! Mixed direct/reciprocal interpolation for balanced Toom-Cook 6.
//!
//! # References
//!
//! - Bodrato, M., & Zanoni, A. (2007). *Integer and Polynomial Multiplication:
//!   Towards Optimal Toom-Cook Matrices*. ISSAC 2007, 17–24.
//!   <https://doi.org/10.1145/1277548.1277552>.

#![expect(
    unsafe_code,
    reason = "Packed equal-width point tables retain signed interpolation guards and bounded radix reconstruction offsets"
)]

use core::ptr::copy_nonoverlapping;

use super::{ArchKernels, Limb, LimbOutput, PointShift, SharedEval, Toom6};

/// The five coupled point pairs consumed by balanced Toom-6 interpolation.
pub struct Values<'buffer, Output = Limb> {
    pub one: &'buffer mut [Output],
    pub two: &'buffer mut [Output],
    pub four: &'buffer mut [Output],
    pub half: &'buffer mut [Output],
    pub quarter: &'buffer mut [Output],
}

impl<'buffer, Output: LimbOutput> Values<'buffer, Output> {
    /// Promotes all coupled tables after their point products and low-block writes.
    ///
    /// # Safety
    /// Every element of each table must contain an initialized limb.
    pub unsafe fn assume_init(self) -> Values<'buffer> {
        // SAFETY: the caller proves the complete first-write phase for every
        // disjoint table; each view retains its unique borrow and lifetime.
        unsafe {
            Values {
                one: Output::assume_init_mut(self.one),
                two: Output::assume_init_mut(self.two),
                four: Output::assume_init_mut(self.four),
                half: Output::assume_init_mut(self.half),
                quarter: Output::assume_init_mut(self.quarter),
            }
        }
    }
}

impl Toom6 {
    /// Separate and couple one direct positive/negative point-product pair.
    pub fn couple_direct(
        packed: &mut [impl LimbOutput],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        zero: &[Limb],
        split_len: usize,
        point: PointShift,
    ) {
        #[expect(
            clippy::as_conversions,
            reason = "PointShift has u32 discriminants 0..=2 on every target"
        )]
        let point_shift = point as u32;
        let value_len = negative.len();
        debug_assert_eq!(
            packed.len(),
            value_len.saturating_add(split_len),
            "coupled Toom-6 buffer has the wrong width"
        );
        {
            // SAFETY: packed has 3m+2 limbs; its m-limb prefix leaves the
            // exact initialized 2m+2-limb product, disjoint from negative.
            let positive =
                unsafe { LimbOutput::assume_init_mut(packed.split_at_mut_unchecked(split_len).1) };
            let mut pair = PairValues {
                positive,
                negative,
                negative_product_is_negative,
            };
            recover_scaled_even_odd(&mut pair);
            SharedEval::sub_full_slices_in_place(pair.positive, zero);
            // SAFETY: the direct schedule uses point_shift<=2, hence 2s<=4.
            let twice_shift = unsafe { point_shift.unchecked_mul(2) };
            SharedEval::exact_div_power_of_two_in_place(pair.positive, twice_shift);
            SharedEval::exact_div_power_of_two_in_place(pair.negative, point_shift);
        }
        pack_even_odd(packed, negative, split_len);
    }
    /// Separate and couple one reciprocal positive/negative point-product pair.
    pub fn couple_reciprocal(
        packed: &mut [impl LimbOutput],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        zero: &[Limb],
        split_len: usize,
        point: PointShift,
    ) {
        #[expect(
            clippy::as_conversions,
            reason = "PointShift has u32 discriminants 0..=2 on every target"
        )]
        let denominator_shift = point as u32;
        let value_len = negative.len();
        debug_assert_eq!(
            packed.len(),
            value_len.saturating_add(split_len),
            "coupled Toom-6 buffer has the wrong width"
        );
        {
            // SAFETY: the m-limb prefix leaves exactly the initialized
            // 2m+2-limb point product in this 3m+2-limb packed buffer.
            let positive =
                unsafe { LimbOutput::assume_init_mut(packed.split_at_mut_unchecked(split_len).1) };
            let mut pair = PairValues {
                positive,
                negative,
                negative_product_is_negative,
            };
            recover_scaled_even_odd(&mut pair);
            // For d=2^s, the even half minus d^10*c0 is
            // d^8*c2+...+c10. Dividing the odd half by d likewise leaves
            // d^8*c1+...+c9: the reversed tables at z=d^2.
            subtract_endpoint(pair.positive, zero, point);
            SharedEval::exact_div_power_of_two_in_place(pair.negative, denominator_shift);
        }
        pack_even_odd(packed, negative, split_len);
    }
    /// Separate and couple one direct degree-eleven point-product pair.
    pub fn couple_direct_half(
        packed: &mut [Limb],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        zero: &[Limb],
        infinity: &[Limb],
        split_len: usize,
        point: PointShift,
    ) {
        #[expect(
            clippy::as_conversions,
            reason = "PointShift has u32 discriminants 0..=2 on every target"
        )]
        let point_shift = point as u32;
        let value_len = negative.len();
        debug_assert_eq!(
            packed.len(),
            value_len.saturating_add(split_len),
            "coupled Toom-6.5 buffer has the wrong width"
        );
        {
            // SAFETY: the 3m+2-limb packed window retains its m-limb prefix
            // before an exact initialized 2m+2-limb point product.
            let positive = unsafe { packed.split_at_mut_unchecked(split_len).1 };
            let mut pair = PairValues {
                positive,
                negative,
                negative_product_is_negative,
            };
            recover_scaled_even_odd(&mut pair);
            SharedEval::sub_full_slices_in_place(pair.positive, zero);
            // SAFETY: PointShift bounds s<=2, giving 2s<=4.
            let twice_shift = unsafe { point_shift.unchecked_mul(2) };
            SharedEval::exact_div_power_of_two_in_place(pair.positive, twice_shift);
            SharedEval::exact_div_power_of_two_in_place(pair.negative, point_shift);
            subtract_endpoint(pair.negative, infinity, point);
        }
        pack_even_odd(packed, negative, split_len);
    }
    /// Separate and couple one reciprocal degree-eleven point-product pair.
    pub fn couple_reciprocal_half(
        packed: &mut [Limb],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        zero: &[Limb],
        infinity: &[Limb],
        split_len: usize,
        point: PointShift,
    ) {
        #[expect(
            clippy::as_conversions,
            reason = "PointShift has u32 discriminants 0..=2 on every target"
        )]
        let denominator_shift = point as u32;
        let value_len = negative.len();
        debug_assert_eq!(
            packed.len(),
            value_len.saturating_add(split_len),
            "coupled Toom-6.5 buffer has the wrong width"
        );
        {
            // SAFETY: the m-limb prefix leaves the exact initialized
            // 2m+2-limb point-product tail of this 3m+2-limb packed buffer.
            let positive = unsafe { packed.split_at_mut_unchecked(split_len).1 };
            let mut pair = PairValues {
                positive,
                negative,
                negative_product_is_negative,
            };
            recover_scaled_even_odd(&mut pair);
            SharedEval::exact_div_power_of_two_in_place(pair.positive, denominator_shift);
            subtract_endpoint(pair.positive, zero, point);
            SharedEval::sub_full_slices_in_place(pair.negative, infinity);
            // Removing c11 leaves d^10*c1 + ... + d^2*c9, with one common
            // z=d^2 factor. Divide it out to obtain z^4*c1 + ... + c9.
            // SAFETY: PointShift bounds s<=2, hence 2s<=4.
            let twice_shift = unsafe { denominator_shift.unchecked_mul(2) };
            SharedEval::exact_div_power_of_two_in_place(pair.negative, twice_shift);
        }
        pack_even_odd(packed, negative, split_len);
    }
    /// Interpolate the five coupled point pairs and reconstruct the product.
    pub fn interpolate_and_reconstruct(dst: &mut [Limb], split_len: usize, values: Values<'_>) {
        let Values {
            one,
            two,
            four,
            half,
            quarter,
        } = values;

        Self::interpolate_values(Values {
            one: &mut *one,
            two: &mut *two,
            four: &mut *four,
            half: &mut *half,
            quarter: &mut *quarter,
        });

        // SAFETY: m<=ceil(maximum actual operand length/6), including half
        // shapes. The slice byte bound with >=2-byte limbs puts 9m below usize::MAX.
        let (third, fifth, seventh, ninth) = unsafe {
            (
                split_len.unchecked_mul(3),
                split_len.unchecked_mul(5),
                split_len.unchecked_mul(7),
                split_len.unchecked_mul(9),
            )
        };
        SharedEval::add_coefficient_in_place(dst, four, split_len);
        SharedEval::add_coefficient_in_place(dst, two, third);
        SharedEval::add_coefficient_in_place(dst, one, fifth);
        SharedEval::add_coefficient_in_place(dst, half, seventh);
        SharedEval::add_coefficient_in_place(dst, quarter, ninth);
    }
    /// Interpolate five packed point pairs in place.
    ///
    /// On return, `four`, `two`, `one`, `half`, and `quarter` respectively hold
    /// the packed coefficient pairs beginning at radix shifts 1, 3, 5, 7, and 9.
    pub fn interpolate_values(values: Values<'_>) {
        let Values {
            one: at_one,
            two: at_four,
            four: at_sixteen,
            half: reversed_four,
            quarter: reversed_sixteen,
        } = values;

        // Each packed table value is O(z)+B^m*E(z). Interpolation is linear, so
        // one table pass recovers c_(2i+1)+B^m*c_(2i+2) for i=0..4.
        // For P(z)=p0+...+p4*z^4, direct points supply P(1), P(4), P(16)
        // and reciprocal points supply R(z)=z^4*P(1/z). With A=p0+p4,
        // B=p1+p3, C=p2, D=p4-p0, E=p3-p1, their sums/differences give
        // U=(S4-32P1)/9=25A+4B, V=(S16-512P1)/225=289A+16B,
        // X=D4/15=17D+4E, Y=D16/255=257D+16E. Hence A=(V-4U)/189,
        // B=(U-25A)/4, D=(Y-4X)/189, E=(X-17D)/4. These identities
        // prove every division exact; signed D/E retain sign extension.
        // SAFETY: all four initialized point buffers are mutually disjoint
        // and share their checked packed width. Both kernels read each limb
        // pair before writing either span. Final carries/borrows are only
        // fixed-width sign extension; the signed matrix bound retains a guard.
        unsafe {
            let _ = ArchKernels::add_sub_limbs_unchecked(
                at_four.as_mut_ptr(),
                reversed_four.as_mut_ptr(),
                at_four.len(),
            );
            let _ = ArchKernels::add_sub_limbs_unchecked(
                at_sixteen.as_mut_ptr(),
                reversed_sixteen.as_mut_ptr(),
                at_sixteen.len(),
            );
        }

        SharedEval::exact_sub_mul_word_odd_in_place(at_four, at_one, 32, 9);
        SharedEval::exact_sub_mul_word_odd_in_place(at_sixteen, at_one, 512, 225);
        SharedEval::exact_sub_mul_word_odd_in_place(at_sixteen, at_four, 4, 189);
        SharedEval::sub_mul_word_in_place(at_four, at_sixteen, 25);
        SharedEval::exact_div_power_of_two_in_place(at_four, 2);
        SharedEval::sub_two_full_slices_in_place(at_one, at_sixteen, at_four);

        SharedEval::exact_div_radix_minus_one_in_place::<15>(reversed_four);
        SharedEval::exact_div_radix_minus_one_in_place::<255>(reversed_sixteen);
        SharedEval::exact_sub_mul_word_odd_in_place(reversed_sixteen, reversed_four, 4, 189);
        SharedEval::sub_mul_word_in_place(reversed_four, reversed_sixteen, 17);
        SharedEval::exact_signed_div_power_of_two_in_place(reversed_four, 2);

        // Recover p4=(A+D)/2, p0=A-p4, p3=(B+E)/2, and p1=B-p3.
        // The modular half-sum discards sign extension from negative D/E;
        // all five resulting coefficients are nonnegative.
        SharedEval::exact_half_modular_sum_in_place(reversed_sixteen, at_sixteen);
        SharedEval::sub_full_slices_in_place(at_sixteen, reversed_sixteen);
        SharedEval::exact_half_modular_sum_in_place(reversed_four, at_four);
        SharedEval::sub_full_slices_in_place(at_four, reversed_four);
    }
}

struct PairValues<'buffer> {
    positive: &'buffer mut [Limb],
    negative: &'buffer mut [Limb],
    negative_product_is_negative: bool,
}

fn recover_scaled_even_odd(pair: &mut PairValues<'_>) {
    if pair.negative_product_is_negative {
        // Evaluation routed (N,P) into (packed,temporary). Hence packed
        // becomes (P-N)/2=E and temporary becomes P-E=(P+N)/2=O.
        SharedEval::exact_half_reverse_difference_in_place(pair.positive, pair.negative);
        SharedEval::sub_full_slices_in_place(pair.negative, pair.positive);
    } else {
        // The conventional (P,N) layout yields temporary=(P-N)/2=O and
        // packed=P-O=(P+N)/2=E. In both cases E therefore remains in the
        // packed B^m-shifted product window and O remains in `negative`.
        SharedEval::exact_half_reverse_difference_in_place(pair.negative, pair.positive);
        SharedEval::sub_full_slices_in_place(pair.positive, pair.negative);
    }
}

fn pack_even_odd(packed: &mut [impl LimbOutput], other: &[Limb], split_len: usize) {
    debug_assert!(
        split_len <= other.len() && split_len <= packed.len() && other.len() <= packed.len(),
        "coupled Toom-6 window cannot contain the shifted point product"
    );
    // E already occupies its final high-shifted window. Copy the disjoint low
    // block of O, then accumulate only O's overlapping tail.
    // SAFETY: other spans 2m+2 limbs and packed spans 3m+2 limbs. Their
    // m-limb prefixes exist and the two buffers are disjoint. Only the source
    // prefix must already be initialized; the destination receives its first write.
    let ((other_low, other_high), (packed_low, _)) = unsafe {
        (
            other.split_at_unchecked(split_len),
            packed.split_at_mut_unchecked(split_len),
        )
    };
    // SAFETY: the source prefix contains m initialized limbs and the destination
    // has the identical limb layout. Separate buffers exclude overlap.
    unsafe {
        copy_nonoverlapping(
            other_low.as_ptr(),
            packed_low.as_mut_ptr().cast(),
            split_len,
        );
    }
    // SAFETY: the Toom-6 layout proves `shift = split_len <= packed.len()` and
    // `other_high.len() = other.len() - split_len <= packed.len() - split_len`.
    // The preceding copy initialized the low prefix; E was already initialized
    // in the entire remaining suffix by the conjugate-product writer.
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
        reason = "PointShift's u32 discriminant is 0, 1, or 2 on every pointer width"
    )]
    let exponent = point as u32;
    // SAFETY: PointShift bounds exponent<=2; 10*exponent<=20 fits u32.
    let shift_bits = unsafe { exponent.unchecked_mul(10) };
    // The complete endpoint has <=2m limbs; shifting by <=20 bits consumes
    // at most one extra limb at B>=2^16. The 2m+2-limb point buffer therefore
    // retains at least one borrow guard without a significance scan.
    #[expect(
        clippy::as_conversions,
        reason = "shift_bits<=20 and Limb::BITS>=16 bound the quotient by one on 16-, 32-, and 64-bit targets"
    )]
    let limb_shift = shift_bits.div_euclid(Limb::BITS) as usize;
    // SAFETY: the remainder is below Limb::BITS, so the unit shift fits a limb.
    let scalar = unsafe { 1_usize.unchecked_shl(shift_bits.rem_euclid(Limb::BITS)) };
    // SAFETY: limb_shift<=1<dst.len()=2m+2; its initialized suffix contains
    // the full <=2m-limb endpoint and at least one disjoint guard limb.
    let shifted_dst = unsafe { dst.split_at_mut_unchecked(limb_shift).1 };
    SharedEval::sub_mul_word_in_place(shifted_dst, src, scalar);
}
