//! Mixed direct/reciprocal interpolation for balanced Toom-Cook 6.

use super::{ArchKernels, Limb, SharedEval, Toom6};

/// The five coupled point pairs consumed by balanced Toom-6 interpolation.
pub struct Values<'buffer> {
    pub one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub four: &'buffer mut [Limb],
    pub half: &'buffer mut [Limb],
    pub quarter: &'buffer mut [Limb],
}

impl Toom6 {
    /// Separate and couple one direct positive/negative point-product pair.
    pub fn couple_direct(
        packed: &mut [Limb],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        zero: &[Limb],
        split_len: usize,
        point_shift: u32,
    ) {
        let value_len = negative.len();
        debug_assert_eq!(
            packed.len(),
            value_len.wrapping_add(split_len),
            "coupled Toom-6 buffer has the wrong width"
        );
        {
            let (_, positive_and_guard) = packed.split_at_mut(split_len);
            let (positive, _) = positive_and_guard.split_at_mut(value_len);
            let mut pair = PairValues {
                positive,
                negative,
                negative_product_is_negative,
            };
            separate_direct_even_odd(&mut pair, zero, point_shift);
        }
        pack_even_odd(packed, negative, split_len);
    }
}

impl Toom6 {
    /// Separate and couple one reciprocal positive/negative point-product pair.
    pub fn couple_reciprocal(
        packed: &mut [Limb],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        zero: &[Limb],
        split_len: usize,
        denominator_shift: u32,
    ) {
        let value_len = negative.len();
        debug_assert_eq!(
            packed.len(),
            value_len.wrapping_add(split_len),
            "coupled Toom-6 buffer has the wrong width"
        );
        {
            let (_, positive_and_guard) = packed.split_at_mut(split_len);
            let (positive, _) = positive_and_guard.split_at_mut(value_len);
            let mut pair = PairValues {
                positive,
                negative,
                negative_product_is_negative,
            };
            separate_reciprocal_even_odd(&mut pair, zero, denominator_shift);
        }
        pack_even_odd(packed, negative, split_len);
    }
}

impl Toom6 {
    /// Separate and couple one direct degree-eleven point-product pair.
    pub fn couple_direct_half(
        packed: &mut [Limb],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        zero: &[Limb],
        infinity: &[Limb],
        split_len: usize,
        point_shift: u32,
    ) {
        let value_len = negative.len();
        debug_assert_eq!(
            packed.len(),
            value_len.wrapping_add(split_len),
            "coupled Toom-6.5 buffer has the wrong width"
        );
        {
            let (_, positive_and_guard) = packed.split_at_mut(split_len);
            let (positive, _) = positive_and_guard.split_at_mut(value_len);
            let mut pair = PairValues {
                positive,
                negative,
                negative_product_is_negative,
            };
            recover_scaled_even_odd(&mut pair);
            SharedEval::sub_full_slices_in_place(pair.positive, zero);
            SharedEval::exact_div_power_of_two_in_place(pair.positive, point_shift.wrapping_mul(2));
            SharedEval::exact_div_power_of_two_in_place(pair.negative, point_shift);
            subtract_shifted_bits(pair.negative, infinity, point_shift.wrapping_mul(10));
        }
        pack_even_odd(packed, negative, split_len);
    }
}

impl Toom6 {
    /// Separate and couple one reciprocal degree-eleven point-product pair.
    pub fn couple_reciprocal_half(
        packed: &mut [Limb],
        negative: &mut [Limb],
        negative_product_is_negative: bool,
        zero: &[Limb],
        infinity: &[Limb],
        split_len: usize,
        denominator_shift: u32,
    ) {
        let value_len = negative.len();
        debug_assert_eq!(
            packed.len(),
            value_len.wrapping_add(split_len),
            "coupled Toom-6.5 buffer has the wrong width"
        );
        {
            let (_, positive_and_guard) = packed.split_at_mut(split_len);
            let (positive, _) = positive_and_guard.split_at_mut(value_len);
            let mut pair = PairValues {
                positive,
                negative,
                negative_product_is_negative,
            };
            recover_scaled_even_odd(&mut pair);
            SharedEval::exact_div_power_of_two_in_place(pair.positive, denominator_shift);
            subtract_shifted_bits(pair.positive, zero, denominator_shift.wrapping_mul(10));
            SharedEval::sub_full_slices_in_place(pair.negative, infinity);
            // Removing c11 leaves d^10*c1 + ... + d^2*c9, with one common
            // z=d^2 factor. Divide it out to obtain z^4*c1 + ... + c9.
            SharedEval::exact_div_power_of_two_in_place(
                pair.negative,
                denominator_shift.wrapping_mul(2),
            );
        }
        pack_even_odd(packed, negative, split_len);
    }
}

impl Toom6 {
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

        SharedEval::add_coefficient_in_place(dst, four, split_len);
        SharedEval::add_coefficient_in_place(dst, two, split_len.wrapping_mul(3));
        SharedEval::add_coefficient_in_place(dst, one, split_len.wrapping_mul(5));
        SharedEval::add_coefficient_in_place(dst, half, split_len.wrapping_mul(7));
        SharedEval::add_coefficient_in_place(dst, quarter, split_len.wrapping_mul(9));
    }
}

impl Toom6 {
    /// Interpolate five packed point pairs in place.
    ///
    /// On return, `four`, `two`, `one`, `half`, and `quarter` respectively hold
    /// the packed coefficient pairs beginning at radix shifts 1, 3, 5, 7, and 9.
    pub fn interpolate_values(values: Values<'_>) {
        let Values {
            one,
            two,
            four,
            half,
            quarter,
        } = values;

        // Each packed table value is O(z)+B^m*E(z). Interpolation is linear, so
        // one table pass recovers c_(2i+1)+B^m*c_(2i+2) for i=0..4.
        interpolate_table(Table {
            at_one: one,
            at_four: two,
            at_sixteen: four,
            reversed_four: half,
            reversed_sixteen: quarter,
        });
    }
}

impl Toom6 {
    /// Add only the coefficient pairs not already placed at shifts 3 and 7.
    pub fn reconstruct_alternating(
        dst: &mut [Limb],
        split_len: usize,
        first: &[Limb],
        fifth: &[Limb],
        ninth: &[Limb],
    ) {
        SharedEval::add_coefficient_in_place(dst, first, split_len);
        SharedEval::add_coefficient_in_place(dst, fifth, split_len.wrapping_mul(5));
        SharedEval::add_coefficient_in_place(dst, ninth, split_len.wrapping_mul(9));
    }
}

struct PairValues<'buffer> {
    positive: &'buffer mut [Limb],
    negative: &'buffer mut [Limb],
    negative_product_is_negative: bool,
}

struct Table<'buffer> {
    at_one: &'buffer mut [Limb],
    at_four: &'buffer mut [Limb],
    at_sixteen: &'buffer mut [Limb],
    reversed_four: &'buffer mut [Limb],
    reversed_sixteen: &'buffer mut [Limb],
}

fn separate_direct_even_odd(pair: &mut PairValues<'_>, zero: &[Limb], point_shift: u32) {
    recover_scaled_even_odd(pair);
    SharedEval::sub_full_slices_in_place(pair.positive, zero);
    SharedEval::exact_div_power_of_two_in_place(pair.positive, point_shift.wrapping_mul(2));
    SharedEval::exact_div_power_of_two_in_place(pair.negative, point_shift);
}

fn separate_reciprocal_even_odd(pair: &mut PairValues<'_>, zero: &[Limb], denominator_shift: u32) {
    recover_scaled_even_odd(pair);
    // Each reciprocal product is d^10*W(+/-1/d), d=2^s. Removing
    // d^10*c0 from its even half yields
    //   d^8*c2 + d^6*c4 + d^4*c6 + d^2*c8 + c10,
    // the reversed even table at d^2. Its odd half is d^9*O(1/d^2);
    // division by d gives the corresponding d^8-scaled reversed table.
    subtract_shifted_bits(pair.positive, zero, denominator_shift.wrapping_mul(10));
    SharedEval::exact_div_power_of_two_in_place(pair.negative, denominator_shift);
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

#[allow(
    unsafe_code,
    reason = "The coupled Toom-6 window retains a split-width prefix before an equal-width point product"
)]
fn pack_even_odd(packed: &mut [Limb], other: &[Limb], split_len: usize) {
    assert!(
        split_len <= other.len() && split_len <= packed.len() && other.len() <= packed.len(),
        "coupled Toom-6 window cannot contain the shifted point product"
    );
    // E already occupies its final high-shifted window. Copy the disjoint low
    // block of O, then accumulate only O's overlapping tail.
    let (other_low, other_high) = other.split_at(split_len);
    let (packed_low, _) = packed.split_at_mut(split_len);
    packed_low.copy_from_slice(other_low);
    // SAFETY: the release check proves `shift = split_len <= packed.len()` and
    // `other_high.len() = other.len() - split_len <= packed.len() - split_len`.
    let _ = unsafe { SharedEval::fused_add_shifted_in_place(packed, other_high, split_len) };
}

fn interpolate_table(table: Table<'_>) {
    let Table {
        at_one,
        at_four,
        at_sixteen,
        reversed_four,
        reversed_sixteen,
    } = table;
    // For P(z)=p0+p1*z+...+p4*z^4, direct evaluation supplies P(1),
    // P(4), P(16), while reciprocal evaluation supplies
    // R(4)=4^4*P(1/4) and R(16)=16^4*P(1/16). Define
    //   A=p0+p4, B=p1+p3, C=p2, D=p4-p0, E=p3-p1.
    // Pair sums isolate A,B,C and pair differences isolate D,E:
    //   (S4-32P1)/9       = 25A+4B                = U
    //   (S16-512P1)/225   = 289A+16B              = V
    //   D4/15             = 17D+4E                = X
    //   D16/255           = 257D+16E              = Y.
    // Hence A=(V-4U)/189, B=(U-25A)/4,
    // D=(Y-4X)/189, and E=(X-17D)/4. Every division is exact by
    // these identities. D and E may be negative, so their halving uses an
    // arithmetic fixed-width shift; recovered p0..p4 are nonnegative.
    sum_and_signed_difference(at_four, reversed_four);
    sum_and_signed_difference(at_sixteen, reversed_sixteen);

    SharedEval::exact_sub_mul_word_odd_in_place(at_four, at_one, 32, 9);
    SharedEval::exact_sub_mul_word_odd_in_place(at_sixteen, at_one, 512, 225);

    SharedEval::exact_sub_mul_word_odd_in_place(at_sixteen, at_four, 4, 189);
    SharedEval::sub_mul_word_in_place(at_four, at_sixteen, 25);
    SharedEval::exact_div_power_of_two_in_place(at_four, 2);
    SharedEval::sub_full_slices_in_place(at_one, at_sixteen);
    SharedEval::sub_full_slices_in_place(at_one, at_four);

    SharedEval::exact_div_radix_minus_one_in_place::<15>(reversed_four);
    SharedEval::exact_div_radix_minus_one_in_place::<255>(reversed_sixteen);
    SharedEval::exact_sub_mul_word_odd_in_place(reversed_sixteen, reversed_four, 4, 189);
    SharedEval::sub_mul_word_in_place(reversed_four, reversed_sixteen, 17);
    SharedEval::exact_signed_div_power_of_two_in_place(reversed_four, 2);

    // Recover p4=(A+D)/2 and p0=A-p4, and likewise p3=(B+E)/2 and
    // p1=B-p3. The modular half-sum discards the sign-extension carry from a
    // negative D or E; each recovered coefficient is proven nonnegative.
    SharedEval::exact_half_modular_sum_in_place(reversed_sixteen, at_sixteen);
    SharedEval::sub_full_slices_in_place(at_sixteen, reversed_sixteen);
    SharedEval::exact_half_modular_sum_in_place(reversed_four, at_four);
    SharedEval::sub_full_slices_in_place(at_four, reversed_four);
}

fn sum_and_signed_difference(forward: &mut [Limb], reversed: &mut [Limb]) {
    debug_assert_eq!(forward.len(), reversed.len(), "paired widths must match");
    // SAFETY: both slices are disjoint interpolation buffers with equal
    // lengths. The kernel reads each pair before writing either destination.
    let (carry, borrow) = unsafe {
        ArchKernels::add_sub_limbs_unchecked(
            forward.as_mut_ptr(),
            reversed.as_mut_ptr(),
            forward.len(),
        )
    };
    // Interpolation is fixed-width: the retained sign guard makes both results
    // valid modulo B^n, so the final carry and borrow are sign extension only.
    let _ = (carry, borrow);
}

fn subtract_shifted_bits(dst: &mut [Limb], src: &[Limb], shift_bits: u32) {
    let source_len = src
        .iter()
        .rposition(|limb| *limb != 0)
        .map_or(0, |index| index.wrapping_add(1));
    if source_len == 0 {
        return;
    }

    // SAFETY: shift_bits ≤ total product bits, quotient always fits in usize.
    let limb_shift =
        unsafe { usize::try_from(shift_bits.div_euclid(Limb::BITS)).unwrap_unchecked() };
    let inner_shift = shift_bits.rem_euclid(Limb::BITS);
    debug_assert!(
        limb_shift.wrapping_add(source_len) < dst.len(),
        "shifted constant product exceeds interpolation guard"
    );
    let (active_src, _) = src.split_at(source_len);
    let (_, shifted_dst) = dst.split_at_mut(limb_shift);
    let scalar = Limb::from(1_u8).wrapping_shl(inner_shift);
    SharedEval::sub_mul_word_in_place(shifted_dst, active_src, scalar);
}
