//! Quadratic-residue screening before exact square-root evaluation.
//!
//! The low byte rejects non-residues modulo 256. Survivors are folded modulo
//! `M = 2^48-1`, then screened modulo all nine coprime prime-power factors of M.
//! Three limbs span an integral number of 48-bit periods on 16-, 32-, and
//! 64-bit targets. A three-limb accumulator retains its cyclic carries during
//! the scan; shifts and end-around reductions are required only after the scan.

#![expect(
    unsafe_code,
    reason = "bounded Mersenne sums and immutable factor tables prove exact arithmetic and residue indices"
)]

use super::{DoubleLimb, InternalMpUint, Limb};

/// Quadratic residues modulo 256, one bit per class, low word first.
const RESIDUES_MOD_256: [u64; 4] = [
    0x0202_0212_0203_0213,
    0x0202_0212_0202_0213,
    0x0202_0212_0203_0212,
    0x0202_0212_0202_0212,
];

/// `9*5*7*13*17*97*241*257*673 = 2^48-1`.
const SCREEN_BITS: u32 = 48;
const SCREEN_MASK: u64 = (1_u64 << SCREEN_BITS) - 1;
const FIRST_SHIFT: u32 = Limb::BITS.rem_euclid(SCREEN_BITS);
const SECOND_SHIFT: u32 = FIRST_SHIFT * 2 % SCREEN_BITS;
const FIRST_SPLIT: u32 = SCREEN_BITS - FIRST_SHIFT;
const SECOND_SPLIT: u32 = SCREEN_BITS - SECOND_SHIFT;
const RESIDUE_WORDS: usize = 673_usize.div_ceil(64);

/// Inverse and permuted quadratic-residue masks for one odd factor of M.
struct ResidueFilter {
    factor: u64,
    inverse: u64,
    words: [u64; RESIDUE_WORDS],
}

const FILTERS: [ResidueFilter; 9] = [
    ResidueFilter::new(9),
    ResidueFilter::new(5),
    ResidueFilter::new(7),
    ResidueFilter::new(13),
    ResidueFilter::new(17),
    ResidueFilter::new(97),
    ResidueFilter::new(241),
    ResidueFilter::new(257),
    ResidueFilter::new(673),
];

/// Namespace for residue screening and integer root operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Roots;

impl Roots {
    /// Returns `false` only when `value` is provably not a perfect square.
    #[must_use]
    pub fn may_be_square(value: &InternalMpUint) -> bool {
        let limbs = value.limbs();
        let Some(&first_limb) = limbs.first() else {
            return true;
        };
        let low_byte = first_limb.to_le_bytes()[0];
        let word_index = usize::from(low_byte >> 6);
        // SAFETY: low_byte>>6<=3 indexes the four initialized residue words.
        let word = unsafe { *RESIDUES_MOD_256.get_unchecked(word_index) };
        if word & (1_u64 << (low_byte & 63)) == 0 {
            return false;
        }

        let (groups, tail) = limbs.as_chunks::<3>();
        let (batches, remaining_groups) = groups.as_chunks::<3>();
        let mut lower_sum = DoubleLimb::MIN;
        let mut high_sum = 0_usize;
        let mut cyclic_carry = 0_usize;
        // Three complete periods share one loop bound and pointer update.
        // The fixed inner array makes all three group updates unconditional.
        for batch in batches {
            for &[low_limb, middle_limb, high_limb] in batch {
                // SAFETY: widening two native limbs exactly fills DoubleLimb.
                let pair = unsafe {
                    DoubleLimb::try_from(low_limb).unwrap_unchecked()
                        | (DoubleLimb::try_from(middle_limb).unwrap_unchecked() << Limb::BITS)
                };
                let (lower, lower_carry) = lower_sum.overflowing_add(pair);
                let (high, carry) = high_sum.carrying_add(high_limb, lower_carry);
                lower_sum = lower;
                high_sum = high;
                // SAFETY: at most one carry per three-limb group bounds the
                // count below ceil(limbs.len()/3)<isize::MAX<Limb::MAX.
                cyclic_carry = unsafe { cyclic_carry.unchecked_add(Limb::from(carry)) };
            }
        }
        for &[low_limb, middle_limb, high_limb] in remaining_groups {
            // Each group has weight one because B^3=1 modulo M. Propagate
            // its lower carry into the third limb, retaining only the final
            // cyclic carry rather than three independent carry counters.
            // SAFETY: each native limb fits the twice-wide unsigned type;
            // shifting the second limb by Limb::BITS fills its high half.
            let pair = unsafe {
                DoubleLimb::try_from(low_limb).unwrap_unchecked()
                    | (DoubleLimb::try_from(middle_limb).unwrap_unchecked() << Limb::BITS)
            };
            let (lower, lower_carry) = lower_sum.overflowing_add(pair);
            let (high, carry) = high_sum.carrying_add(high_limb, lower_carry);
            lower_sum = lower;
            high_sum = high;
            // SAFETY: one carry per group is bounded by ceil(limbs.len()/3),
            // below isize::MAX and hence below Limb::MAX on every pointer width.
            cyclic_carry = unsafe { cyclic_carry.unchecked_add(Limb::from(carry)) };
        }
        if let Some((&low_limb, rest)) = tail.split_first() {
            // SAFETY: both tail limbs fit the twice-wide type. The optional
            // second limb genuinely is absent when the tail has length one.
            let pair = unsafe {
                DoubleLimb::try_from(low_limb).unwrap_unchecked()
                    | (DoubleLimb::try_from(rest.first().copied().unwrap_or(0)).unwrap_unchecked()
                        << Limb::BITS)
            };
            let (lower, lower_carry) = lower_sum.overflowing_add(pair);
            let (high, carry) = high_sum.overflowing_add(Limb::from(lower_carry));
            lower_sum = lower;
            high_sum = high;
            // SAFETY: the tail adds at most one carry to the same bounded count.
            cyclic_carry = unsafe { cyclic_carry.unchecked_add(Limb::from(carry)) };
        }
        #[expect(
            clippy::as_conversions,
            reason = "truncating DoubleLimb to Limb isolates each half; native usize limbs fit u64 on 16-, 32-, and 64-bit targets"
        )]
        #[cfg_attr(
            not(target_pointer_width = "16"),
            expect(
                clippy::cast_possible_truncation,
                reason = "The low cast extracts one limb modulo B; the shifted high half is below B on every supported target"
            )
        )]
        // DoubleLimb holds exactly two native limbs. Truncation selects the low
        // half; shifting by Limb::BITS selects the high half. Each fits u64.
        let (low, middle) = (
            (lower_sum as Limb) as u64,
            ((lower_sum >> Limb::BITS) as Limb) as u64,
        );
        #[expect(
            clippy::as_conversions,
            reason = "native usize limbs fit u64 on 16-, 32-, and 64-bit targets"
        )]
        let (high, carry) = (high_sum as u64, cyclic_carry as u64);

        // For 0<=s<48, x*2^s is congruent to the shifted low (48-s)
        // bits plus the remaining high bits. The cyclic carry has weight
        // B^3=1 modulo M. All four weighted terms are below 2M.
        // SAFETY: every masked-and-shifted part and every high part is
        // below 2^48; four terms have sum<8*2^48<2^51 on every limb width.
        let sum = unsafe {
            (low & SCREEN_MASK)
                .unchecked_add(low >> SCREEN_BITS)
                .unchecked_add((middle & ((1_u64 << FIRST_SPLIT) - 1)) << FIRST_SHIFT)
                .unchecked_add(middle >> FIRST_SPLIT)
                .unchecked_add((high & ((1_u64 << SECOND_SPLIT) - 1)) << SECOND_SHIFT)
                .unchecked_add(high >> SECOND_SPLIT)
                .unchecked_add(carry & SCREEN_MASK)
                .unchecked_add(carry >> SCREEN_BITS)
        };
        // SAFETY: sum<2^51 gives a first fold<=M+7. A second restores
        // residue<=M, retaining M as the cyclic representative of zero.
        let residue = unsafe {
            let folded = (sum & SCREEN_MASK).unchecked_add(sum >> SCREEN_BITS);
            (folded & SCREEN_MASK).unchecked_add(folded >> SCREEN_BITS)
        };

        for filter in &FILTERS {
            // q*factor = residue + class*2^48. Since factor divides 2^48-1,
            // residue=-class (mod factor). The precomputed masks use this
            // permutation; no remainder or division is needed at runtime.
            let quotient = residue.wrapping_mul(filter.inverse) & SCREEN_MASK;
            // SAFETY: quotient<2^48 and factor<=673<2^10 give product<2^58.
            let product = unsafe { quotient.unchecked_mul(filter.factor) };
            // SAFETY: class=product>>48<factor<=673, so product>>54<11.
            // The word index fits usize and the eleven initialized masks.
            let mask = unsafe {
                let index = usize::try_from(product >> 54).unwrap_unchecked();
                *filter.words.get_unchecked(index)
            };
            if mask & (1_u64 << ((product >> SCREEN_BITS) & 63)) == 0 {
                return false;
            }
        }
        true
    }
}

impl ResidueFilter {
    /// Constructs a factor inverse and masks indexed by the negated residue.
    const fn new(factor: u64) -> Self {
        assert!(
            factor >= 3 && factor <= 673 && factor & 1 != 0,
            "residue factors are odd and bounded by the largest factor 673"
        );
        assert!(
            SCREEN_MASK.rem_euclid(factor) == 0,
            "each residue factor divides the Mersenne screening modulus"
        );
        let mut inverse = 1_u64;
        let mut bits = 1_u32;
        while bits < u64::BITS {
            // Newton doubling is in Z/(2^64); its low 48 bits invert factor.
            inverse = inverse.wrapping_mul(2_u64.wrapping_sub(factor.wrapping_mul(inverse)));
            // SAFETY: bits is a power of two below 64, so 2*bits<=64.
            bits = unsafe { bits.unchecked_mul(2) };
        }
        let mut words = [0_u64; RESIDUE_WORDS];
        let mut root = 0_u64;
        while root < factor {
            // SAFETY: root<factor<=673 gives root^2<2^20.
            let square = unsafe { root.unchecked_mul(root) }.rem_euclid(factor);
            // SAFETY: square<factor bounds the exact nonnegative difference.
            let class = unsafe { factor.unchecked_sub(square) }.rem_euclid(factor);
            #[expect(
                clippy::as_conversions,
                reason = "class<673 gives a word index below 11 on all pointer widths"
            )]
            let index = (class >> 6) as usize;
            // SAFETY: index<RESIDUE_WORDS bounds the initialized array slot;
            // this constant builder has its sole mutable access to the table.
            unsafe {
                *words.as_mut_ptr().add(index) |= 1_u64 << (class & 63);
            }
            // SAFETY: root<factor<=673 gives root+1<=673 without overflow.
            root = unsafe { root.unchecked_add(1) };
        }
        Self {
            factor,
            inverse: inverse & SCREEN_MASK,
            words,
        }
    }
}
