//! In-place radix-2 DIF/DIT FFT engine for Fermat-ring coefficients.
//!
//! Operates directly on a flat `&mut [Limb]` coefficient matrix,
//! using a single scratch slot for the twiddle multiplication in
//! each butterfly. No heap allocation occurs during the transform.

#![allow(
    unsafe_code,
    reason = "FFT butterfly uses unchecked coefficient access on validated flat buffer indices"
)]

use core::{ptr::from_mut, slice::from_raw_parts_mut};

use super::{ArchKernels, Limb, SsaRing, SsaTransform};

/// Namespace for the Fermat-ring FFT: the butterflies, the matrix addressing
/// they run over, and the transforms that drive them end to end.
///
impl SsaTransform {
    #[allow(
        clippy::too_many_lines,
        reason = "the paired DIF/DIT loops share one proved buffer layout; splitting them would add one-use wrappers and hide the transform invariant"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "Internal algorithm needs all buffers passed explicitly"
    )]
    /// Perform an in-place radix-2 FFT over the Fermat ring
    /// $\mathbb{Z}/(2^n + 1)$.
    ///
    /// `transform_len` coefficients of `cl = SsaRing::coeff_limbs(mod_bits)`
    /// limbs each are stored contiguously in `matrix`. The forward transform
    /// uses decimation-in-frequency and emits bit-reversed frequencies; the
    /// inverse uses decimation-in-time and consumes that order, so neither
    /// direction needs a coefficient permutation.
    ///
    /// # Arguments
    ///
    /// - `matrix`: flat coefficient buffer holding `transform_len * cl` limbs.
    /// - `transform_len`: number of coefficients, a power of two.
    /// - `root_shift`: twiddle factor $\omega = 2^{\text{root\_shift}}$ as a
    ///   power-of-two shift.
    /// - `mod_bits`: Fermat modulus bit width $n$.
    /// - `inverse`: select the inverse DIT transform rather than forward DIF.
    /// - `upper_half_zero`: forward-only; the specialized first DIF stage
    ///   requires the upper coefficient half of `matrix` to be zero.
    /// - `scratch`: staging buffer of at least `cl` limbs for twiddle staging.
    /// - `transpose_scratch`: when `transform_len >= 4` and
    ///   `transpose_scratch.len() >= matrix.len()`, the four-step path uses it
    ///   as a `matrix.len()`-limb transpose target; otherwise it is unused.
    ///
    /// # Safety
    ///
    /// - `matrix` holds `transform_len * cl` initialized, writable limbs.
    /// - `scratch` holds at least `cl` initialized, writable limbs.
    /// - `transform_len` is a power of two.
    /// - `root_shift` is already reduced modulo `2 * mod_bits`, and
    ///   `2 * mod_bits` does not overflow `usize`.
    /// - `matrix`, `scratch`, and `transpose_scratch` are disjoint buffers; in
    ///   particular `scratch` may not alias `matrix` (twiddles are staged in
    ///   `scratch` while `matrix` coefficients are read and overwritten).
    /// - When `upper_half_zero` is true, the upper `transform_len / 2 * cl`
    ///   limbs of `matrix` are zero.
    pub unsafe fn fft_in_place(
        matrix: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        mod_bits: usize,
        inverse: bool,
        upper_half_zero: bool,
        scratch: &mut [Limb],
        transpose_scratch: &mut [Limb],
    ) {
        let cl = SsaRing::coeff_limbs(mod_bits);
        let period = mod_bits.wrapping_mul(2);
        let add_sub_kernel = ArchKernels::selected_add_sub_from_limbs_unchecked();

        if transform_len >= 4 && transpose_scratch.len() >= matrix.len() {
            // SAFETY: inputs validated.
            unsafe {
                return cyclic_fft_4step_in_place(
                    matrix,
                    transform_len,
                    root_shift,
                    mod_bits,
                    inverse,
                    upper_half_zero,
                    scratch,
                    transpose_scratch,
                );
            }
        }

        if !inverse {
            if upper_half_zero {
                let half_len = transform_len >> 1;
                let half_matrix_len = half_len.wrapping_mul(cl);
                let (low_matrix, high_matrix) = matrix.split_at_mut(half_matrix_len);
                debug_assert!(
                    high_matrix.iter().all(|limb| *limb == 0),
                    "the specialized DIF stage requires a zero upper matrix half"
                );
                let mut twiddle_shift = 0_usize;
                for (low_slot, high_slot) in low_matrix
                    .chunks_exact_mut(cl)
                    .zip(high_matrix.chunks_exact_mut(cl))
                {
                    // With high=0, the first DIF butterfly is exactly
                    // (low+high, (low-high)*w) = (low, low*w).
                    if twiddle_shift == 0 {
                        high_slot.copy_from_slice(low_slot);
                    } else {
                        // SAFETY: low_slot and high_slot are disjoint cl-limb
                        // spans and the low coefficient is canonical.
                        unsafe {
                            SsaRing::shift_from(high_slot, low_slot, twiddle_shift, mod_bits);
                        }
                    }
                    twiddle_shift = twiddle_shift.wrapping_add(root_shift);
                    if twiddle_shift >= period {
                        twiddle_shift = twiddle_shift.wrapping_sub(period);
                    }
                }
                // The specialized stage consumed the outermost DIF level, so the two
                // contiguous halves are now independent transforms at twice the root.
                let next_root = SsaRing::reduce_mod_period(root_shift.wrapping_mul(2), period);
                // SAFETY: both halves are complete `half_len * cl` coefficient spans
                // and `scratch` still holds at least `cl` limbs.
                unsafe {
                    fft_recursive_dif(
                        low_matrix,
                        half_len,
                        next_root,
                        mod_bits,
                        scratch,
                        add_sub_kernel,
                    );
                    fft_recursive_dif(
                        high_matrix,
                        half_len,
                        next_root,
                        mod_bits,
                        scratch,
                        add_sub_kernel,
                    );
                }
            } else {
                // SAFETY: the caller guarantees `transform_len * cl` limbs and a
                // `cl`-limb scratch slot.
                unsafe {
                    fft_recursive_dif(
                        matrix,
                        transform_len,
                        SsaRing::reduce_mod_period(root_shift, period),
                        mod_bits,
                        scratch,
                        add_sub_kernel,
                    );
                }
            }
            return;
        }

        // SAFETY: the caller guarantees `transform_len * cl` limbs and a `cl`-limb
        // scratch slot.
        unsafe {
            fft_recursive_dit(
                matrix,
                transform_len,
                SsaRing::reduce_mod_period(root_shift, period),
                mod_bits,
                scratch,
                add_sub_kernel,
            );
        }
    }
}

// ── Forward / Inverse FFT ────────────────────────────────────────────────────

/// Performs an in-place radix-2 FFT over the Fermat ring $\mathbb{Z}/(2^n + 1)$.
///
/// The twiddle factor $\omega = 2^{\text{root\_shift}}$ is a principal root
/// of unity in the ring, where multiplication by $2^k$ is a free bit-shift
/// reduced modulo $2^n + 1$. The forward transform uses DIF and emits
/// bit-reversed frequencies; the inverse uses DIT and consumes that order, so
/// neither direction needs a coefficient permutation.
///
/// # Arguments
/// - `matrix`: flat coefficient buffer with `transform_len * cl` limbs
/// - `transform_len`: number of coefficients (power of 2)
/// - `root_shift`: twiddle factor expressed as a power-of-two shift
/// - `mod_bits`: Fermat ring modulus bit width $n$
/// - `inverse`: selects inverse DIT rather than forward DIF
/// - `upper_half_zero`: whether the forward input's upper coefficient half is zero
/// - `scratch`: temporary buffer of at least `cl` limbs for butterfly staging
///
/// # Safety
/// - `matrix.len() >= transform_len * cl`.
/// - `scratch.len() >= cl`.
/// - `transform_len` is a power of 2.
/// - `upper_half_zero` is used only for the forward transform and proves that
///   the upper half of `matrix` is zero.
#[allow(clippy::too_many_arguments, reason = "FFT requires all parameters")]
#[allow(clippy::too_many_lines, reason = "Unrolled internal complex function")]
unsafe fn cyclic_fft_4step_in_place(
    matrix: &mut [Limb],
    transform_len: usize,
    root_shift: usize,
    mod_bits: usize,
    inverse: bool,
    upper_half_zero: bool,
    scratch: &mut [Limb],
    transpose_scratch: &mut [Limb],
) {
    let cl = SsaRing::coeff_limbs(mod_bits);
    let transform_log = transform_len.trailing_zeros();
    let r_log = transform_log >> 1;
    let c_log = transform_log.wrapping_sub(r_log);
    let r_len = 1_usize << r_log;
    let c_len = 1_usize << c_log;
    let r_cl = r_len.wrapping_mul(cl);
    let c_cl = c_len.wrapping_mul(cl);

    let matrix_ptr = matrix.as_mut_ptr();
    let trans_ptr = transpose_scratch.as_mut_ptr();

    if inverse {
        let c_root_shift = root_shift.wrapping_mul(r_len);
        for r_idx in 0..r_len {
            let start = r_idx.wrapping_mul(c_cl);
            // SAFETY: bounds derived from lengths.
            let row = unsafe { from_raw_parts_mut(matrix_ptr.add(start), c_cl) };
            // SAFETY: inputs validated.
            unsafe {
                SsaTransform::fft_in_place(
                    row,
                    c_len,
                    c_root_shift,
                    mod_bits,
                    true,
                    false,
                    scratch,
                    &mut [],
                );
            }
        }

        let inv_root_shift = mod_bits.wrapping_mul(2).wrapping_sub(root_shift);
        SsaTransform::transpose_inverse_shift_out_of_place(
            matrix,
            transpose_scratch,
            r_len,
            c_len,
            cl,
            mod_bits,
            inv_root_shift,
        );

        let r_root_shift = root_shift.wrapping_mul(c_len);
        for c_idx in 0..c_len {
            let start = c_idx.wrapping_mul(r_cl);
            // SAFETY: bounds derived from lengths.
            let row = unsafe { from_raw_parts_mut(trans_ptr.add(start), r_cl) };
            // SAFETY: inputs validated.
            unsafe {
                SsaTransform::fft_in_place(
                    row,
                    r_len,
                    r_root_shift,
                    mod_bits,
                    true,
                    false,
                    scratch,
                    &mut [],
                );
            }
        }

        SsaTransform::transpose_out_of_place(transpose_scratch, matrix, c_len, r_len, cl);
    } else {
        SsaTransform::transpose_out_of_place(matrix, transpose_scratch, r_len, c_len, cl);

        let r_root_shift = root_shift.wrapping_mul(c_len);
        for c_idx in 0..c_len {
            let start = c_idx.wrapping_mul(r_cl);
            // SAFETY: bounds derived from lengths.
            let row = unsafe { from_raw_parts_mut(trans_ptr.add(start), r_cl) };
            // SAFETY: inputs validated.
            unsafe {
                SsaTransform::fft_in_place(
                    row,
                    r_len,
                    r_root_shift,
                    mod_bits,
                    false,
                    upper_half_zero,
                    scratch,
                    &mut [],
                );
            }
        }

        SsaTransform::transpose_shift_out_of_place(
            transpose_scratch,
            matrix,
            c_len,
            r_len,
            cl,
            mod_bits,
            root_shift,
        );

        let c_root_shift = root_shift.wrapping_mul(r_len);
        for r_idx in 0..r_len {
            let start = r_idx.wrapping_mul(c_cl);
            // SAFETY: bounds derived from lengths.
            let row = unsafe { from_raw_parts_mut(matrix_ptr.add(start), c_cl) };
            // SAFETY: inputs validated.
            unsafe {
                SsaTransform::fft_in_place(
                    row,
                    c_len,
                    c_root_shift,
                    mod_bits,
                    false,
                    false,
                    scratch,
                    &mut [],
                );
            }
        }
    }
}

/// Recursive decimation-in-frequency forward transform.
///
/// Splitting into two contiguous halves after the outermost butterfly stage
/// makes every sub-transform a contiguous sub-slice, so each one that fits a
/// cache level completes all of its remaining stages while resident. That is
/// the same locality the explicit four-step layout buys, obtained at every
/// scale instead of one, and without moving a single coefficient.
///
/// # Safety
/// - `matrix` holds `transform_len * SsaRing::coeff_limbs(mod_bits)` limbs.
/// - `scratch` holds at least `SsaRing::coeff_limbs(mod_bits)` limbs.
/// - `transform_len` is a power of two.
/// - `root_shift` is already reduced modulo `2 * mod_bits`.
unsafe fn fft_recursive_dif(
    matrix: &mut [Limb],
    transform_len: usize,
    root_shift: usize,
    mod_bits: usize,
    scratch: &mut [Limb],
    add_sub_kernel: unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb),
) {
    if transform_len < 2 {
        return;
    }
    let cl = SsaRing::coeff_limbs(mod_bits);
    let period = mod_bits.wrapping_mul(2);
    let half_len = transform_len >> 1;
    let (low_half, high_half) = matrix.split_at_mut(half_len.wrapping_mul(cl));

    let mut twiddle_shift = 0_usize;
    for (low_slot, high_slot) in low_half
        .chunks_exact_mut(cl)
        .zip(high_half.chunks_exact_mut(cl))
    {
        if twiddle_shift == 0 {
            let high_dest = from_mut::<[Limb]>(high_slot);
            let high_source = high_dest.cast::<Limb>().cast_const();
            // SAFETY: low_slot and high_slot are disjoint cl-limb spans, and the
            // exact difference/source alias is the documented permitted case.
            unsafe {
                SsaRing::add_sub(low_slot, high_dest, high_source, mod_bits, add_sub_kernel);
            }
        } else {
            let high_source = high_slot.as_ptr();
            // SAFETY: low_slot, high_slot, and scratch are disjoint cl-limb
            // spans, so the staged difference is read back before the shift
            // overwrites the high coefficient.
            unsafe {
                SsaRing::add_sub(
                    low_slot,
                    from_mut::<[Limb]>(scratch),
                    high_source,
                    mod_bits,
                    add_sub_kernel,
                );
                SsaRing::shift_from(high_slot, scratch, twiddle_shift, mod_bits);
            }
        }
        twiddle_shift = twiddle_shift.wrapping_add(root_shift);
        if twiddle_shift >= period {
            twiddle_shift = twiddle_shift.wrapping_sub(period);
        }
    }

    let next_root = SsaRing::reduce_mod_period(root_shift.wrapping_mul(2), period);
    // SAFETY: each half is a complete `half_len * cl` span and scratch is
    // unchanged, so both recursive calls keep this function's contract.
    unsafe {
        fft_recursive_dif(
            low_half,
            half_len,
            next_root,
            mod_bits,
            scratch,
            add_sub_kernel,
        );
        fft_recursive_dif(
            high_half,
            half_len,
            next_root,
            mod_bits,
            scratch,
            add_sub_kernel,
        );
    }
}

/// Recursive decimation-in-time inverse transform.
///
/// Mirrors [`fft_recursive_dif`]: the two contiguous halves are transformed
/// first and combined afterwards, so the same cache residency applies.
///
/// # Safety
/// Identical to [`fft_recursive_dif`].
unsafe fn fft_recursive_dit(
    matrix: &mut [Limb],
    transform_len: usize,
    root_shift: usize,
    mod_bits: usize,
    scratch: &mut [Limb],
    add_sub_kernel: unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb),
) {
    if transform_len < 2 {
        return;
    }
    let cl = SsaRing::coeff_limbs(mod_bits);
    let period = mod_bits.wrapping_mul(2);
    let half_len = transform_len >> 1;
    let (low_half, high_half) = matrix.split_at_mut(half_len.wrapping_mul(cl));

    let next_root = SsaRing::reduce_mod_period(root_shift.wrapping_mul(2), period);
    // SAFETY: each half is a complete `half_len * cl` span, matching this
    // function's own contract.
    unsafe {
        fft_recursive_dit(
            low_half,
            half_len,
            next_root,
            mod_bits,
            scratch,
            add_sub_kernel,
        );
        fft_recursive_dit(
            high_half,
            half_len,
            next_root,
            mod_bits,
            scratch,
            add_sub_kernel,
        );
    }

    // The inverse twiddle is the forward one negated in the exponent group, and
    // a zero step stays zero because the running shift is reduced below.
    let twiddle_step = period.wrapping_sub(root_shift);
    let mut twiddle_shift = 0_usize;
    for (low_slot, high_slot) in low_half
        .chunks_exact_mut(cl)
        .zip(high_half.chunks_exact_mut(cl))
    {
        let high_dest = from_mut::<[Limb]>(high_slot);
        let shifted_source = if twiddle_shift == 0 {
            high_dest.cast::<Limb>().cast_const()
        } else {
            // SAFETY: high_slot and scratch are disjoint cl-limb spans.
            unsafe {
                SsaRing::shift_from(scratch, high_slot, twiddle_shift, mod_bits);
            }
            scratch.as_ptr()
        };
        // SAFETY: low_slot and high_slot are disjoint cl-limb spans and
        // shifted_source is either the aliased high slot or the staged scratch.
        unsafe {
            SsaRing::add_sub(
                low_slot,
                high_dest,
                shifted_source,
                mod_bits,
                add_sub_kernel,
            );
        }
        twiddle_shift = twiddle_shift.wrapping_add(twiddle_step);
        if twiddle_shift >= period {
            twiddle_shift = twiddle_shift.wrapping_sub(period);
        }
    }
}
