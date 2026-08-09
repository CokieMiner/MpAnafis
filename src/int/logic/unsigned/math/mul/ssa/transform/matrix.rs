//! Flat coefficient matrix addressing for Fermat-ring FFT.
//!
//! All coefficients are stored contiguously in one `&mut [Limb]` buffer. Each
//! coefficient slot has a fixed width of `coeff_limbs` limbs, eliminating
//! per-element heap allocation entirely.

use super::{Limb, SSA_TRANSPOSE_TILE_LIMBS, SsaRing, SsaTransform};

/// Coefficient addressing and the transposes the four-step layout runs over.
///
/// Contributed to the [`SsaTransform`] namespace declared in
/// [`drive`](super::drive).
impl SsaTransform {
    /// Returns a shared slice of the coefficient at `index`.
    ///
    /// # Safety
    /// `index < transform_len` and `buf.len() >= transform_len * coeff_limbs`.
    #[allow(
        clippy::inline_always,
        reason = "zero-cost pointer arithmetic on the hot FFT path"
    )]
    #[inline(always)]
    pub unsafe fn coeff(buf: &[Limb], index: usize, coeff_limbs: usize) -> &[Limb] {
        let offset = index.wrapping_mul(coeff_limbs);
        // SAFETY: caller guarantees index < transform_len and buf is large enough.
        unsafe { buf.get_unchecked(offset..offset.wrapping_add(coeff_limbs)) }
    }

    /// Returns a mutable slice of the coefficient at `index`.
    ///
    /// # Safety
    /// `index < transform_len` and `buf.len() >= transform_len * coeff_limbs`.
    #[allow(
        clippy::inline_always,
        reason = "zero-cost pointer arithmetic on the hot FFT path"
    )]
    #[inline(always)]
    pub unsafe fn coeff_mut(buf: &mut [Limb], index: usize, coeff_limbs: usize) -> &mut [Limb] {
        let offset = index.wrapping_mul(coeff_limbs);
        // SAFETY: caller guarantees index < transform_len and buf is large enough.
        unsafe { buf.get_unchecked_mut(offset..offset.wrapping_add(coeff_limbs)) }
    }

    /// Out-of-place cache-blocked transpose.
    /// Transposes a matrix of `rows` x `cols` elements.
    /// Each element is a coefficient of exactly `cl` limbs.
    pub fn transpose_out_of_place(
        src: &[Limb],
        dst: &mut [Limb],
        rows: usize,
        cols: usize,
        cl: usize,
    ) {
        transpose_blocked(
            src,
            dst,
            rows,
            cols,
            cl,
            |_| (),
            |dst_coeff, src_coeff, (), _| dst_coeff.copy_from_slice(src_coeff),
        );
    }

    /// Out-of-place cache-blocked transpose combined with Fermat Twiddle shifts.
    /// Transposes a matrix of `rows` x `cols` elements, and applies the twiddle shift.
    /// `cols` MUST be a power of 2.
    pub fn transpose_shift_out_of_place(
        src: &[Limb],
        dst: &mut [Limb],
        rows: usize,
        cols: usize,
        cl: usize,
        mod_bits: usize,
        root_shift: usize,
    ) {
        // Every twiddle exponent carries the factor `root_shift`, so a zero root
        // makes every shift zero and this degenerates to the plain transpose.
        if root_shift == 0 {
            Self::transpose_out_of_place(src, dst, rows, cols, cl);
            return;
        }

        let shift_mask = usize::BITS.wrapping_sub(cols.trailing_zeros());
        let period = mod_bits.wrapping_mul(2);

        // The forward twiddle reverses the *column*, so only `r` hoists per row.
        transpose_blocked(
            src,
            dst,
            rows,
            cols,
            cl,
            |r| r,
            |dst_coeff, src_coeff, &r, c| {
                let k_log = c.reverse_bits() >> shift_mask;
                let twiddle_exponent = r.wrapping_mul(k_log);
                // SAFETY: the driver hands out disjoint source and destination
                // coefficients of exactly `cl` limbs from a semi-normalized matrix.
                unsafe {
                    shift_coefficient(
                        dst_coeff,
                        src_coeff,
                        twiddle_exponent,
                        root_shift,
                        period,
                        mod_bits,
                    );
                }
            },
        );
    }

    /// Out-of-place cache-blocked transpose fused with the inverse four-step
    /// twiddle.
    ///
    /// After the inverse row transforms, source coordinates `(r, c)` move to
    /// destination coordinates `(c, r)` and receive
    /// `root^(c * bit_reverse(r))`. Applying the shift while the coefficient is
    /// already moving avoids the former transpose followed by an in-place shift,
    /// whose aliasing contract required another complete copy through scratch.
    pub fn transpose_inverse_shift_out_of_place(
        src: &[Limb],
        dst: &mut [Limb],
        rows: usize,
        cols: usize,
        cl: usize,
        mod_bits: usize,
        root_shift: usize,
    ) {
        if root_shift == 0 {
            Self::transpose_out_of_place(src, dst, rows, cols, cl);
            return;
        }

        let shift_mask = usize::BITS.wrapping_sub(rows.trailing_zeros());
        let period = mod_bits.wrapping_mul(2);

        // The inverse twiddle reverses the *row*, so the reversal hoists per row and
        // the inner loop is left with one multiply.
        transpose_blocked(
            src,
            dst,
            rows,
            cols,
            cl,
            |r| r.reverse_bits() >> shift_mask,
            |dst_coeff, src_coeff, &reversed_row, c| {
                let twiddle_exponent = c.wrapping_mul(reversed_row);
                // SAFETY: the driver hands out disjoint source and destination
                // coefficients of exactly `cl` limbs from a semi-normalized matrix.
                unsafe {
                    shift_coefficient(
                        dst_coeff,
                        src_coeff,
                        twiddle_exponent,
                        root_shift,
                        period,
                        mod_bits,
                    );
                }
            },
        );
    }
}

// ── Index-to-slice accessors ─────────────────────────────────────────────────

// ── Matrix transpose ─────────────────────────────────────────────────────────

/// Block edge length for cache-blocked transposes, in coefficients.
///
/// `SSA_TRANSPOSE_TILE_LIMBS` budgets the target-sensitive working set. A square
/// tile of edge `e` reads one source block and writes one destination block, so
/// it touches `2 * e * e * cl` limbs. The edge is therefore the square root of
/// the budget over twice the coefficient width, not a plain division: a linear
/// form is dimensionally wrong and overshoots the budget by the edge itself,
/// which for narrow coefficients means tiles tens of times larger than
/// intended. The clamp bounds both very narrow and very wide coefficients.
#[allow(
    clippy::inline_always,
    reason = "constant-folded blocking parameter computed once per transpose"
)]
#[inline(always)]
fn transpose_block_edge(cl: usize) -> usize {
    SSA_TRANSPOSE_TILE_LIMBS
        .div_euclid(cl.max(1).wrapping_mul(2))
        .isqrt()
        .clamp(4, 32)
}

/// The cache-blocked traversal every out-of-place transpose runs.
///
/// The three transposes below differ only in what each coefficient receives on
/// the way across — nothing, the forward four-step twiddle, or the inverse one —
/// so the blocking, the index arithmetic and the one unsafe access live here and
/// each variant supplies only its own per-element work.
///
/// `row_state` runs once per source row and `place` once per coefficient, which
/// is what lets a variant hoist whatever depends on `r` alone out of the inner
/// loop; the inverse twiddle needs exactly that for its bit reversal. Both hooks
/// are monomorphized into the loop, so this costs nothing against the three
/// hand-written copies it replaces.
#[allow(
    clippy::inline_always,
    reason = "the per-element hooks must monomorphize into the blocked loop"
)]
#[inline(always)]
fn transpose_blocked<State, Row, Place>(
    src: &[Limb],
    dst: &mut [Limb],
    rows: usize,
    cols: usize,
    cl: usize,
    mut row_state: Row,
    mut place: Place,
) where
    Row: FnMut(usize) -> State,
    Place: FnMut(&mut [Limb], &[Limb], &State, usize),
{
    let block_edge = transpose_block_edge(cl);
    // Column-major destination stride: advancing `c` by one moves a whole
    // source column, so the multiply hoists out of the inner loop.
    let dst_col_stride = rows.wrapping_mul(cl);
    for r_blk in (0..rows).step_by(block_edge) {
        for c_blk in (0..cols).step_by(block_edge) {
            let r_end = r_blk.wrapping_add(block_edge).min(rows);
            let c_end = c_blk.wrapping_add(block_edge).min(cols);

            for r in r_blk..r_end {
                let state = row_state(r);
                let src_row_offset = r.wrapping_mul(cols);
                let mut src_idx = src_row_offset.wrapping_add(c_blk).wrapping_mul(cl);
                let mut dst_idx = c_blk
                    .wrapping_mul(dst_col_stride)
                    .wrapping_add(r.wrapping_mul(cl));
                for c in c_blk..c_end {
                    // SAFETY: the blocked coordinates stay inside the
                    // `rows * cols` source and transposed destination spans.
                    unsafe {
                        let src_coeff = src.get_unchecked(src_idx..src_idx.wrapping_add(cl));
                        let dst_coeff = dst.get_unchecked_mut(dst_idx..dst_idx.wrapping_add(cl));
                        place(dst_coeff, src_coeff, &state, c);
                    }
                    src_idx = src_idx.wrapping_add(cl);
                    dst_idx = dst_idx.wrapping_add(dst_col_stride);
                }
            }
        }
    }
}

/// Move one coefficient across, applying its Fermat twiddle shift.
///
/// A zero shift is a plain copy: `SsaRing::shift_from` would still be correct, but
/// the copy skips its modular fold, and an exponent of zero is common enough
/// along the first row and column to be worth the branch.
///
/// # Safety
///
/// Inherited from [`SsaRing::shift_from`]: `dst_coeff` and `src_coeff` each hold
/// at least `SsaRing::coeff_limbs(mod_bits)` limbs, they do not overlap, and `src_coeff`
/// is a semi-normalized Fermat residue.
#[allow(
    clippy::inline_always,
    reason = "one element of the blocked transpose inner loop"
)]
#[inline(always)]
unsafe fn shift_coefficient(
    dst_coeff: &mut [Limb],
    src_coeff: &[Limb],
    twiddle_exponent: usize,
    root_shift: usize,
    period: usize,
    mod_bits: usize,
) {
    let twiddle_shift =
        SsaRing::reduce_mod_period(twiddle_exponent.wrapping_mul(root_shift), period);
    if twiddle_shift == 0 {
        dst_coeff.copy_from_slice(src_coeff);
    } else {
        // SAFETY: the caller guarantees the two coefficient spans are disjoint,
        // wide enough for this ring, and that the source is semi-normalized.
        unsafe {
            SsaRing::shift_from(dst_coeff, src_coeff, twiddle_shift, mod_bits);
        }
    }
}
