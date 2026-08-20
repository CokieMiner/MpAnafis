//! Subquadratic Half-GCD (HGCD) divide-and-conquer implementation.

use core::{
    cmp::Ordering,
    mem::{replace, swap},
};

use alloc::vec::Vec;

use super::{DivScratch, Division, Gcd, HgcdMatrix, InternalMpUint, Limb};

impl Gcd {
    /// Threshold for switching from quadratic Lehmer to subquadratic Half-GCD.
    pub const HGCD_THRESHOLD: usize = 64;

    /// Maximum recursion depth allowed for Half-GCD to strictly prevent stack overflow.
    pub const MAX_HGCD_DEPTH: usize = 16;

    /// Reduces `(u, v)` in place using Half-GCD to approximately half of `u`'s length.
    pub fn hgcd_reduce(
        u: &mut InternalMpUint,
        v: &mut InternalMpUint,
        scratch: &mut DivScratch,
        u_backup: &mut Vec<Limb>,
        v_backup: &mut Vec<Limb>,
    ) {
        let mut u_len = u.limbs().len();
        let mut v_len = v.limbs().len();
        if u_len == 0 || v_len == 0 {
            return;
        }

        match (*u).cmp(v) {
            Ordering::Equal => {
                *v = InternalMpUint::zero();
                return;
            }
            Ordering::Less => {
                swap(u, v);
            }
            Ordering::Greater => {}
        }

        u_len = u.limbs().len();
        v_len = v.limbs().len();
        let target_len = (u_len.wrapping_add(1)) >> 1;
        if v_len <= target_len {
            return;
        }

        if u_len <= Self::HGCD_THRESHOLD {
            Self::hgcd_lehmer_in_place(u, v, target_len, scratch, u_backup, v_backup);
            return;
        }

        let k = target_len >> 1;
        if k == 0 {
            Self::hgcd_lehmer_in_place(u, v, target_len, scratch, u_backup, v_backup);
            return;
        }

        let a1 = u
            .limbs()
            .get(k..)
            .map_or_else(InternalMpUint::zero, InternalMpUint::from_limbs_slice);
        let b1 = v
            .limbs()
            .get(k..)
            .map_or_else(InternalMpUint::zero, InternalMpUint::from_limbs_slice);

        let r_mat = Self::hgcd_matrix(&a1, &b1, scratch);
        let applied = r_mat.apply_to_pair(u, v);
        if !applied || v.limbs().len() <= target_len || v.is_zero() {
            return;
        }

        let mut q = InternalMpUint::zero();
        let mut rem = InternalMpUint::zero();
        Division::div_rem_into(u, v, &mut q, &mut rem, scratch);
        swap(u, v);
        swap(v, &mut rem);

        if v.limbs().len() <= target_len || v.is_zero() {
            return;
        }

        let cur_u_len = u.limbs().len();
        if cur_u_len > target_len {
            let k2 = (cur_u_len.wrapping_sub(target_len)) >> 1;
            if k2 > 0 {
                let a2 = u
                    .limbs()
                    .get(k2..)
                    .map_or_else(InternalMpUint::zero, InternalMpUint::from_limbs_slice);
                let b2 = v
                    .limbs()
                    .get(k2..)
                    .map_or_else(InternalMpUint::zero, InternalMpUint::from_limbs_slice);

                if a2.limbs().len() < cur_u_len {
                    let s_mat = Self::hgcd_matrix(&a2, &b2, scratch);
                    let _ = s_mat.apply_to_pair(u, v);
                }
            }
        }
    }

    /// Computes the Half-GCD transition matrix `M` with default maximum recursion depth.
    pub fn hgcd_matrix(
        u: &InternalMpUint,
        v: &InternalMpUint,
        scratch: &mut DivScratch,
    ) -> HgcdMatrix {
        Self::hgcd_matrix_depth(u, v, scratch, Self::MAX_HGCD_DEPTH)
    }

    /// Computes the Half-GCD transition matrix `M` with bounded recursion depth.
    pub fn hgcd_matrix_depth(
        u: &InternalMpUint,
        v: &InternalMpUint,
        scratch: &mut DivScratch,
        depth: usize,
    ) -> HgcdMatrix {
        let u_len = u.limbs().len();
        let v_len = v.limbs().len();
        if u_len == 0 || v_len == 0 || u.cmp(v) != Ordering::Greater || depth == 0 {
            return HgcdMatrix::identity();
        }

        let target_len = (u_len.wrapping_add(1)) >> 1;
        if v_len <= target_len {
            return HgcdMatrix::identity();
        }

        if u_len <= Self::HGCD_THRESHOLD {
            return Self::hgcd_basecase_matrix(u.clone(), v.clone(), target_len, scratch);
        }

        let k = target_len >> 1;
        if k == 0 {
            return Self::hgcd_basecase_matrix(u.clone(), v.clone(), target_len, scratch);
        }

        let a1 = u
            .limbs()
            .get(k..)
            .map_or_else(InternalMpUint::zero, InternalMpUint::from_limbs_slice);
        let b1 = v
            .limbs()
            .get(k..)
            .map_or_else(InternalMpUint::zero, InternalMpUint::from_limbs_slice);

        if a1.limbs().len() >= u_len {
            return Self::hgcd_basecase_matrix(u.clone(), v.clone(), target_len, scratch);
        }

        let r_mat = Self::hgcd_matrix_depth(&a1, &b1, scratch, depth.wrapping_sub(1));

        let mut curr_u = u.clone();
        let mut curr_v = v.clone();
        if !r_mat.apply_to_pair(&mut curr_u, &mut curr_v) {
            return Self::hgcd_basecase_matrix(u.clone(), v.clone(), target_len, scratch);
        }

        if curr_v.limbs().len() <= target_len || curr_v.is_zero() {
            return r_mat;
        }

        let mut q = InternalMpUint::zero();
        let mut rem = InternalMpUint::zero();
        Division::div_rem_into(&curr_u, &curr_v, &mut q, &mut rem, scratch);

        let q_mat = HgcdMatrix::from_quotient(q);
        let mut combined_mat = r_mat.mul(&q_mat);

        curr_u = replace(&mut curr_v, rem);

        if curr_v.limbs().len() <= target_len || curr_v.is_zero() {
            return combined_mat;
        }

        let cur_u_len = curr_u.limbs().len();
        if cur_u_len > target_len {
            let k2 = (cur_u_len.wrapping_sub(target_len)) >> 1;
            if k2 > 0 {
                let a2 = curr_u
                    .limbs()
                    .get(k2..)
                    .map_or_else(InternalMpUint::zero, InternalMpUint::from_limbs_slice);
                let b2 = curr_v
                    .limbs()
                    .get(k2..)
                    .map_or_else(InternalMpUint::zero, InternalMpUint::from_limbs_slice);

                if a2.limbs().len() < cur_u_len {
                    let s_mat = Self::hgcd_matrix_depth(&a2, &b2, scratch, depth.wrapping_sub(1));
                    if s_mat.apply_to_pair(&mut curr_u, &mut curr_v) {
                        combined_mat = combined_mat.mul(&s_mat);
                    }
                }
            }
        }

        combined_mat
    }

    /// In-place Lehmer reduction of `(u, v)` down to `target_len` without matrix allocation.
    pub fn hgcd_lehmer_in_place(
        u: &mut InternalMpUint,
        v: &mut InternalMpUint,
        target_len: usize,
        scratch: &mut DivScratch,
        u_backup: &mut Vec<Limb>,
        v_backup: &mut Vec<Limb>,
    ) {
        let mut q = InternalMpUint::zero();
        let mut rem = InternalMpUint::zero();

        while !v.is_zero() && v.limbs().len() > target_len {
            if (*u).cmp(v) == Ordering::Less {
                swap(u, v);
            }
            if v.limbs().len() <= 4 {
                Division::rem_into(u, v, &mut rem, scratch);
                swap(u, v);
                swap(v, &mut rem);
                break;
            }

            let (u0, v0, u1, v1, even) = if u.limbs().len() == v.limbs().len()
                && u.limbs().len() >= Self::WIDE_LEHMER_THRESHOLD
            {
                let (u_hat, v_hat) = Self::extract_top_two_limbs(u.limbs(), v.limbs());
                Self::lehmer_simulate_wide(u_hat, v_hat)
            } else {
                let (u_hat, v_hat) = Self::extract_top_limb(u.limbs(), v.limbs());
                Self::lehmer_simulate(u_hat, v_hat)
            };

            let is_identity = u0 == 1 && v0 == 0 && u1 == 0 && v1 == 1;
            if is_identity {
                Division::div_rem_into(u, v, &mut q, &mut rem, scratch);
                swap(u, v);
                swap(v, &mut rem);
            } else {
                let ok = Self::lehmer_update(u, v, u_backup, v_backup, u0, v0, u1, v1, even);
                if !ok {
                    Division::div_rem_into(u, v, &mut q, &mut rem, scratch);
                    swap(u, v);
                    swap(v, &mut rem);
                }
            }
        }
    }

    /// Basecase transition matrix computation using Euclidean division steps.
    pub fn hgcd_basecase_matrix(
        mut r0: InternalMpUint,
        mut r1: InternalMpUint,
        target_len: usize,
        scratch: &mut DivScratch,
    ) -> HgcdMatrix {
        let mut mat = HgcdMatrix::identity();
        let mut q = InternalMpUint::zero();
        let mut next_r = InternalMpUint::zero();

        while !r1.is_zero() && r1.limbs().len() > target_len {
            Division::div_rem_into(&r0, &r1, &mut q, &mut next_r, scratch);
            let q_mat = HgcdMatrix::from_quotient(replace(&mut q, InternalMpUint::zero()));
            mat = mat.mul(&q_mat);
            swap(&mut r0, &mut r1);
            swap(&mut r1, &mut next_r);
        }
        mat
    }
}
