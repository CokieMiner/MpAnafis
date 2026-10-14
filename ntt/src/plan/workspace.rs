//! Retained twiddle tables, validated geometry, and aligned reusable transform buffers.

#![expect(
    unsafe_code,
    reason = "Prepared geometry proves catalog and exclusive arena spans before unchecked access"
)]

use alloc::vec::Vec;

use crate::{
    arch::NttFloatKernels,
    transform::{Tables, Twiddles},
};

use super::{DefaultExecutor, Limb, Multiplication, NttPlan, PRIME_CATALOG, ParallelExecutor};

/// Reusable native-limb multiplication workspace.
#[derive(Clone, Debug)]
pub struct NttWorkspace {
    plan: Option<NttPlan>,
    transform_len: usize,
    cached_primes: usize,
    forward: Vec<f64>,
    inverse: Vec<f64>,
    arena: Vec<f64>,
    kernels: NttFloatKernels,
}

impl NttWorkspace {
    /// Creates empty storage and resolves supported CPU kernels once.
    #[must_use]
    pub fn new() -> Self {
        Self {
            plan: None,
            transform_len: 0,
            cached_primes: 0,
            forward: Vec::new(),
            inverse: Vec::new(),
            arena: Vec::new(),
            kernels: NttFloatKernels::get(),
        }
    }

    /// Retains a native multiplication plan and prepares its tables and buffers.
    pub fn prepare(&mut self, plan: NttPlan) {
        DefaultExecutor::with_resolved(|executor| self.prepare_with_executor(plan, executor));
    }

    /// Prepares a native plan using an explicit executor.
    pub fn prepare_with_executor<E: ParallelExecutor>(&mut self, plan: NttPlan, executor: &E) {
        self.prepare_storage(plan, false, executor);
    }

    /// Multiplies into the exact native output prefix with default execution.
    ///
    /// Admission failures preserve `dst`, including its suffix.
    pub fn mul(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) -> bool {
        DefaultExecutor::with_resolved(|executor| self.mul_with_executor(dst, a, b, executor))
    }

    /// Multiplies using the retained matching plan or selects a new native plan.
    pub fn mul_with_executor<E: ParallelExecutor>(
        &mut self,
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        executor: &E,
    ) -> bool {
        let Some(output_len) = a.len().checked_add(b.len()) else {
            return false;
        };
        let Some(output) = dst.get_mut(..output_len) else {
            return false;
        };
        if a.is_empty() || b.is_empty() {
            output.fill(0);
            return true;
        }
        let Some(plan) = self
            .plan
            .filter(|p| p.len_a == a.len() && p.len_b == b.len())
            .or_else(|| NttPlan::for_multiplication(a.len(), b.len()))
        else {
            return false;
        };
        self.prepare_storage(plan, false, executor);
        Multiplication::execute::<false, E>(
            output,
            (a, b),
            &mut self.arena,
            Tables {
                forward: &self.forward,
                inverse: &self.inverse,
            },
            &plan,
            &self.kernels,
            executor,
        );
        true
    }

    /// Squares into the exact native output prefix with default execution.
    pub fn sqr(&mut self, dst: &mut [Limb], a: &[Limb]) -> bool {
        DefaultExecutor::with_resolved(|executor| self.sqr_with_executor(dst, a, executor))
    }

    /// Squares using a retained matching plan and explicit execution policy.
    pub fn sqr_with_executor<E: ParallelExecutor>(
        &mut self,
        dst: &mut [Limb],
        a: &[Limb],
        executor: &E,
    ) -> bool {
        let Some(output_len) = a.len().checked_mul(2) else {
            return false;
        };
        let Some(output) = dst.get_mut(..output_len) else {
            return false;
        };
        if a.is_empty() {
            output.fill(0);
            return true;
        }
        let Some(plan) = self
            .plan
            .filter(|p| p.len_a == a.len() && p.len_b == a.len())
            .or_else(|| NttPlan::for_multiplication(a.len(), a.len()))
        else {
            return false;
        };
        self.prepare_storage(plan, true, executor);
        Multiplication::execute::<true, E>(
            output,
            (a, a),
            &mut self.arena,
            Tables {
                forward: &self.forward,
                inverse: &self.inverse,
            },
            &plan,
            &self.kernels,
            executor,
        );
        true
    }

    fn prepare_storage<E: ParallelExecutor>(&mut self, plan: NttPlan, square: bool, executor: &E) {
        let geometry = &plan.planned;
        let needed = plan.arena_len;
        if self.transform_len != geometry.transform_len || self.cached_primes < geometry.profile.np
        {
            self.forward.resize(needed, 0.0);
            self.inverse.resize(needed, 0.0);
            // SAFETY: the immutable plan admits np in 1..=MAX_PRIMES.
            let primes = unsafe { PRIME_CATALOG.get_unchecked(..geometry.profile.np) };
            Twiddles::prepare(
                &mut self.forward,
                &mut self.inverse,
                geometry.transform_len,
                primes,
                executor.parallelism().get(),
                executor,
            );
            self.transform_len = geometry.transform_len;
            self.cached_primes = geometry.profile.np;
        }
        let operands = if square {
            needed
        } else {
            // SAFETY: admission checks four arena spans, so two spans fit usize.
            unsafe { needed.unchecked_mul(2) }
        };
        if self.arena.len() < operands {
            self.arena.resize(operands, 0.0);
        }
        self.plan = Some(plan);
    }
}

impl Default for NttWorkspace {
    fn default() -> Self {
        Self::new()
    }
}
