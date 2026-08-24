//! Evaluation-point orchestration and reciprocal coupling for Toom-8/8.5.

use super::{
    CouplingContext, EvaluationDirection, EvaluationPoint, Limb, MulEvaluationBuffers,
    SqrEvaluationBuffers, Toom8, Values,
};

impl Toom8 {
    pub fn evaluate_and_couple_mul(
        values: &mut Values<'_>,
        temporary: &mut [Limb],
        evaluations: &mut MulEvaluationBuffers<'_>,
        a: &[Limb],
        b: &[Limb],
        context: &CouplingContext<'_>,
    ) {
        let direct = EvaluationDirection::Direct;
        let reciprocal = EvaluationDirection::Reciprocal;
        evaluate_mul_point(
            values.one,
            temporary,
            evaluations,
            a,
            b,
            context,
            EvaluationPoint {
                direction: direct,
                shift: 0,
            },
        );
        // Pair each direct row with z^6*P(1/z) immediately. The reverse
        // difference convention feeds the antisymmetric interpolation schedule,
        // and consuming both 3m-limb rows here keeps the recursive products hot.
        evaluate_mul_point(
            values.two,
            temporary,
            evaluations,
            a,
            b,
            context,
            EvaluationPoint {
                direction: direct,
                shift: 1,
            },
        );
        evaluate_mul_point(
            values.half,
            temporary,
            evaluations,
            a,
            b,
            context,
            EvaluationPoint {
                direction: reciprocal,
                shift: 1,
            },
        );
        Self::sum_and_reverse_difference(values.two, values.half);
        evaluate_mul_point(
            values.four,
            temporary,
            evaluations,
            a,
            b,
            context,
            EvaluationPoint {
                direction: direct,
                shift: 2,
            },
        );
        evaluate_mul_point(
            values.quarter,
            temporary,
            evaluations,
            a,
            b,
            context,
            EvaluationPoint {
                direction: reciprocal,
                shift: 2,
            },
        );
        Self::sum_and_reverse_difference(values.four, values.quarter);
        evaluate_mul_point(
            values.eight,
            temporary,
            evaluations,
            a,
            b,
            context,
            EvaluationPoint {
                direction: direct,
                shift: 3,
            },
        );
        evaluate_mul_point(
            values.eighth,
            temporary,
            evaluations,
            a,
            b,
            context,
            EvaluationPoint {
                direction: reciprocal,
                shift: 3,
            },
        );
        Self::sum_and_reverse_difference(values.eight, values.eighth);
    }

    pub fn evaluate_and_couple_sqr(
        values: &mut Values<'_>,
        temporary: &mut [Limb],
        evaluations: &mut SqrEvaluationBuffers<'_>,
        operand: &[Limb],
        context: &CouplingContext<'_>,
    ) {
        let direct = EvaluationDirection::Direct;
        let reciprocal = EvaluationDirection::Reciprocal;
        evaluate_sqr_point(
            values.one,
            temporary,
            evaluations,
            operand,
            context,
            EvaluationPoint {
                direction: direct,
                shift: 0,
            },
        );
        // Squaring has the same reciprocal Vandermonde pairing as multiplication;
        // transform each row pair before the next recursive square evicts it.
        evaluate_sqr_point(
            values.two,
            temporary,
            evaluations,
            operand,
            context,
            EvaluationPoint {
                direction: direct,
                shift: 1,
            },
        );
        evaluate_sqr_point(
            values.half,
            temporary,
            evaluations,
            operand,
            context,
            EvaluationPoint {
                direction: reciprocal,
                shift: 1,
            },
        );
        Self::sum_and_reverse_difference(values.two, values.half);
        evaluate_sqr_point(
            values.four,
            temporary,
            evaluations,
            operand,
            context,
            EvaluationPoint {
                direction: direct,
                shift: 2,
            },
        );
        evaluate_sqr_point(
            values.quarter,
            temporary,
            evaluations,
            operand,
            context,
            EvaluationPoint {
                direction: reciprocal,
                shift: 2,
            },
        );
        Self::sum_and_reverse_difference(values.four, values.quarter);
        evaluate_sqr_point(
            values.eight,
            temporary,
            evaluations,
            operand,
            context,
            EvaluationPoint {
                direction: direct,
                shift: 3,
            },
        );
        evaluate_sqr_point(
            values.eighth,
            temporary,
            evaluations,
            operand,
            context,
            EvaluationPoint {
                direction: reciprocal,
                shift: 3,
            },
        );
        Self::sum_and_reverse_difference(values.eight, values.eighth);
    }
}

fn evaluate_mul_point(
    packed: &mut [Limb],
    temporary: &mut [Limb],
    evaluations: &mut MulEvaluationBuffers<'_>,
    a: &[Limb],
    b: &[Limb],
    context: &CouplingContext<'_>,
    evaluation_point: EvaluationPoint,
) {
    let value_len = packed.len().wrapping_sub(context.split_len);
    let (negative, _) = temporary.split_at_mut(value_len);
    let sign = Toom8::evaluate_mul_pair(
        packed,
        negative,
        evaluations,
        a,
        b,
        context.split_len,
        evaluation_point,
    );
    match evaluation_point.direction {
        EvaluationDirection::Direct => {
            Toom8::couple_direct(packed, negative, sign, context, evaluation_point.shift);
        }
        EvaluationDirection::Reciprocal => {
            Toom8::couple_reciprocal(packed, negative, sign, context, evaluation_point.shift);
        }
    }
}

fn evaluate_sqr_point(
    packed: &mut [Limb],
    temporary: &mut [Limb],
    evaluations: &mut SqrEvaluationBuffers<'_>,
    operand: &[Limb],
    context: &CouplingContext<'_>,
    evaluation_point: EvaluationPoint,
) {
    let value_len = packed.len().wrapping_sub(context.split_len);
    let (negative, _) = temporary.split_at_mut(value_len);
    Toom8::evaluate_sqr_pair(
        packed,
        negative,
        evaluations,
        operand,
        context.split_len,
        evaluation_point,
    );
    match evaluation_point.direction {
        EvaluationDirection::Direct => {
            Toom8::couple_direct(packed, negative, false, context, evaluation_point.shift);
        }
        EvaluationDirection::Reciprocal => {
            Toom8::couple_reciprocal(packed, negative, false, context, evaluation_point.shift);
        }
    }
}
