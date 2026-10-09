//! Paired benchmark declarations with symmetric setup, sampling, and timing.

/// Declares one function/scenario with identical sampling and loop structure.
/// Setup closures return operand batches; operation closures evaluate one item.
#[macro_export]
macro_rules! paired_bench {
    ($name:ident, $widths:expr, mp: $mp_setup:expr => $mp_op:expr,
        rug: $rug_setup:expr => $rug_op:expr
        $(, flint: $flint_setup:expr => $flint_op:expr)? $(,)?) => {
        $crate::int::support::paired_bench!($name, $widths,
            samples = ($crate::int::support::SAMPLE_SIZE_WIDE, $crate::int::support::SAMPLE_COUNT_WIDE),
            mp: $mp_setup => $mp_op, rug: $rug_setup => $rug_op
            $(, flint: $flint_setup => $flint_op)?);
    };
    ($name:ident, $widths:expr, samples = ($size:expr, $count:expr),
        mp: $mp_setup:expr => $mp_op:expr, rug: $rug_setup:expr => $rug_op:expr
        $(, flint: $flint_setup:expr => $flint_op:expr)?
        $(, scenarios = $scenarios:path)? $(,)?) => {
        $crate::int::support::paired_bench!($name, $widths, samples = ($size, $count),
            mp: $mp_setup => $mp_op, rug: $rug_setup => $rug_op,
            verify = $crate::int::support::verify_pair
            $(, flint: $flint_setup => $flint_op)? $(, scenarios = $scenarios)?);
    };
    ($name:ident, $widths:expr, samples = ($size:expr, $count:expr),
        mp: $mp_setup:expr => $mp_op:expr, rug: $rug_setup:expr => $rug_op:expr,
        verify = $verify:expr $(, flint: $flint_setup:expr => $flint_op:expr)?
        $(, scenarios = $scenarios:path)? $(,)?) => {
        mod $name {
            #[allow(clippy::wildcard_imports, reason = "Benchmark expressions resolve their category's explicit imports")]
            use super::*;

            #[divan::bench(args = $widths, sample_size = $size, sample_count = $count)]
            fn mp(bencher: divan::Bencher, bits: usize) {
                let inputs = ($mp_setup)(bits);
                #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                {
                    let peer_inputs = ($rug_setup)(bits);
                    ($verify)(&inputs, &peer_inputs, $mp_op, $rug_op);
                }
                $(
                    #[cfg(all(feature = "_internal-tune", target_arch = "x86_64", target_os = "linux", target_pointer_width = "64"))]
                    {
                        let flint_inputs = ($flint_setup)(bits);
                        $crate::int::support::verify_pair(&inputs, &flint_inputs, $mp_op, $flint_op);
                    }
                )?
                $crate::int::support::measure(bencher, &inputs, |input| {
                    let _output = divan::black_box(($mp_op)(input));
                });
            }

            #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
            #[divan::bench(args = $widths, sample_size = $size, sample_count = $count)]
            fn rug(bencher: divan::Bencher, bits: usize) {
                let inputs = ($rug_setup)(bits);
                let peer_inputs = ($mp_setup)(bits);
                ($verify)(&peer_inputs, &inputs, $mp_op, $rug_op);
                $crate::int::support::measure(bencher, &inputs, |input| {
                    let _output = divan::black_box(($rug_op)(input));
                });
            }

            $(
                #[cfg(all(feature = "_internal-tune", target_arch = "x86_64", target_os = "linux", target_pointer_width = "64"))]
                #[divan::bench(args = $widths, sample_size = $size, sample_count = $count)]
                fn flint(bencher: divan::Bencher, bits: usize) {
                    let inputs = ($flint_setup)(bits);
                    let peer_inputs = ($mp_setup)(bits);
                    $crate::int::support::verify_pair(&peer_inputs, &inputs, $mp_op, $flint_op);
                    $crate::int::support::measure(bencher, &inputs, |input| {
                        let _output = divan::black_box(($flint_op)(input));
                    });
                }
            )?

            $crate::int::support::paired_bench!(@scenarios
                [$($scenarios)?], [$($flint_op)?],
                $widths, $size, $count, $mp_op, $rug_op, $verify);
        }
    };
    (@scenarios [], $_references:tt,
        $_widths:expr, $_size:expr, $_count:expr, $_mp:expr, $_rug:expr, $_verify:expr) => {};
    (@scenarios [$scenarios:path], [],
        $widths:expr, $size:expr, $count:expr, $mp_op:expr, $rug_op:expr, $verify:expr) => {
        $scenarios!($widths, $size, $count, $mp_op, $rug_op, $verify);
    };
    (@scenarios [$scenarios:path], [$flint_op:expr],
        $widths:expr, $size:expr, $count:expr, $mp_op:expr, $rug_op:expr, $verify:expr) => {
        $scenarios!($widths, $size, $count, $mp_op, $rug_op, $verify, flint: $flint_op);
    };
}

/// Declares destination-reusing assignments with symmetric prepared state.
#[macro_export]
macro_rules! paired_assign {
    ($name:ident, $widths:expr,
        mp: $mp_setup:expr, $mp_state:expr => $mp_op:expr,
        rug: $rug_setup:expr, $rug_state:expr => $rug_op:expr $(,)?) => {
        mod $name {
            use $crate::int::support::{SAMPLE_COUNT_WIDE, SAMPLE_SIZE_WIDE};

            #[allow(
                clippy::wildcard_imports,
                reason = "Benchmark expressions resolve their category's explicit imports"
            )]
            use super::*;

            #[divan::bench(args = $widths, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
            fn mp(bencher: divan::Bencher, bits: usize) {
                let inputs = ($mp_setup)(bits);
                #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                $crate::int::support::verify_assignment(
                    &inputs,
                    &($rug_setup)(bits),
                    ($mp_state)(bits),
                    ($rug_state)(bits),
                    $mp_op,
                    $rug_op,
                );
                $crate::int::support::measure_assignment(
                    bencher,
                    &inputs,
                    ($mp_state)(bits),
                    $mp_op,
                );
            }

            #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
            #[divan::bench(args = $widths, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
            fn rug(bencher: divan::Bencher, bits: usize) {
                let inputs = ($rug_setup)(bits);
                $crate::int::support::verify_assignment(
                    &($mp_setup)(bits),
                    &inputs,
                    ($mp_state)(bits),
                    ($rug_state)(bits),
                    $mp_op,
                    $rug_op,
                );
                $crate::int::support::measure_assignment(
                    bencher,
                    &inputs,
                    ($rug_state)(bits),
                    $rug_op,
                );
            }
        }
    };
}

/// Resets a mutating receiver outside each timed invocation.
#[macro_export]
macro_rules! paired_mutate {
    ($name:ident, $widths:expr, mp: $mp_setup:expr => $mp_op:expr,
        rug: $rug_setup:expr => $rug_op:expr $(,)?) => {
        mod $name {
            use $crate::int::support::{SAMPLE_COUNT_WIDE, SAMPLE_SIZE_WIDE};

            #[allow(
                clippy::wildcard_imports,
                reason = "Benchmark expressions resolve their category's explicit imports"
            )]
            use super::*;

            #[divan::bench(args = $widths, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
            fn mp(bencher: divan::Bencher, bits: usize) {
                let input = ($mp_setup)(bits);
                #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                verify(bits);
                $crate::int::support::measure_mutation(bencher, input, $mp_op);
            }

            #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
            #[divan::bench(args = $widths, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
            fn rug(bencher: divan::Bencher, bits: usize) {
                let input = ($rug_setup)(bits);
                verify(bits);
                $crate::int::support::measure_mutation(bencher, input, $rug_op);
            }

            #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
            fn verify(bits: usize) {
                let mut mp = ($mp_setup)(bits);
                let mut rug = ($rug_setup)(bits);
                assert_eq!(
                    $crate::int::support::Outcome::encode(&mp),
                    $crate::int::support::Outcome::encode(&rug)
                );
                ($mp_op)(&mut mp);
                ($rug_op)(&mut rug);
                assert_eq!(
                    $crate::int::support::Outcome::encode(&mp),
                    $crate::int::support::Outcome::encode(&rug)
                );
            }
        }
    };
}
