//! Shared SIMD schedule; each backend supplies its ISA operations and lane count.

macro_rules! vector_backend {
    ($name:ident, $vector:ty, $lanes:literal, $features:literal,
     $load:path, $store:path, $splat:path, $add:path, $sub:path, $mul:path,
     $fmsub:expr, $fnmadd:expr, $round:expr) => {
        /// Architecture-specific compensated residue kernels.
        #[derive(Debug)]
        pub struct $name;

        impl $name {
            /// Executes SIMD radix-4 butterflies over an admitted quarter range.
            ///
            /// # Safety
            /// The CPU supports this backend's target features. `values` covers
            /// four initialized, exclusively writable quarter spans and `twiddles`
            /// covers three readable quarter spans. `start <= end <= quarter`.
            /// Residues satisfy the scalar kernel's field and magnitude bounds;
            /// twiddles are centered in (-p/2, p/2).
            #[target_feature(enable = $features)]
            pub unsafe fn radix4<const INVERSE: bool>(
                values: *mut f64,
                twiddles: *const f64,
                quarter: usize,
                start: usize,
                end: usize,
                p: f64,
                pinv: f64,
            ) {
                // SAFETY: the selector establishes ISA support. Each vector
                // iteration has lanes <= end-index and stays inside the four
                // disjoint quarters and three twiddle spans.
                unsafe {
                    let modulus = $splat(p);
                    let reciprocal = $splat(pinv);
                    let q2 = quarter.unchecked_mul(2);
                    let q3 = quarter.unchecked_mul(3);
                    let mut index = start;
                    while end.unchecked_sub(index) >= $lanes {
                        let a = $load(values.add(index));
                        let b = $load(values.add(quarter.unchecked_add(index)));
                        let c = $load(values.add(q2.unchecked_add(index)));
                        let d = $load(values.add(q3.unchecked_add(index)));
                        let w0 = $load(twiddles.add(index));
                        let w1 = $load(twiddles.add(quarter.unchecked_add(index)));
                        let w2 = $load(twiddles.add(q2.unchecked_add(index)));
                        let (y0, y1, y2, y3) = if INVERSE {
                            let tb = multiply_vector(b, w2, modulus, reciprocal);
                            let td = multiply_vector(d, w2, modulus, reciprocal);
                            let s0 = reduce_vector($add(a, tb), modulus, reciprocal);
                            let s1 = reduce_vector($sub(a, tb), modulus, reciprocal);
                            let t0 = reduce_vector($add(c, td), modulus, reciprocal);
                            let t1 = reduce_vector($sub(c, td), modulus, reciprocal);
                            let u0 = multiply_vector(t0, w0, modulus, reciprocal);
                            let u1 = multiply_vector(t1, w1, modulus, reciprocal);
                            ($add(s0, u0), $add(s1, u1), $sub(s0, u0), $sub(s1, u1))
                        } else {
                            let ar = reduce_vector(a, modulus, reciprocal);
                            let s0 = reduce_vector($add(ar, c), modulus, reciprocal);
                            let s1 = multiply_vector($sub(ar, c), w0, modulus, reciprocal);
                            let t0 = reduce_vector($add(b, d), modulus, reciprocal);
                            let t1 = multiply_vector($sub(b, d), w1, modulus, reciprocal);
                            let u1 = multiply_vector($sub(s0, t0), w2, modulus, reciprocal);
                            let u3 = multiply_vector($sub(s1, t1), w2, modulus, reciprocal);
                            ($add(s0, t0), u1, $add(s1, t1), u3)
                        };
                        $store(values.add(index), y0);
                        $store(values.add(quarter.unchecked_add(index)), y1);
                        $store(values.add(q2.unchecked_add(index)), y2);
                        $store(values.add(q3.unchecked_add(index)), y3);
                        index = index.unchecked_add($lanes);
                    }
                    Scalar::radix4::<INVERSE>(values, twiddles, quarter, index, end, p, pinv);
                }
            }

            /// Multiplies or squares frequency coefficients with compensated FMA products.
            ///
            /// # Safety
            /// The selector proves ISA support. `left` covers `len` initialized
            /// writable residues; `right` covers a disjoint readable span for
            /// multiplication and is unused for squaring. Magnitudes are below `2*p`.
            #[target_feature(enable = $features)]
            pub unsafe fn pointwise<const SQUARE: bool>(
                left: *mut f64,
                right: *const f64,
                len: usize,
                p: f64,
                pinv: f64,
            ) {
                // SAFETY: the selector establishes ISA support; complete
                // vectors stay below len and scalar cleanup covers only the tail.
                unsafe {
                    let modulus = $splat(p);
                    let reciprocal = $splat(pinv);
                    let mut index = 0;
                    while len.unchecked_sub(index) >= $lanes {
                        let a = $load(left.add(index));
                        let b = if SQUARE { a } else { $load(right.add(index)) };
                        $store(left.add(index), multiply_vector(a, b, modulus, reciprocal));
                        index = index.unchecked_add($lanes);
                    }
                    let tail = if SQUARE {
                        core::ptr::null()
                    } else {
                        right.add(index)
                    };
                    Scalar::pointwise::<SQUARE>(
                        left.add(index),
                        tail,
                        len.unchecked_sub(index),
                        p,
                        pinv,
                    );
                }
            }

            /// Scales the active inverse-transform coefficient prefix.
            ///
            /// # Safety
            /// ISA support is established. `values` covers `len` writable
            /// initialized residues below `2*p`; `factor` lies in `0..p`.
            #[target_feature(enable = $features)]
            pub unsafe fn scale(values: *mut f64, len: usize, factor: f64, p: f64, pinv: f64) {
                // SAFETY: the selector establishes ISA support; vector
                // accesses and scalar cleanup remain in the admitted output prefix.
                unsafe {
                    let modulus = $splat(p);
                    let reciprocal = $splat(pinv);
                    let multiplier = $splat(factor);
                    let mut index = 0;
                    while len.unchecked_sub(index) >= $lanes {
                        let a = $load(values.add(index));
                        $store(
                            values.add(index),
                            multiply_vector(a, multiplier, modulus, reciprocal),
                        );
                        index = index.unchecked_add($lanes);
                    }
                    Scalar::scale(values.add(index), len.unchecked_sub(index), factor, p, pinv);
                }
            }
        }

        #[inline]
        #[target_feature(enable = $features)]
        unsafe fn reduce_vector(a: $vector, p: $vector, pinv: $vector) -> $vector {
            ($fnmadd)(($round)($mul(a, pinv)), p, a)
        }

        #[inline]
        #[target_feature(enable = $features)]
        unsafe fn multiply_vector(a: $vector, b: $vector, p: $vector, pinv: $vector) -> $vector {
            // The FMA recovers the exact product error; the compensated
            // remainder is an exact integer of magnitude below 2*p.
            let high = $mul(a, b);
            let quotient = ($round)($mul(high, pinv));
            let low = ($fmsub)(a, b, high);
            $add(($fnmadd)(quotient, p, high), low)
        }
    };
}
