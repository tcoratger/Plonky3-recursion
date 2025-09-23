#![allow(clippy::needless_range_loop)]
use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_circuit::tables::MulTrace;
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

use super::utils::pad_to_power_of_two;

/// AIR for proving extension field multiplication operations: lhs * rhs = result.
///
/// This chip is specialized for extension field multiplication over degree D > 1,
/// enforcing the constraint `out = a * b` over the chosen extension field.
///
/// Unlike the general `MulAir` which handles both base and extension fields,
/// this chip is optimized specifically for extension field operations with
/// better support for various extension types.
///
/// Column layout (main trace):
///   [lhs[0..D-1], lhs_index, rhs[0..D-1], rhs_index, result[0..D-1], result_index]
///   (width = 3*D + 3)
///
/// The chip uses lookups to the central Witness table to bind its inputs and outputs.
#[derive(Debug, Clone)]
pub struct ExtMulAir<F, const D: usize> {
    /// Number of logical extension multiplication operations in the trace.
    pub num_ops: usize,
    /// Number of lanes (operations) packed per row.
    pub lanes: usize,
    /// For binomial extensions x^D = W over a polynomial basis
    pub w_binomial: F,
    _phantom: core::marker::PhantomData<F>,
}

impl<F: Field + PrimeCharacteristicRing, const D: usize> ExtMulAir<F, D> {
    /// Number of base-field columns contributed by a single extension multiplication lane.
    pub const LANE_WIDTH: usize = 3 * D + 3;

    /// Constructor for binomial polynomial-basis extensions x^D = W.
    ///
    /// This chip requires D >= 2 (for D==1, use the regular MulAir).
    pub const fn new(num_ops: usize, lanes: usize, w: F) -> Self {
        assert!(lanes > 0, "lane count must be non-zero");
        assert!(
            D >= 2,
            "ExtMulAir requires D >= 2 (use MulAir for base field)"
        );
        Self {
            num_ops,
            lanes,
            w_binomial: w,
            _phantom: core::marker::PhantomData,
        }
    }

    pub const fn lane_width() -> usize {
        Self::LANE_WIDTH
    }

    pub fn total_width(&self) -> usize {
        self.lanes * Self::lane_width()
    }

    /// Convert `MulTrace` to a row-major matrix, packing `lanes` extension multiplications per row.
    ///
    /// Layout per lane: `[lhs[D], lhs_idx, rhs[D], rhs_idx, result[D], result_idx]`.
    pub fn trace_to_matrix<ExtF: BasedVectorSpace<F>>(
        trace: &MulTrace<ExtF>,
        lanes: usize,
    ) -> RowMajorMatrix<F> {
        assert!(lanes > 0, "lane count must be non-zero");
        assert!(D >= 2, "ExtMulAir trace requires extension degree D >= 2");

        let lane_width = Self::lane_width();
        let width = lane_width * lanes;
        let op_count = trace.lhs_values.len();
        let row_count = op_count.div_ceil(lanes);

        let mut values = Vec::with_capacity(width * row_count.max(1));

        for row in 0..row_count {
            for lane in 0..lanes {
                let op_idx = row * lanes + lane;
                if op_idx < op_count {
                    // LHS limbs + index
                    let lhs_coeffs = trace.lhs_values[op_idx].as_basis_coefficients_slice();
                    assert_eq!(lhs_coeffs.len(), D, "Extension degree mismatch for lhs");
                    values.extend_from_slice(lhs_coeffs);
                    values.push(F::from_u64(trace.lhs_index[op_idx] as u64));

                    // RHS limbs + index
                    let rhs_coeffs = trace.rhs_values[op_idx].as_basis_coefficients_slice();
                    assert_eq!(rhs_coeffs.len(), D, "Extension degree mismatch for rhs");
                    values.extend_from_slice(rhs_coeffs);
                    values.push(F::from_u64(trace.rhs_index[op_idx] as u64));

                    // Result limbs + index
                    let result_coeffs = trace.result_values[op_idx].as_basis_coefficients_slice();
                    assert_eq!(
                        result_coeffs.len(),
                        D,
                        "Extension degree mismatch for result",
                    );
                    values.extend_from_slice(result_coeffs);
                    values.push(F::from_u64(trace.result_index[op_idx] as u64));
                } else {
                    // Filler lane: append zeros for unused slot to keep the row width uniform.
                    values.resize(values.len() + lane_width, F::ZERO);
                }
            }
        }

        pad_to_power_of_two(&mut values, width, row_count);

        RowMajorMatrix::new(values, width)
    }
}

impl<F: Field, const D: usize> BaseAir<F> for ExtMulAir<F, D> {
    fn width(&self) -> usize {
        self.total_width()
    }
}

impl<AB: AirBuilder, const D: usize> Air<AB> for ExtMulAir<AB::F, D>
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        debug_assert_eq!(main.width(), self.total_width(), "column width mismatch");

        let local = main.row_slice(0).expect("matrix must be non-empty");
        let local = &*local;
        let lane_width = Self::lane_width();

        // Extension field multiplication using binomial reduction x^D = W
        let w = AB::Expr::from(self.w_binomial);

        for lane in 0..self.lanes {
            let mut cursor = lane * lane_width;

            // Parse LHS extension element (D coefficients + index)
            let lhs_slice = &local[cursor..cursor + D];
            // Skip lhs coefficients and index
            cursor += D + 1;

            // Parse RHS extension element (D coefficients + index)
            let rhs_slice = &local[cursor..cursor + D];
            // Skip rhs coefficients and index
            cursor += D + 1;

            // Parse result extension element (D coefficients + index)
            let result_slice = &local[cursor..cursor + D];

            // Compute extension field multiplication using schoolbook algorithm.
            //
            // TODO: optimize this part later.
            //
            // For binomial extension x^D = W, we have:
            // (a_0 + a_1*x + ... + a_{D-1}*x^{D-1}) * (b_0 + b_1*x + ... + b_{D-1}*x^{D-1})
            // = sum_{i,j} a_i * b_j * x^{i+j}
            // where x^k = W * x^{k-D} for k >= D (binomial reduction)

            let mut acc: Vec<AB::Expr> = (0..D).map(|_| AB::Expr::ZERO).collect();

            for i in 0..D {
                for j in 0..D {
                    let term = lhs_slice[i].clone() * rhs_slice[j].clone();
                    let k = i + j;
                    if k < D {
                        // Coefficient of x^k
                        acc[k] = acc[k].clone() + term;
                    } else {
                        // Coefficient of x^{k-D} after reduction x^k = W * x^{k-D}
                        acc[k - D] = acc[k - D].clone() + w.clone() * term;
                    }
                }
            }

            // Enforce that computed product equals the claimed result
            for k in 0..D {
                builder.assert_zero(result_slice[k].clone() - acc[k].clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear as Val;
    use p3_circuit::tables::MulTrace;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
    use p3_uni_stark::{prove, verify};

    use super::*;
    use crate::air::test_utils::build_test_config;

    type Ext4 = BinomialExtensionField<Val, 4>;

    #[test]
    fn test_ext_mul_trace_to_matrix_basic() {
        // Test basic extension field multiplication trace conversion
        let a = Ext4::from_basis_coefficients_slice(&[
            Val::from_u64(1),
            Val::from_u64(2),
            Val::from_u64(3),
            Val::from_u64(4),
        ])
        .unwrap();
        let b = Ext4::from_basis_coefficients_slice(&[
            Val::from_u64(5),
            Val::from_u64(6),
            Val::from_u64(7),
            Val::from_u64(8),
        ])
        .unwrap();
        // Genuine extension field multiplication
        let c = a * b;

        let trace = MulTrace {
            lhs_values: vec![a],
            lhs_index: vec![1],
            rhs_values: vec![b],
            rhs_index: vec![2],
            result_values: vec![c],
            result_index: vec![3],
        };

        let matrix = ExtMulAir::<Val, 4>::trace_to_matrix(&trace, 1);

        // Verify matrix dimensions
        assert_eq!(matrix.height(), 1);
        assert_eq!(matrix.width(), ExtMulAir::<Val, 4>::lane_width());

        // Verify first row contains our data
        let row_slice = matrix.row_slice(0).unwrap();

        // Check LHS coefficients
        let a_coeffs = a.as_basis_coefficients_slice();
        assert_eq!(&row_slice[0..4], a_coeffs);
        // lhs_index
        assert_eq!(row_slice[4], Val::from_u64(1));

        // Check RHS coefficients
        let b_coeffs = b.as_basis_coefficients_slice();
        assert_eq!(&row_slice[5..9], b_coeffs);
        // rhs_index
        assert_eq!(row_slice[9], Val::from_u64(2));

        // Check result coefficients
        let c_coeffs = c.as_basis_coefficients_slice();
        assert_eq!(&row_slice[10..14], c_coeffs);
        // result_index
        assert_eq!(row_slice[14], Val::from_u64(3));
    }

    #[test]
    fn test_ext_mul_air_constraint_validation() {
        // Compute W parameter from the field definition
        let x = Ext4::from_basis_coefficients_slice(&[Val::ZERO, Val::ONE, Val::ZERO, Val::ZERO])
            .unwrap();
        let x4 = x.exp_u64(4);
        let x4_coeffs = <Ext4 as BasedVectorSpace<Val>>::as_basis_coefficients_slice(&x4);
        let w: Val = x4_coeffs[0];

        let a = Ext4::from_basis_coefficients_slice(&[
            Val::from_u64(2),
            Val::from_u64(1),
            Val::from_u64(0),
            Val::from_u64(1),
        ])
        .unwrap();
        let b = Ext4::from_basis_coefficients_slice(&[
            Val::from_u64(1),
            Val::from_u64(3),
            Val::from_u64(2),
            Val::from_u64(0),
        ])
        .unwrap();
        let c = a * b;

        let trace = MulTrace {
            lhs_values: vec![a],
            lhs_index: vec![10],
            rhs_values: vec![b],
            rhs_index: vec![11],
            result_values: vec![c],
            result_index: vec![12],
        };

        let matrix = ExtMulAir::<Val, 4>::trace_to_matrix(&trace, 1);
        let config = build_test_config();
        let pis: Vec<Val> = vec![];

        let air = ExtMulAir::<Val, 4>::new(1, 1, w);
        let proof = prove(&config, &air, matrix, &pis);
        verify(&config, &air, &proof, &pis).expect("ExtMul verification failed");
    }

    #[test]
    fn test_ext_mul_multiple_lanes() {
        // Compute W parameter from the field definition
        let x = Ext4::from_basis_coefficients_slice(&[Val::ZERO, Val::ONE, Val::ZERO, Val::ZERO])
            .unwrap();
        let x4 = x.exp_u64(4);
        let x4_coeffs = <Ext4 as BasedVectorSpace<Val>>::as_basis_coefficients_slice(&x4);
        let w: Val = x4_coeffs[0];

        let ops = vec![
            (
                Ext4::from_basis_coefficients_slice(&[
                    Val::from_u64(1),
                    Val::from_u64(0),
                    Val::from_u64(0),
                    Val::from_u64(0),
                ])
                .unwrap(),
                Ext4::from_basis_coefficients_slice(&[
                    Val::from_u64(2),
                    Val::from_u64(0),
                    Val::from_u64(0),
                    Val::from_u64(0),
                ])
                .unwrap(),
            ),
            (
                Ext4::from_basis_coefficients_slice(&[
                    Val::from_u64(0),
                    Val::from_u64(1),
                    Val::from_u64(0),
                    Val::from_u64(0),
                ])
                .unwrap(),
                Ext4::from_basis_coefficients_slice(&[
                    Val::from_u64(0),
                    Val::from_u64(1),
                    Val::from_u64(0),
                    Val::from_u64(0),
                ])
                .unwrap(),
            ),
        ];

        let mut lhs_values = Vec::new();
        let mut rhs_values = Vec::new();
        let mut result_values = Vec::new();
        let mut lhs_index = Vec::new();
        let mut rhs_index = Vec::new();
        let mut result_index = Vec::new();

        for (i, (a, b)) in ops.into_iter().enumerate() {
            let c = a * b;
            lhs_values.push(a);
            rhs_values.push(b);
            result_values.push(c);
            lhs_index.push(i as u32 * 3);
            rhs_index.push(i as u32 * 3 + 1);
            result_index.push(i as u32 * 3 + 2);
        }

        let trace = MulTrace {
            lhs_values,
            lhs_index,
            rhs_values,
            rhs_index,
            result_values,
            result_index,
        };

        let lanes = 2;
        let matrix = ExtMulAir::<Val, 4>::trace_to_matrix(&trace, lanes);

        // Should fit both operations in a single row
        assert_eq!(matrix.height(), 1);
        assert_eq!(matrix.width(), ExtMulAir::<Val, 4>::lane_width() * lanes);

        let config = build_test_config();
        let pis: Vec<Val> = vec![];

        let air = ExtMulAir::<Val, 4>::new(2, lanes, w);
        let proof = prove(&config, &air, matrix, &pis);
        verify(&config, &air, &proof, &pis).expect("Multi-lane ExtMul verification failed");
    }

    #[test]
    #[should_panic(expected = "ExtMulAir requires D >= 2")]
    fn test_ext_mul_rejects_base_field() {
        // ExtMulAir should reject degree 1 (base field)
        ExtMulAir::<Val, 1>::new(1, 1, Val::from_u64(0));
    }
}
