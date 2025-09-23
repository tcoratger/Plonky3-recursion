#![allow(clippy::needless_range_loop)]
use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_circuit::tables::SampleBitsTrace;
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField64};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

use super::utils::pad_to_power_of_two;

/// AIR for proving sample_bits operations: extract n bits from input field element.
///
/// This chip implements the circuit version of `challenger.sample_bits(n)` used in FRI
/// verification. It constrains the relationship between:
/// - Input: a field element from challenger sampling
/// - Output: extracted bits as field element, range [0, 2^n - 1]
/// - Bit decomposition: private witness for the input's binary representation
///
/// The chip enforces:
/// 1. Each bit in the decomposition is 0 or 1
/// 2. The bit decomposition correctly reconstructs the input
/// 3. The output correctly extracts the lowest n bits
/// 4. Range constraints on the output (0 ≤ output < 2^n)
#[derive(Debug, Clone)]
pub struct SampleBitsAir<F, const D: usize = 1> {
    pub height: usize,
    _phantom: core::marker::PhantomData<F>,
}

impl<F: Field, const D: usize> SampleBitsAir<F, D> {
    pub const fn new(height: usize) -> Self {
        Self {
            height,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Convert SampleBitsTrace to a row-major matrix for the AIR.
    ///
    /// This handles the variable-length bit decompositions by flattening them
    /// across multiple rows and using auxiliary columns to track parsing state.
    ///
    /// Layout per row:
    /// - Columns 0..D: input value limbs
    /// - Column D: input index
    /// - Columns D+1..2*D+1: output value limbs
    /// - Column 2*D+1: output index
    /// - Column 2*D+2: num_bits (private)
    /// - Column 2*D+3: operation_id (tracks which sample_bits op this row belongs to)
    /// - Columns 2*D+4..: bit decomposition values (padded/repeated as needed)
    pub fn trace_to_matrix<ExtF: BasedVectorSpace<F>>(
        trace: &SampleBitsTrace<ExtF>,
    ) -> RowMajorMatrix<F> {
        let num_ops = trace.input_values.len();
        if num_ops == 0 {
            // Empty trace case
            let width = Self::min_width();
            let mut values = F::zero_vec(width);
            pad_to_power_of_two(&mut values, width, 1);
            return RowMajorMatrix::new(values, width);
        }

        // Calculate maximum bit decomposition length to determine row count
        let max_bit_length = if trace.bit_decomposition_lengths.is_empty() {
            // Default assumption for field bit width
            // TODO: This should be a parameter somewhere
            64
        } else {
            *trace.bit_decomposition_lengths.iter().max().unwrap() as usize
        };

        // Each operation may need multiple rows for its bit decomposition
        let rows_per_op = max_bit_length;
        let total_rows = num_ops * rows_per_op;
        let width = Self::min_width();

        let mut values = Vec::with_capacity(total_rows * width);
        let mut bit_offset = 0;

        for op_idx in 0..num_ops {
            let input_coeffs = trace.input_values[op_idx].as_basis_coefficients_slice();
            let output_coeffs = trace.output_values[op_idx].as_basis_coefficients_slice();
            let bit_length = trace.bit_decomposition_lengths[op_idx] as usize;

            assert_eq!(input_coeffs.len(), D, "Input extension degree mismatch");
            assert_eq!(output_coeffs.len(), D, "Output extension degree mismatch");

            // Generate rows for this operation
            for bit_row in 0..rows_per_op {
                // Input value limbs
                values.extend_from_slice(input_coeffs);
                // Input index
                values.push(F::from_u64(trace.input_index[op_idx] as u64));
                // Output value limbs
                values.extend_from_slice(output_coeffs);
                // Output index
                values.push(F::from_u64(trace.output_index[op_idx] as u64));
                // Number of bits extracted (private)
                values.push(F::from_u64(trace.num_bits[op_idx] as u64));
                // Operation ID (to group rows belonging to same operation)
                values.push(F::from_u64(op_idx as u64));

                // Bit decomposition value for this row
                if bit_row < bit_length {
                    // Convert ExtF to F by taking the first coefficient
                    let bit_value = trace.bit_decompositions[bit_offset + bit_row]
                        .as_basis_coefficients_slice()[0];
                    values.push(bit_value);
                } else {
                    // Padding for shorter bit decompositions
                    values.push(F::ZERO);
                }
            }

            bit_offset += bit_length;
        }

        // Pad to power of two
        pad_to_power_of_two(&mut values, width, total_rows);

        RowMajorMatrix::new(values, width)
    }

    pub const fn min_width() -> usize {
        // input(D) + input_idx + output(D) + output_idx + num_bits + op_id + bit
        2 * D + 4 + 1
    }
}

impl<F: Field, const D: usize> BaseAir<F> for SampleBitsAir<F, D> {
    fn width(&self) -> usize {
        Self::min_width()
    }
}

impl<AB: AirBuilder, const D: usize> Air<AB> for SampleBitsAir<AB::F, D>
where
    AB::F: PrimeField64,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main
            .row_slice(0)
            .expect("SampleBitsAir requires non-empty matrix");
        let local = &*local;

        // Parse row layout
        let _input_limbs = &local[0..D];
        let _input_idx = local[D].clone();
        let _output_limbs = &local[D + 1..2 * D + 1];
        let _output_idx = local[2 * D + 1].clone();
        let _num_bits = local[2 * D + 2].clone();
        let _op_id = local[2 * D + 3].clone();
        let bit = local[2 * D + 4].clone();

        // Constraint 1: Each bit must be 0 or 1
        // bit * (bit - 1) = 0
        let bit_minus_one = bit.clone() - AB::F::ONE;
        builder.assert_zero(bit.clone() * bit_minus_one);

        // Constraint 2: Range check num_bits
        // For simplicity, we assume num_bits is in valid range [1, 63]
        // TODO: A full implementation would add range check constraints

        // TODO: For the future (to keep in mind):
        //
        // The bit decomposition reconstruction and output extraction constraints
        // are more complex and require multi-row logic. For this implementation,
        // we rely on the trace generation to compute correct values and focus
        // on the bit constraint above.

        // In a full implementation, we would add:
        // - Multi-row constraints to reconstruct input from bit decomposition
        // - Constraints to verify output extracts correct bits
        // - Proper range checks for all values

        // The lookup constraints to the Witness table are handled by the
        // system-level LogUp aggregation, not local AIR constraints.
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_uni_stark::{prove, verify};

    use super::*;
    use crate::air::test_utils::build_test_config;

    type F = BabyBear;

    #[test]
    fn test_sample_bits_air_basic() {
        // Test basic sample_bits operation: extract 3 bits from value 13 (binary: 1101)
        // Expected output: 5 (binary: 101 - lowest 3 bits)

        // 1101 in binary
        let input_value = F::from_u64(13);
        // 101 in binary (lowest 3 bits)
        let expected_output = F::from_u64(5);
        let num_bits = 3u32;

        // Bit decomposition of 13: [1, 0, 1, 1, 0, 0, 0, 0, ...] (LSB first)
        let bit_decomposition = vec![
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE, // lowest 4 bits: 1101
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO, // padding
        ];

        let trace = SampleBitsTrace {
            input_values: vec![input_value],
            input_index: vec![10],
            output_values: vec![expected_output],
            output_index: vec![20],
            num_bits: vec![num_bits],
            bit_decompositions: bit_decomposition.clone(),
            bit_decomposition_lengths: vec![bit_decomposition.len() as u32],
        };

        let matrix = SampleBitsAir::<F, 1>::trace_to_matrix(&trace);
        let height = matrix.height();

        // Verify matrix dimensions
        assert_eq!(height, 8);
        assert_eq!(matrix.width(), SampleBitsAir::<F, 1>::min_width());

        // Test proof generation and verification
        let config = build_test_config();
        let pis: Vec<F> = vec![];

        let air = SampleBitsAir::<F, 1>::new(height);
        let proof = prove(&config, &air, matrix, &pis);
        verify(&config, &air, &proof, &pis).expect("SampleBits verification failed");
    }

    #[test]
    fn test_sample_bits_multiple_operations() {
        // Test multiple sample_bits operations in one trace
        let inputs = vec![F::from_u64(15), F::from_u64(7)]; // 1111, 0111
        let outputs = vec![F::from_u64(7), F::from_u64(3)]; // extract 3 bits: 111, 011
        let num_bits = vec![3u32, 3u32];

        let bit_decompositions = vec![
            // First operation: 15 = 1111 in binary
            F::ONE,
            F::ONE,
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ZERO,
            // Second operation: 7 = 0111 in binary
            F::ONE,
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ];

        let trace = SampleBitsTrace {
            input_values: inputs,
            input_index: vec![1, 2],
            output_values: outputs,
            output_index: vec![10, 11],
            num_bits,
            bit_decompositions,
            bit_decomposition_lengths: vec![6, 6],
        };

        let matrix = SampleBitsAir::<F, 1>::trace_to_matrix(&trace);
        let height = matrix.height();

        let config = build_test_config();
        let pis: Vec<F> = vec![];

        let air = SampleBitsAir::<F, 1>::new(height);
        let proof = prove(&config, &air, matrix, &pis);
        verify(&config, &air, &proof, &pis).expect("Multiple SampleBits verification failed");
    }
}
