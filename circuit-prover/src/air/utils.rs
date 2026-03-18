use alloc::vec::Vec;
use core::iter;

use itertools::Itertools;
use p3_air::{AirBuilder, AirLayout};
use p3_field::Field;
use p3_lookup::lookup_traits::{Direction, LookupInput};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression, SymbolicVariable};

use super::alu_air::{
    PREP_A_IDX, PREP_A_IS_READER, PREP_C_IS_READER, PREP_MULT_A, PREP_MULT_B, PREP_MULT_OUT,
};

pub fn get_index_lookups<F: Field, const D: usize>(
    main_start: usize,
    preprocessed_start: usize,
    num_lookups: usize,
    main: &[SymbolicVariable<F>],
    preprocessed: &[SymbolicVariable<F>],
    direction: Direction,
) -> Vec<LookupInput<F>> {
    (0..num_lookups)
        .map(|i| {
            let idx = SymbolicExpression::from(preprocessed[1 + preprocessed_start + i]);

            let multiplicity = SymbolicExpression::from(preprocessed[preprocessed_start]);

            let values = (0..D).map(|j| SymbolicExpression::from(main[main_start + i * D + j]));
            let inps = iter::once(idx).chain(values).collect::<Vec<_>>();

            (inps, multiplicity, direction)
        })
        .collect()
}

/// Get ALU lookups for the 4 operands (a, b, c, out).
///
/// Uses the `PREP_*` constants from [`super::alu_air`] for column offsets.
pub fn get_alu_index_lookups<F: Field, const D: usize>(
    main_start: usize,
    preprocessed_start: usize,
    main: &[SymbolicVariable<F>],
    preprocessed: &[SymbolicVariable<F>],
) -> Vec<LookupInput<F>> {
    let mult_a = SymbolicExpression::from(preprocessed[preprocessed_start + PREP_MULT_A]);
    let mult_b = SymbolicExpression::from(preprocessed[preprocessed_start + PREP_MULT_B]);
    let mult_out = SymbolicExpression::from(preprocessed[preprocessed_start + PREP_MULT_OUT]);
    let a_is_reader = SymbolicExpression::from(preprocessed[preprocessed_start + PREP_A_IS_READER]);
    let c_is_reader = SymbolicExpression::from(preprocessed[preprocessed_start + PREP_C_IS_READER]);

    let eff_mult_a = mult_a.clone() * a_is_reader;
    let eff_mult_c = mult_a * c_is_reader;

    let multiplicities = [eff_mult_a, mult_b, eff_mult_c, mult_out];

    (0..4)
        .map(|i| {
            let idx = SymbolicExpression::from(preprocessed[preprocessed_start + PREP_A_IDX + i]);

            let values = (0..D).map(|j| SymbolicExpression::from(main[main_start + i * D + j]));
            let inps = iter::once(idx).chain(values).collect::<Vec<_>>();

            (inps, multiplicities[i].clone(), Direction::Receive)
        })
        .collect()
}

/// Helper to create symbolic air builder and extract symbolic variables for lookup generation.
///
/// Returns `(symbolic_main_local, preprocessed_local)` slices for use in lookup generation.
pub fn create_symbolic_variables<F: Field>(
    preprocessed_width: usize,
    main_width: usize,
    num_public_values: usize,
    num_permutation_cols: usize,
) -> (Vec<SymbolicVariable<F>>, Vec<SymbolicVariable<F>>) {
    let layout = AirLayout {
        preprocessed_width,
        main_width,
        num_public_values,
        permutation_width: num_permutation_cols,
        num_permutation_challenges: 0,
        num_permutation_values: 0,
        num_periodic_columns: 0,
    };
    let symbolic_air_builder = SymbolicAirBuilder::<F>::new(layout);

    let symbolic_main = symbolic_air_builder.main();
    let symbolic_main_local = symbolic_main.row_slice(0).unwrap().to_vec();

    let preprocessed = symbolic_air_builder.preprocessed().clone();
    let preprocessed_local = preprocessed
        .row_slice(0)
        .expect("The preprocessed matrix has only one row?")
        .to_vec();

    (symbolic_main_local, preprocessed_local)
}

/// Helper to create preprocessed trace with multiplicity insertion and padding.
///
/// Takes preprocessed values (one per op, without multiplicities) and inserts multiplicities
/// at the start of each lane, then pads to power-of-two and min_height.
/// This version assumes each op has exactly one preprocessed value (like PublicAir).
pub fn create_preprocessed_trace_with_multiplicity<F: Field>(
    preprocessed_values: &[F],
    preprocessed_lane_width: usize,
    num_lanes: usize,
    min_height: usize,
    get_multiplicity: impl Fn(usize) -> F,
) -> RowMajorMatrix<F> {
    let num_ops = preprocessed_values.len();
    let num_rows = num_ops.div_ceil(num_lanes);
    let row_width = num_lanes * preprocessed_lane_width;

    let mut values = Vec::with_capacity(num_rows * row_width);
    for row_idx in 0..num_rows {
        for lane in 0..num_lanes {
            let op_idx = row_idx * num_lanes + lane;
            if op_idx < num_ops {
                values.push(get_multiplicity(op_idx));
                values.push(preprocessed_values[op_idx]);
                // Add remaining columns if preprocessed_lane_width > 2
                if preprocessed_lane_width > 2 {
                    values.extend(core::iter::repeat_n(F::ZERO, preprocessed_lane_width - 2));
                }
            } else {
                values.extend(core::iter::repeat_n(F::ZERO, preprocessed_lane_width));
            }
        }
    }

    let mut mat = RowMajorMatrix::new(values, row_width);
    mat.pad_to_min_power_of_two_height(min_height, F::ZERO);
    mat
}

/// Helper to create preprocessed trace for single-row-per-op AIRs (like ConstAir).
///
/// Inserts multiplicity before each preprocessed value, then pads.
pub fn create_simple_preprocessed_trace<F: Field>(
    preprocessed_values: &[F],
    preprocessed_width: usize,
    min_height: usize,
) -> RowMajorMatrix<F> {
    let preprocessed_with_multiplicity: Vec<F> = preprocessed_values
        .iter()
        .flat_map(|v| [F::ONE, *v])
        .collect();

    let mut mat = RowMajorMatrix::new(preprocessed_with_multiplicity, preprocessed_width);
    mat.pad_to_min_power_of_two_height(min_height, F::ZERO);
    mat
}

/// Helper to create preprocessed trace for AIRs with chunked preprocessed values (like AluAir).
///
/// Takes preprocessed values grouped in chunks of (preprocessed_lane_width - 1) per op,
/// inserts multiplicity at the start of each chunk, then pads.
pub fn create_chunked_preprocessed_trace<F: Field>(
    preprocessed_values: &[F],
    preprocessed_lane_width: usize,
    num_lanes: usize,
    min_height: usize,
) -> RowMajorMatrix<F> {
    // Add multiplicity to preprocessed values
    let mut preprocessed_with_multiplicity: Vec<F> = preprocessed_values
        .iter()
        .chunks(preprocessed_lane_width - 1)
        .into_iter()
        .flat_map(|chunk| iter::once(F::ONE).chain(chunk.into_iter().cloned()))
        .collect();

    debug_assert!(
        preprocessed_with_multiplicity
            .len()
            .is_multiple_of(preprocessed_lane_width),
        "Preprocessed trace length mismatch"
    );

    let preprocessed_width = num_lanes * preprocessed_lane_width;
    let padding_len =
        preprocessed_width - preprocessed_with_multiplicity.len() % preprocessed_width;
    if padding_len != preprocessed_width {
        preprocessed_with_multiplicity.extend(core::iter::repeat_n(F::ZERO, padding_len));
    }

    let mut mat = RowMajorMatrix::new(preprocessed_with_multiplicity, preprocessed_width);
    mat.pad_to_min_power_of_two_height(min_height, F::ZERO);
    mat
}
