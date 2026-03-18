use alloc::vec::Vec;
use core::iter;

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

/// Helper to create a preprocessed trace from complete preprocessed values (already including
/// multiplicities). Simply reshapes the flat per-op data into a row-major matrix.
pub fn create_direct_preprocessed_trace<F: Field>(
    preprocessed_values: &[F],
    preprocessed_lane_width: usize,
    num_lanes: usize,
    min_height: usize,
) -> RowMajorMatrix<F> {
    let preprocessed_width = num_lanes * preprocessed_lane_width;

    let mut values = preprocessed_values.to_vec();

    // Pad to a multiple of the full row width
    if preprocessed_width > 0 && !values.len().is_multiple_of(preprocessed_width) {
        let padding = preprocessed_width - (values.len() % preprocessed_width);
        values.extend(core::iter::repeat_n(F::ZERO, padding));
    }

    // Ensure at least one row
    if values.is_empty() {
        values.extend(core::iter::repeat_n(F::ZERO, preprocessed_width.max(1)));
    }

    let mut mat = RowMajorMatrix::new(values, preprocessed_width);
    mat.pad_to_min_power_of_two_height(min_height, F::ZERO);
    mat
}

/// Like [`create_direct_preprocessed_trace`], but allocates the *final* width
/// directly by appending a fixed number of zero columns to every row.
///
/// This is useful when an AIR wants its preprocessed trace to have extra
/// global columns (e.g. selectors) that are always zero in the unscheduled
/// case: we can avoid building a narrow matrix and widening it row-by-row.
pub fn create_direct_preprocessed_trace_with_extra<F: Field>(
    preprocessed_values: &[F],
    preprocessed_lane_width: usize,
    extra_cols_per_row: usize,
    num_lanes: usize,
    min_height: usize,
) -> RowMajorMatrix<F> {
    let base_width = num_lanes * preprocessed_lane_width;

    let mut values = preprocessed_values.to_vec();

    // Pad to a multiple of the *base* row width.
    if base_width > 0 && !values.len().is_multiple_of(base_width) {
        let padding = base_width - (values.len() % base_width);
        values.extend(core::iter::repeat_n(F::ZERO, padding));
    }

    // Ensure at least one base row.
    if values.is_empty() {
        values.extend(core::iter::repeat_n(F::ZERO, base_width.max(1)));
    }

    let num_rows = values.len() / base_width;
    let target_width = base_width + extra_cols_per_row;

    // Build the widened matrix in one pass: copy each base row into the
    // prefix of the widened row and leave the extra columns zero.
    let mut widened = RowMajorMatrix::new(F::zero_vec(num_rows * target_width), target_width);
    for r in 0..num_rows {
        let src_start = r * base_width;
        let dst_start = r * target_width;
        widened.values[dst_start..dst_start + base_width]
            .copy_from_slice(&values[src_start..src_start + base_width]);
        // trailing extra_cols_per_row entries stay zero
    }

    widened.pad_to_min_power_of_two_height(min_height, F::ZERO);
    widened
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
