use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter;
use core::ops::{Add, Mul, Sub};

use hashbrown::HashMap;
use p3_field::Field;
use strum::EnumCount;

use crate::CircuitError;
use crate::op::{NonPrimitiveOpConfig, NonPrimitiveOpType, Op, PrimitiveOpType};
use crate::tables::{CircuitRunner, TraceGeneratorFn};
use crate::types::{ExprId, WitnessId};

/// Trait encapsulating the required field operations for circuits
pub trait CircuitField:
    Clone
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + PartialEq
    + Debug
    + Field
{
}

impl<F> CircuitField for F where
    F: Clone
        + Default
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + PartialEq
        + Debug
        + Field
{
}

/// Static circuit specification containing constraint system and metadata
///
/// This represents the compiled output of a `CircuitBuilder`. It contains:
/// - Primitive operations (add, multiply, subtract, constants, public inputs)
/// - Non-primitive operations (complex operations like MMCS verification)
/// - Public input metadata and witness table structure
///
/// The circuit is static and serializable. Use `.runner()` to create
/// a `CircuitRunner` for execution with specific input values.
#[derive(Debug)]
pub struct Circuit<F> {
    /// Number of witness table rows
    pub witness_count: u32,
    /// Primitive operations in topological order
    pub primitive_ops: Vec<Op<F>>,
    /// Non-primitive operations
    pub non_primitive_ops: Vec<Op<F>>,
    /// Public input witness indices
    pub public_rows: Vec<WitnessId>,
    /// Total number of public field elements
    pub public_flat_len: usize,
    /// Enabled non-primitive operation types with their respective configuration
    pub enabled_ops: HashMap<NonPrimitiveOpType, NonPrimitiveOpConfig>,
    pub expr_to_widx: HashMap<ExprId, WitnessId>,
    /// Registered non-primitive trace generators.
    pub non_primitive_trace_generators: HashMap<NonPrimitiveOpType, TraceGeneratorFn<F>>,
}

impl<F: Field + Clone> Clone for Circuit<F> {
    fn clone(&self) -> Self {
        Self {
            witness_count: self.witness_count,
            primitive_ops: self.primitive_ops.clone(),
            non_primitive_ops: self.non_primitive_ops.clone(),
            public_rows: self.public_rows.clone(),
            public_flat_len: self.public_flat_len,
            enabled_ops: self.enabled_ops.clone(),
            expr_to_widx: self.expr_to_widx.clone(),
            non_primitive_trace_generators: self.non_primitive_trace_generators.clone(),
        }
    }
}

impl<F: Field> Circuit<F> {
    pub fn new(witness_count: u32, expr_to_widx: HashMap<ExprId, WitnessId>) -> Self {
        Self {
            witness_count,
            primitive_ops: Vec::new(),
            non_primitive_ops: Vec::new(),
            public_rows: Vec::new(),
            public_flat_len: 0,
            enabled_ops: HashMap::new(),
            expr_to_widx,
            non_primitive_trace_generators: HashMap::new(),
        }
    }

    /// Generates preprocessed columns for all primitive operation types.
    ///
    /// # Overview
    ///
    /// Preprocessed columns are fixed, circuit-dependent values computed once during setup
    /// and committed before proof generation.
    ///
    /// # Output Structure
    ///
    /// Returns a `Vec<Vec<F>>` with exactly `PrimitiveOpType::COUNT` inner vectors,
    /// one per operation type, indexed by `PrimitiveOpType as usize`:
    ///
    /// | Index | Operation | Column Layout                             | Width |
    /// |-------|-----------|-------------------------------------------|-------|
    /// | 0     | Witness   | `[idx_0, idx_1, ..., idx_max]`            | 1     |
    /// | 1     | Const     | `[out_0, out_1, ...]`                     | 1     |
    /// | 2     | Public    | `[out_0, out_1, ...]`                     | 1     |
    /// | 3     | Add       | `[a_0, b_0, out_0, a_1, b_1, out_1, ...]` | 3     |
    /// | 4     | Mul       | `[a_0, b_0, out_0, a_1, b_1, out_1, ...]` | 3     |
    ///
    /// # Example
    ///
    /// For a circuit with:
    /// - `Const { out: 0, val: 5 }`
    /// - `Const { out: 1, val: 3 }`
    /// - `Add { a: 0, b: 1, out: 2 }`
    ///
    /// Returns:
    /// ```text
    /// [
    ///   [0, 1, 2],        // Witness: indices 0..=2
    ///   [0, 1],           // Const: outputs [0, 1]
    ///   [],               // Public: empty
    ///   [0, 1, 2],        // Add: (a=0, b=1, out=2)
    ///   [],               // Mul: empty
    /// ]
    /// ```
    pub fn generate_preprocessed_columns(&self) -> Result<Vec<Vec<F>>, CircuitError> {
        // Allocate one empty vector per primitive operation type (Witness, Const, Public, Add, Mul).
        let mut preprocessed = vec![vec![]; PrimitiveOpType::COUNT];

        // Track the maximum witness index to determine the Witness table size.
        let mut max_idx = 0;

        // Process each primitive operation, extracting its witness indices.
        for prim in &self.primitive_ops {
            match prim {
                // Const: stores a constant value at witness[out].
                // Preprocessed data: the output witness index.
                Op::Const { out, .. } => {
                    let table_idx = PrimitiveOpType::Const as usize;
                    preprocessed[table_idx].extend(&[F::from_u32(out.0)]);
                    max_idx = max_idx.max(out.0);
                }
                // Public: loads a public input into witness[out].
                // Preprocessed data: the output witness index.
                Op::Public { out, .. } => {
                    let table_idx = PrimitiveOpType::Public as usize;
                    preprocessed[table_idx].extend(&[F::from_u32(out.0)]);
                    max_idx = max_idx.max(out.0);
                }
                // Add: computes witness[out] = witness[a] + witness[b].
                // Preprocessed data: input indices a, b and output index out.
                Op::Add { a, b, out } => {
                    let table_idx = PrimitiveOpType::Add as usize;
                    preprocessed[table_idx].extend(&[
                        F::from_u32(a.0),
                        F::from_u32(b.0),
                        F::from_u32(out.0),
                    ]);
                    max_idx = max_idx.max(a.0).max(b.0).max(out.0);
                }
                // Mul: computes witness[out] = witness[a] * witness[b].
                // Preprocessed data: input indices a, b and output index out.
                Op::Mul { a, b, out } => {
                    let table_idx = PrimitiveOpType::Mul as usize;
                    preprocessed[table_idx].extend(&[
                        F::from_u32(a.0),
                        F::from_u32(b.0),
                        F::from_u32(out.0),
                    ]);
                    max_idx = max_idx.max(a.0).max(b.0).max(out.0);
                }
                // Unconstrained: sets arbitrary witness values via hints.
                // No preprocessed column data, but outputs affect max_idx.
                Op::Unconstrained { outputs, .. } => {
                    max_idx = iter::once(max_idx)
                        .chain(outputs.iter().map(|&output| output.0))
                        .max()
                        .unwrap_or(max_idx);
                }
                // Non-primitive ops should not appear in primitive_ops.
                Op::NonPrimitiveOpWithExecutor { .. } => panic!(
                    "preprocessed values are not yet implemented for non primitive operations."
                ),
            }
        }

        // Generate the Witness table's preprocessed column: sequential indices from 0 to max_idx.
        //
        // This enables the AIR to verify that all witness lookups reference valid slots.
        let table_idx = PrimitiveOpType::Witness as usize;
        preprocessed[table_idx].extend((0..=max_idx).map(|i| F::from_u32(i)));

        Ok(preprocessed)
    }
}

impl<F: CircuitField> Circuit<F> {
    /// Create a circuit runner for execution and trace generation
    pub fn runner(self) -> CircuitRunner<F> {
        CircuitRunner::new(self)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use hashbrown::HashMap;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use strum::EnumCount;

    use super::*;
    use crate::op::{DefaultHint, PrimitiveOpType};
    use crate::types::WitnessId;

    /// Use BabyBear as the test field for all preprocessed column tests.
    type F = BabyBear;

    /// Helper: Create a circuit with specified primitive operations
    ///
    /// Each test specifies only the operations it needs.
    ///
    /// Parameters:
    ///   ops: Vector of primitive operations to include in the circuit
    ///
    /// Returns:
    ///   A Circuit<F> instance ready for testing
    fn make_circuit(ops: Vec<Op<F>>) -> Circuit<F> {
        // Create a minimal circuit with no witness slots pre-allocated.
        //
        // The actual witness count doesn't affect preprocessed column generation.
        let mut circuit = Circuit::new(0, HashMap::new());

        // Populate the circuit's primitive operations from the provided list.
        circuit.primitive_ops = ops;

        circuit
    }

    #[test]
    fn test_empty_circuit_returns_five_empty_columns_with_single_witness() {
        // Test: Empty circuit produces correct structure with zero witness indices
        //
        // An empty circuit has no operations, so:
        //   - All operation-specific columns (Const, Public, Add, Mul) should be empty
        //   - Witness column should contain only index 0 (since max_idx starts at 0)
        //
        // This tests the base case and verifies the output vector has the correct length.

        // Construct a circuit with no operations.
        let circuit: Circuit<F> = make_circuit(vec![]);

        // Generate preprocessed columns for this empty circuit.
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Verify we have exactly 5 columns (one per PrimitiveOpType).
        assert_eq!(
            result.len(),
            PrimitiveOpType::COUNT,
            "Output must have exactly {} columns, one per primitive op type",
            PrimitiveOpType::COUNT
        );

        // Witness column (index 0): contains [0] because max_idx defaults to 0.
        // This ensures the witness table has at least one row even in empty circuits.
        assert_eq!(
            result[PrimitiveOpType::Witness as usize],
            vec![F::ZERO],
            "Empty circuit witness column must contain [0]"
        );

        // All other columns must be empty since no operations exist.
        assert!(
            result[PrimitiveOpType::Const as usize].is_empty(),
            "Const column must be empty"
        );
        assert!(
            result[PrimitiveOpType::Public as usize].is_empty(),
            "Public column must be empty"
        );
        assert!(
            result[PrimitiveOpType::Add as usize].is_empty(),
            "Add column must be empty"
        );
        assert!(
            result[PrimitiveOpType::Mul as usize].is_empty(),
            "Mul column must be empty"
        );
    }

    #[test]
    fn test_single_const_populates_const_column_with_output_index() {
        // Test: Single Const operation populates Const column correctly
        //
        // A Const operation stores a fixed value at a specific witness index.
        // Preprocessed data: only the output witness index (value is runtime data).
        //
        // This verifies:
        //   - Const column contains the output index
        //   - Witness column spans 0..=out
        //   - Other columns remain empty

        // Create a single Const operation: witness[5] = 42.
        let ops = vec![Op::Const {
            out: WitnessId(5),
            val: F::from_u64(42),
        }];

        // Build circuit and generate preprocessed columns.
        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Const column (index 1): contains [5] (the output witness index).
        assert_eq!(
            result[PrimitiveOpType::Const as usize],
            vec![F::from_u32(5)],
            "Const column must contain the output witness index"
        );

        // Witness column (index 0): contains [0, 1, 2, 3, 4, 5] since max_idx = 5.
        let expected_witness: Vec<F> = (0..=5).map(F::from_u32).collect();
        assert_eq!(
            result[PrimitiveOpType::Witness as usize],
            expected_witness,
            "Witness column must span 0..=max_idx"
        );

        // Other columns must remain empty.
        assert!(result[PrimitiveOpType::Public as usize].is_empty());
        assert!(result[PrimitiveOpType::Add as usize].is_empty());
        assert!(result[PrimitiveOpType::Mul as usize].is_empty());
    }

    #[test]
    fn test_multiple_consts_append_indices_in_order() {
        // Test: Multiple Const operations append to Const column in order
        //
        // Multiple Const ops produce multiple entries in the Const column.
        // Witness column size is determined by the maximum index across all ops.

        // Create three Const operations with non-contiguous indices.
        let ops = vec![
            Op::Const {
                out: WitnessId(2),
                val: F::from_u64(10),
            },
            Op::Const {
                out: WitnessId(7),
                val: F::from_u64(20),
            },
            Op::Const {
                out: WitnessId(4),
                val: F::from_u64(30),
            },
        ];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Const column contains indices in operation order: [2, 7, 4].
        assert_eq!(
            result[PrimitiveOpType::Const as usize],
            vec![F::from_u32(2), F::from_u32(7), F::from_u32(4)],
            "Const column must preserve operation order"
        );

        // Witness column spans 0..=7 (max index is 7).
        let expected_witness: Vec<F> = (0..=7).map(F::from_u32).collect();
        assert_eq!(result[PrimitiveOpType::Witness as usize], expected_witness);
    }

    #[test]
    fn test_single_public_populates_public_column_with_output_index() {
        // Test: Single Public operation populates Public column correctly
        //
        // A Public operation loads a public input into a witness slot.
        // Preprocessed data: only the output witness index (public_pos is runtime).

        // Create a Public operation: witness[3] = public_inputs[0].
        let ops = vec![Op::Public {
            out: WitnessId(3),
            public_pos: 0,
        }];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Public column (index 2): contains [3].
        assert_eq!(
            result[PrimitiveOpType::Public as usize],
            vec![F::from_u32(3)],
            "Public column must contain the output witness index"
        );

        // Witness column spans 0..=3.
        let expected_witness: Vec<F> = (0..=3).map(F::from_u32).collect();
        assert_eq!(result[PrimitiveOpType::Witness as usize], expected_witness);

        // Other columns must remain empty.
        assert!(result[PrimitiveOpType::Const as usize].is_empty());
        assert!(result[PrimitiveOpType::Add as usize].is_empty());
        assert!(result[PrimitiveOpType::Mul as usize].is_empty());
    }

    #[test]
    fn test_single_add_populates_add_column_with_triplet() {
        // Test: Single Add operation populates Add column with (a, b, out) triplet
        //
        // An Add operation computes witness[out] = witness[a] + witness[b].
        // Preprocessed data: (a, b, out) indices as a contiguous triplet.

        // Create an Add operation: witness[10] = witness[2] + witness[5].
        let ops = vec![Op::Add {
            a: WitnessId(2),
            b: WitnessId(5),
            out: WitnessId(10),
        }];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Add column (index 3): contains [2, 5, 10] as a flattened triplet.
        assert_eq!(
            result[PrimitiveOpType::Add as usize],
            vec![F::from_u32(2), F::from_u32(5), F::from_u32(10)],
            "Add column must contain (a, b, out) triplet"
        );

        // Witness column spans 0..=10 (max is out=10).
        let expected_witness: Vec<F> = (0..=10).map(F::from_u32).collect();
        assert_eq!(result[PrimitiveOpType::Witness as usize], expected_witness);

        // Other columns must remain empty.
        assert!(result[PrimitiveOpType::Const as usize].is_empty());
        assert!(result[PrimitiveOpType::Public as usize].is_empty());
        assert!(result[PrimitiveOpType::Mul as usize].is_empty());
    }

    #[test]
    fn test_single_mul_populates_mul_column_with_triplet() {
        // Test: Single Mul operation populates Mul column with (a, b, out) triplet
        //
        // A Mul operation computes witness[out] = witness[a] * witness[b].
        // Preprocessed data layout is identical to Add: (a, b, out).

        // Create a Mul operation: witness[8] = witness[1] * witness[3].
        let ops = vec![Op::Mul {
            a: WitnessId(1),
            b: WitnessId(3),
            out: WitnessId(8),
        }];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Mul column (index 4): contains [1, 3, 8].
        assert_eq!(
            result[PrimitiveOpType::Mul as usize],
            vec![F::from_u32(1), F::from_u32(3), F::from_u32(8)],
            "Mul column must contain (a, b, out) triplet"
        );

        // Witness column spans 0..=8.
        let expected_witness: Vec<F> = (0..=8).map(F::from_u32).collect();
        assert_eq!(result[PrimitiveOpType::Witness as usize], expected_witness);
    }

    #[test]
    fn test_unconstrained_affects_max_idx_without_column_data() {
        // Test: Unconstrained operation affects max_idx but produces no column data
        //
        // Unconstrained operations fill witness slots with hints (non-deterministic values).
        // They don't contribute to preprocessed columns but their outputs affect max_idx.

        // Create an Unconstrained operation with outputs at indices 4 and 9.
        let ops = vec![Op::Unconstrained {
            inputs: vec![],
            outputs: vec![WitnessId(4), WitnessId(9)],
            filler: DefaultHint::boxed_default(),
        }];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // All operation-specific columns must be empty (Unconstrained has no column data).
        assert!(
            result[PrimitiveOpType::Const as usize].is_empty(),
            "Unconstrained must not populate Const column"
        );
        assert!(
            result[PrimitiveOpType::Public as usize].is_empty(),
            "Unconstrained must not populate Public column"
        );
        assert!(
            result[PrimitiveOpType::Add as usize].is_empty(),
            "Unconstrained must not populate Add column"
        );
        assert!(
            result[PrimitiveOpType::Mul as usize].is_empty(),
            "Unconstrained must not populate Mul column"
        );

        // Witness column spans 0..=9 because output index 9 is the max.
        let expected_witness: Vec<F> = (0..=9).map(F::from_u32).collect();
        assert_eq!(
            result[PrimitiveOpType::Witness as usize],
            expected_witness,
            "Witness column must reflect Unconstrained output indices"
        );
    }

    #[test]
    fn test_mixed_operations_populate_all_columns_correctly() {
        // Test: Mixed operations populate all columns correctly
        //
        // A realistic circuit uses multiple operation types. This test verifies:
        //   - Each operation type populates its correct column
        //   - max_idx is the global maximum across all operations
        //   - Column data preserves operation order within each type

        // Create a mixed circuit simulating: result = (const_a + public_b) * const_c
        let ops = vec![
            // witness[0] = 100 (constant)
            Op::Const {
                out: WitnessId(0),
                val: F::from_u64(100),
            },
            // witness[1] = public_inputs[0]
            Op::Public {
                out: WitnessId(1),
                public_pos: 0,
            },
            // witness[2] = 200 (another constant)
            Op::Const {
                out: WitnessId(2),
                val: F::from_u64(200),
            },
            // witness[3] = witness[0] + witness[1]
            Op::Add {
                a: WitnessId(0),
                b: WitnessId(1),
                out: WitnessId(3),
            },
            // witness[4] = witness[3] * witness[2]
            Op::Mul {
                a: WitnessId(3),
                b: WitnessId(2),
                out: WitnessId(4),
            },
        ];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Const column: [0, 2] (two Const ops in order).
        assert_eq!(
            result[PrimitiveOpType::Const as usize],
            vec![F::ZERO, F::from_u32(2)],
            "Const column must contain outputs [0, 2] in order"
        );

        // Public column: [1] (one Public op).
        assert_eq!(
            result[PrimitiveOpType::Public as usize],
            vec![F::from_u32(1)],
            "Public column must contain output [1]"
        );

        // Add column: [0, 1, 3] (one Add op triplet).
        assert_eq!(
            result[PrimitiveOpType::Add as usize],
            vec![F::ZERO, F::from_u32(1), F::from_u32(3)],
            "Add column must contain triplet (0, 1, 3)"
        );

        // Mul column: [3, 2, 4] (one Mul op triplet).
        assert_eq!(
            result[PrimitiveOpType::Mul as usize],
            vec![F::from_u32(3), F::from_u32(2), F::from_u32(4)],
            "Mul column must contain triplet (3, 2, 4)"
        );

        // Witness column: 0..=4 (max index is 4 from Mul output).
        let expected_witness: Vec<F> = (0..=4).map(F::from_u32).collect();
        assert_eq!(
            result[PrimitiveOpType::Witness as usize],
            expected_witness,
            "Witness column must span 0..=4"
        );
    }

    #[test]
    fn test_max_idx_determined_by_highest_index_across_all_ops() {
        // Test: max_idx determined by highest index regardless of operation type
        //
        // This test verifies that max_idx correctly tracks the global maximum across:
        //   - Const/Public output indices
        //   - Add/Mul input and output indices
        //   - Unconstrained output indices
        //
        // The Unconstrained op has the highest index, so it determines the witness table size.

        let ops = vec![
            // Const at low index
            Op::Const {
                out: WitnessId(2),
                val: F::from_u64(1),
            },
            // Add with medium indices
            Op::Add {
                a: WitnessId(0),
                b: WitnessId(1),
                out: WitnessId(5),
            },
            // Unconstrained with highest index (20)
            Op::Unconstrained {
                inputs: vec![],
                outputs: vec![WitnessId(20)],
                filler: DefaultHint::boxed_default(),
            },
            // Mul with indices below Unconstrained
            Op::Mul {
                a: WitnessId(3),
                b: WitnessId(4),
                out: WitnessId(10),
            },
        ];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Witness column must span 0..=20 (Unconstrained output is the max).
        let expected_witness: Vec<F> = (0..=20).map(F::from_u32).collect();
        assert_eq!(
            result[PrimitiveOpType::Witness as usize],
            expected_witness,
            "Witness column must span 0..=20 (highest index from Unconstrained)"
        );

        // Verify operation-specific columns are populated correctly.
        assert_eq!(
            result[PrimitiveOpType::Const as usize],
            vec![F::from_u32(2)]
        );
        assert_eq!(
            result[PrimitiveOpType::Add as usize],
            vec![F::ZERO, F::from_u32(1), F::from_u32(5)]
        );
        assert_eq!(
            result[PrimitiveOpType::Mul as usize],
            vec![F::from_u32(3), F::from_u32(4), F::from_u32(10)]
        );
    }

    #[test]
    fn test_multiple_arithmetic_ops_append_triplets_sequentially() {
        // Test: Multiple Add/Mul operations append triplets sequentially
        //
        // When multiple Add or Mul operations exist, their (a, b, out) triplets are
        // concatenated in operation order. This test verifies correct flattening.

        let ops = vec![
            // First Add: (0, 1, 2)
            Op::Add {
                a: WitnessId(0),
                b: WitnessId(1),
                out: WitnessId(2),
            },
            // Second Add: (2, 3, 4)
            Op::Add {
                a: WitnessId(2),
                b: WitnessId(3),
                out: WitnessId(4),
            },
            // First Mul: (4, 5, 6)
            Op::Mul {
                a: WitnessId(4),
                b: WitnessId(5),
                out: WitnessId(6),
            },
            // Second Mul: (6, 7, 8)
            Op::Mul {
                a: WitnessId(6),
                b: WitnessId(7),
                out: WitnessId(8),
            },
        ];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Add column: two triplets flattened as [0, 1, 2, 2, 3, 4].
        assert_eq!(
            result[PrimitiveOpType::Add as usize],
            vec![
                F::ZERO,
                F::from_u32(1),
                F::from_u32(2),
                F::from_u32(2),
                F::from_u32(3),
                F::from_u32(4)
            ],
            "Add column must contain two consecutive triplets"
        );

        // Mul column: two triplets flattened as [4, 5, 6, 6, 7, 8].
        assert_eq!(
            result[PrimitiveOpType::Mul as usize],
            vec![
                F::from_u32(4),
                F::from_u32(5),
                F::from_u32(6),
                F::from_u32(6),
                F::from_u32(7),
                F::from_u32(8)
            ],
            "Mul column must contain two consecutive triplets"
        );

        // Witness spans 0..=8.
        let expected_witness: Vec<F> = (0..=8).map(F::from_u32).collect();
        assert_eq!(result[PrimitiveOpType::Witness as usize], expected_witness);
    }

    #[test]
    fn test_add_input_indices_contribute_to_max_idx() {
        // Test: Input indices of Add/Mul contribute to max_idx
        //
        // The max_idx calculation considers ALL indices in Add/Mul: inputs a, b AND output.
        // This test ensures input indices that exceed the output are properly tracked.

        // Add operation where input 'b' has the highest index.
        let ops = vec![Op::Add {
            a: WitnessId(0),
            b: WitnessId(15), // Highest index
            out: WitnessId(5),
        }];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Witness column must span 0..=15 (input b=15 is the max).
        let expected_witness: Vec<F> = (0..=15).map(F::from_u32).collect();
        assert_eq!(
            result[PrimitiveOpType::Witness as usize],
            expected_witness,
            "Witness column must account for Add input indices"
        );
    }

    #[test]
    fn test_operations_at_zero_index() {
        // Test: Zero-index operations handled correctly
        //
        // Operations using witness index 0 should work correctly (edge case for off-by-one errors).

        let ops = vec![
            // Const at index 0
            Op::Const {
                out: WitnessId(0),
                val: F::from_u64(0),
            },
            // Add using index 0 as both input and output
            Op::Add {
                a: WitnessId(0),
                b: WitnessId(0),
                out: WitnessId(0),
            },
        ];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Const column: [0]
        assert_eq!(result[PrimitiveOpType::Const as usize], vec![F::ZERO]);

        // Add column: [0, 0, 0]
        assert_eq!(
            result[PrimitiveOpType::Add as usize],
            vec![F::ZERO, F::ZERO, F::ZERO]
        );

        // Witness column: [0] (max_idx = 0)
        assert_eq!(
            result[PrimitiveOpType::Witness as usize],
            vec![F::ZERO],
            "Witness column must contain [0] when max_idx is 0"
        );
    }

    #[test]
    fn test_unconstrained_with_empty_outputs() {
        // Test: Unconstrained with empty outputs doesn't affect max_idx
        //
        // An Unconstrained operation with no outputs should not change max_idx.

        let ops = vec![
            // Const sets max_idx to 3
            Op::Const {
                out: WitnessId(3),
                val: F::from_u64(42),
            },
            // Unconstrained with no outputs - should not change max_idx
            Op::Unconstrained {
                inputs: vec![WitnessId(0)],
                outputs: vec![],
                filler: DefaultHint::boxed_default(),
            },
        ];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns().unwrap();

        // Witness column: 0..=3 (Const determines max_idx)
        let expected_witness: Vec<F> = (0..=3).map(F::from_u32).collect();
        assert_eq!(
            result[PrimitiveOpType::Witness as usize],
            expected_witness,
            "Empty Unconstrained outputs should not affect max_idx"
        );
    }
}
