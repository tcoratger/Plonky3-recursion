use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::Field;

use super::context::ExecutionContext;
use super::npo::NpoTypeId;
use crate::types::WitnessId;
use crate::{CircuitError, PreprocessedColumns};

/// Trait for operation-specific execution state.
///
/// Each non-primitive operation type can define its own state struct that persists
/// across invocations during circuit execution. This enables features like:
/// - Permutation chaining (storing previous output for next input)
/// - Recording execution data for canonical trace generation
///
/// Implementors must support downcasting via `Any` for type-safe retrieval.
pub trait OpExecutionState: core::any::Any + Send + Sync + Debug {
    /// Downcast to concrete type for reading.
    fn as_any(&self) -> &dyn core::any::Any;
    /// Downcast to concrete type for mutation.
    fn as_any_mut(&mut self) -> &mut dyn core::any::Any;
}

/// Trait for executable non-primitive operations
///
/// This trait enables dynamic dispatch and allows each operation to control
/// its own execution logic with full access to the execution context.
pub trait NonPrimitiveExecutor<F: Field>: Debug {
    /// Execute the operation with full context access
    ///
    /// # Arguments
    /// * `inputs` - Input witness indices
    /// * `outputs` - Output witness indices
    /// * `ctx` - Execution context with access to witness table, private data, and configs
    fn execute(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        ctx: &mut ExecutionContext<'_, F>,
    ) -> Result<(), CircuitError>;

    /// Get operation type identifier (for config lookup, error reporting)
    fn op_type(&self) -> &NpoTypeId;

    /// Allow downcasting to concrete executor types
    fn as_any(&self) -> &dyn core::any::Any;

    /// Update the preprocessed values related to this operation. This consists of:
    /// - the preprocessed values for the associated table
    /// - the multiplicity for the `Witness` table.
    ///
    /// Uses the `PreprocessedColumns` API to ensure witness multiplicities are updated
    /// consistently when reading from the witness table. Duplicate-output detection
    /// (which outputs were already defined by an earlier op) is handled generically
    /// by `generate_preprocessed_columns` after this method returns.
    fn preprocess(
        &self,
        _inputs: &[Vec<WitnessId>],
        _outputs: &[Vec<WitnessId>],
        _preprocessed: &mut PreprocessedColumns<F>,
    ) -> Result<(), CircuitError> {
        Ok(())
    }

    /// How many leading output groups are exposed as creators on the witness-checks bus.
    /// `generate_preprocessed_columns` only performs duplicate-output detection on the
    /// first `n` groups returned here.
    ///
    /// Override when some outputs are private (e.g. capacity elements in a permutation).
    /// Default: all groups.
    fn num_exposed_outputs(&self) -> Option<usize> {
        None
    }

    /// Clone as trait object
    fn boxed(&self) -> Box<dyn NonPrimitiveExecutor<F>>;
}

impl<F: Field> Clone for Box<dyn NonPrimitiveExecutor<F>> {
    fn clone(&self) -> Self {
        self.boxed()
    }
}

/// Trait for executable hint operations.
///
/// Hints are non-deterministic witness assignments that do not have associated AIR tables
/// or traces. They operate directly on the witness array.
pub trait HintExecutor<F: Field>: Debug {
    /// Execute the hint.
    ///
    /// - `inputs`: Witness IDs to read from
    /// - `outputs`: Witness IDs to write to
    /// - `witness`: Mutable reference to the witness table
    fn execute(
        &self,
        inputs: &[WitnessId],
        outputs: &[WitnessId],
        witness: &mut [Option<F>],
    ) -> Result<(), CircuitError>;

    /// Clone as trait object.
    fn boxed(&self) -> Box<dyn HintExecutor<F>>;
}

impl<F: Field> Clone for Box<dyn HintExecutor<F>> {
    fn clone(&self) -> Self {
        self.boxed()
    }
}
