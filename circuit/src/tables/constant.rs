use alloc::vec::Vec;

use crate::types::WitnessId;

/// Constant values table.
///
/// Stores all compile-time known constant values used in the circuit.
/// Each constant binds to a specific witness ID.
/// Both prover and verifier know these values in advance.
#[derive(Debug, Clone)]
pub struct ConstTrace<F> {
    /// Witness IDs that each constant binds to.
    ///
    /// Maps each constant to its location in the witness table.
    pub index: Vec<WitnessId>,
    /// Constant field element values.
    ///
    /// These values remain fixed across all executions.
    pub values: Vec<F>,
}
