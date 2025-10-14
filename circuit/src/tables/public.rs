use alloc::vec::Vec;

use crate::types::WitnessId;

/// Public input table.
///
/// Tracks all public inputs to the circuit.
/// Both prover and verifier know these values.
/// They represent the externally visible interface to the computation.
#[derive(Debug, Clone)]
pub struct PublicTrace<F> {
    /// Witness IDs of each public input.
    ///
    /// Identifies which witness slots contain public values.
    pub index: Vec<WitnessId>,

    /// Public input field element values.
    ///
    /// Provided by the external caller.
    /// Serve as the starting point for computation.
    pub values: Vec<F>,
}
