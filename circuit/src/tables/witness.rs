use alloc::vec::Vec;

use crate::types::WitnessId;

/// Central witness table.
///
/// Primary storage for all intermediate values in the circuit.
/// Acts as the "bus" that all operation tables reference.
/// Witnesses have sequential IDs starting from 0.
#[derive(Debug, Clone)]
pub struct WitnessTrace<F> {
    /// Sequential witness IDs: WitnessId(0), WitnessId(1), WitnessId(2), ...
    ///
    /// Forms a preprocessed column for lookups from other tables.
    pub index: Vec<WitnessId>,

    /// Witness field element values.
    ///
    /// Each value is one computation result.
    /// Computed during circuit execution.
    pub values: Vec<F>,
}
