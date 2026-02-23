use alloc::vec;
use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;

use crate::CircuitError;
use crate::types::WitnessId;

/// Memory-efficient witness storage with split initialization tracking.
///
/// Stores field element values separately from their initialization flags.
pub(crate) struct WitnessTable<F> {
    /// Field element storage.
    ///
    /// Slots are zero-initialized on allocation.
    ///
    /// A slot's value is only meaningful when the corresponding flag is set.
    values: Vec<F>,

    /// Per-slot initialization flags.
    ///
    /// `true` indicates:
    /// - the slot has been explicitly written,
    /// - its value can be read safely.
    initialized: Vec<bool>,
}

impl<F: PrimeCharacteristicRing> WitnessTable<F> {
    /// Allocate a table with the given number of slots, all uninitialized.
    pub fn new(count: usize) -> Self {
        Self {
            values: F::zero_vec(count),
            initialized: vec![false; count],
        }
    }

    /// Return the total number of slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return whether the slot at `idx` has been written.
    #[inline]
    pub fn is_initialized(&self, idx: usize) -> bool {
        self.initialized[idx]
    }

    /// Read the value at the given witness index.
    ///
    /// Returns `WitnessNotSet` if the slot is out of bounds or uninitialized.
    #[inline]
    pub fn get(&self, widx: WitnessId) -> Result<F, CircuitError>
    where
        F: Clone,
    {
        let idx = widx.0 as usize;

        // Bounds-check and initialization-check in one branch.
        if idx < self.initialized.len() && self.initialized[idx] {
            Ok(self.values[idx].clone())
        } else {
            Err(CircuitError::WitnessNotSet { witness_id: widx })
        }
    }

    /// Return a reference to the value at `idx` without any checks.
    ///
    /// # Safety contract
    ///
    /// The caller must guarantee that the slot has been initialized.
    #[inline]
    pub fn get_value_unchecked(&self, idx: usize) -> &F {
        &self.values[idx]
    }

    /// Write a value and mark the slot as initialized.
    ///
    /// No conflict detection is performed here — the caller is responsible
    /// for checking whether a different value already occupies this slot.
    #[inline]
    pub fn set_unchecked(&mut self, idx: usize, value: F) {
        self.values[idx] = value;
        self.initialized[idx] = true;
    }

    /// Borrow the raw values slice.
    ///
    /// Intended for trace builders that iterate every slot after execution.
    pub fn values(&self) -> &[F] {
        &self.values
    }

    /// Borrow the initialization flags slice.
    ///
    /// Intended for trace builders that need to distinguish set vs. unset slots.
    pub fn initialized(&self) -> &[bool] {
        &self.initialized
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_new_empty_table() {
        let table = WitnessTable::<F>::new(0);
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_new_allocates_uninitialized_slots() {
        let table = WitnessTable::<F>::new(4);

        assert_eq!(table.len(), 4);
        for i in 0..4 {
            assert!(!table.is_initialized(i));
        }
    }

    #[test]
    fn test_set_marks_slot_initialized() {
        let mut table = WitnessTable::<F>::new(2);

        table.set_unchecked(0, F::from_u64(7));

        assert!(table.is_initialized(0));
        assert!(!table.is_initialized(1));
    }

    #[test]
    fn test_set_overwrites_previous_value() {
        let mut table = WitnessTable::<F>::new(1);

        table.set_unchecked(0, F::from_u64(10));
        table.set_unchecked(0, F::from_u64(20));

        assert_eq!(table.get(WitnessId(0)).unwrap(), F::from_u64(20));
    }

    #[test]
    fn test_get_returns_written_value() {
        let mut table = WitnessTable::<F>::new(3);
        let val = F::from_u64(42);

        table.set_unchecked(1, val);

        assert_eq!(table.get(WitnessId(1)).unwrap(), val);
    }

    #[test]
    fn test_get_uninitialized_slot_returns_error() {
        let table = WitnessTable::<F>::new(3);

        let err = table.get(WitnessId(0)).unwrap_err();

        assert!(matches!(
            err,
            CircuitError::WitnessNotSet { witness_id } if witness_id == WitnessId(0)
        ));
    }

    #[test]
    fn test_get_out_of_bounds_returns_error() {
        let table = WitnessTable::<F>::new(2);

        let err = table.get(WitnessId(10)).unwrap_err();

        assert!(matches!(
            err,
            CircuitError::WitnessNotSet { witness_id } if witness_id == WitnessId(10)
        ));
    }

    // ── get_value_unchecked ────────────────────────────────────────

    #[test]
    fn test_get_value_unchecked_returns_reference() {
        let mut table = WitnessTable::<F>::new(2);
        let val = F::from_u64(99);

        table.set_unchecked(1, val);

        assert_eq!(*table.get_value_unchecked(1), val);
    }

    #[test]
    fn test_values_slice_reflects_writes() {
        let mut table = WitnessTable::<F>::new(3);

        table.set_unchecked(0, F::from_u64(1));
        table.set_unchecked(2, F::from_u64(3));

        let vals = table.values();
        assert_eq!(vals.len(), 3);
        assert_eq!(vals[0], F::from_u64(1));
        assert_eq!(vals[2], F::from_u64(3));
    }

    #[test]
    fn test_initialized_slice_reflects_writes() {
        let mut table = WitnessTable::<F>::new(3);

        table.set_unchecked(0, F::from_u64(1));
        table.set_unchecked(2, F::from_u64(3));

        assert_eq!(table.initialized(), &[true, false, true]);
    }
}
