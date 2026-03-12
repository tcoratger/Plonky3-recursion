use alloc::vec::Vec;
use core::fmt;

use serde::{Deserialize, Serialize};

/// Witness ID type - a unique identifier for extension field values in the global witness bus
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WitnessId(pub u32);

impl WitnessId {
    /// Follow rewrite chains to find the canonical witness ID.
    pub fn resolve(self, rewrite: &hashbrown::HashMap<Self, Self>) -> Self {
        let mut cur = self;
        while let Some(&next) = rewrite.get(&cur) {
            cur = next;
        }
        cur
    }

    /// Check if this witness ID has been marked as defined in the tracking vector.
    pub fn is_defined(self, defined: &[bool]) -> bool {
        defined.get(self.0 as usize).copied().unwrap_or(false)
    }

    /// Mark this witness ID as defined, growing the vector if needed.
    pub fn mark_defined(self, defined: &mut Vec<bool>) {
        let idx = self.0 as usize;
        if idx >= defined.len() {
            defined.resize(idx + 1, false);
        }
        defined[idx] = true;
    }
}

impl fmt::Display for WitnessId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "w{}", self.0)
    }
}

/// Handle to an expression in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExprId(pub u32);

impl fmt::Display for ExprId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

impl ExprId {
    /// The zero expression ID - always points to Const(0)
    pub const ZERO: Self = Self(0);
}

/// Handle to a non-primitive operation (for setting private data later)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NonPrimitiveOpId(pub u32);

impl fmt::Display for NonPrimitiveOpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "id{}", self.0)
    }
}

/// Witness allocator for monotonic index assignment
#[derive(Debug, Clone, Default)]
pub struct WitnessAllocator {
    next_idx: u32,
}

impl WitnessAllocator {
    pub const fn new() -> Self {
        Self { next_idx: 0 }
    }

    pub const fn alloc(&mut self) -> WitnessId {
        let idx = WitnessId(self.next_idx);
        self.next_idx += 1;
        idx
    }

    pub const fn witness_count(&self) -> u32 {
        self.next_idx
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec};

    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_witness_id_display() {
        let idx = WitnessId(42);
        assert_eq!(format!("{idx}"), "w42");
    }

    #[test]
    fn test_witness_allocator() {
        let mut allocator = WitnessAllocator::new();

        let w0 = allocator.alloc();
        let w1 = allocator.alloc();
        let w2 = allocator.alloc();

        assert_eq!(w0, WitnessId(0));
        assert_eq!(w1, WitnessId(1));
        assert_eq!(w2, WitnessId(2));
        assert_eq!(allocator.witness_count(), 3);
    }

    #[test]
    fn test_witness_id_is_defined_in_bounds() {
        let defined = vec![false, true, false, true];
        assert!(!WitnessId(0).is_defined(&defined));
        assert!(WitnessId(1).is_defined(&defined));
        assert!(!WitnessId(2).is_defined(&defined));
        assert!(WitnessId(3).is_defined(&defined));
    }

    #[test]
    fn test_witness_id_is_defined_out_of_bounds() {
        let defined = vec![true];
        assert!(!WitnessId(1).is_defined(&defined));
        assert!(!WitnessId(100).is_defined(&defined));
    }

    #[test]
    fn test_witness_id_is_defined_empty() {
        let defined: Vec<bool> = vec![];
        assert!(!WitnessId(0).is_defined(&defined));
    }

    #[test]
    fn test_witness_id_mark_defined_in_bounds() {
        let mut defined = vec![false, false, false];
        WitnessId(1).mark_defined(&mut defined);
        assert_eq!(defined, vec![false, true, false]);
    }

    #[test]
    fn test_witness_id_mark_defined_grows_vector() {
        let mut defined = vec![true];
        WitnessId(3).mark_defined(&mut defined);
        assert_eq!(defined, vec![true, false, false, true]);
    }

    #[test]
    fn test_witness_id_mark_defined_empty_vector() {
        let mut defined: Vec<bool> = vec![];
        WitnessId(2).mark_defined(&mut defined);
        assert_eq!(defined, vec![false, false, true]);
    }

    #[test]
    fn test_witness_id_mark_defined_idempotent() {
        let mut defined = vec![false, false];
        WitnessId(1).mark_defined(&mut defined);
        WitnessId(1).mark_defined(&mut defined);
        assert_eq!(defined, vec![false, true]);
    }

    #[test]
    fn test_witness_id_resolve_empty_rewrite() {
        let rewrite = hashbrown::HashMap::new();
        assert_eq!(WitnessId(5).resolve(&rewrite), WitnessId(5));
    }

    #[test]
    fn test_witness_id_resolve_single_step() {
        let mut rewrite = hashbrown::HashMap::new();
        rewrite.insert(WitnessId(1), WitnessId(2));
        assert_eq!(WitnessId(1).resolve(&rewrite), WitnessId(2));
    }

    #[test]
    fn test_witness_id_resolve_chain() {
        let mut rewrite = hashbrown::HashMap::new();
        rewrite.insert(WitnessId(0), WitnessId(1));
        rewrite.insert(WitnessId(1), WitnessId(2));
        rewrite.insert(WitnessId(2), WitnessId(3));
        assert_eq!(WitnessId(0).resolve(&rewrite), WitnessId(3));
    }

    #[test]
    fn test_witness_id_resolve_no_match() {
        let mut rewrite = hashbrown::HashMap::new();
        rewrite.insert(WitnessId(10), WitnessId(20));
        assert_eq!(WitnessId(5).resolve(&rewrite), WitnessId(5));
    }

    proptest! {
        #[test]
        fn witness_id_mark_defined_then_is_defined(idx in 0u32..256) {
            let wid = WitnessId(idx);
            let mut defined = vec![];

            prop_assert!(!wid.is_defined(&defined), "should not be defined initially");

            wid.mark_defined(&mut defined);

            prop_assert!(wid.is_defined(&defined), "should be defined after marking");
            prop_assert_eq!(defined.len(), idx as usize + 1, "vector should grow to fit");

            // Marking again should be idempotent.
            wid.mark_defined(&mut defined);
            prop_assert!(wid.is_defined(&defined), "should still be defined after re-marking");
            prop_assert_eq!(defined.len(), idx as usize + 1, "vector length should not change");
        }

        #[test]
        fn witness_id_ordering(a in 0u32..u32::MAX, b in 0u32..u32::MAX) {
            let id_a = WitnessId(a);
            let id_b = WitnessId(b);

            if a < b {
                prop_assert!(id_a < id_b, "ordering should match inner value");
            } else if a > b {
                prop_assert!(id_a > id_b, "ordering should match inner value");
            } else {
                prop_assert_eq!(id_a, id_b, "equal values should compare equal");
            }
        }

        #[test]
        fn expr_id_ordering(a in 0u32..u32::MAX, b in 0u32..u32::MAX) {
            let id_a = ExprId(a);
            let id_b = ExprId(b);

            if a < b {
                prop_assert!(id_a < id_b, "ordering should match inner value");
            } else if a > b {
                prop_assert!(id_a > id_b, "ordering should match inner value");
            } else {
                prop_assert_eq!(id_a, id_b, "equal values should compare equal");
            }
        }

        #[test]
        fn non_primitive_op_id_equality(a in 0u32..u32::MAX, b in 0u32..u32::MAX) {
            let id_a1 = NonPrimitiveOpId(a);
            let id_a2 = NonPrimitiveOpId(a);
            let id_b = NonPrimitiveOpId(b);

            prop_assert_eq!(id_a1, id_a2, "same value should be equal");
            if a != b {
                prop_assert_ne!(id_a1, id_b, "different values should not be equal");
            }
        }

        #[test]
        fn witness_allocator_unique(count in 1usize..100) {
            let mut allocator = WitnessAllocator::new();
            let mut seen = hashbrown::HashSet::new();

            for _ in 0..count {
                let id = allocator.alloc();
                prop_assert!(seen.insert(id), "each allocation should be unique");
            }
        }

        #[test]
        fn witness_allocator_count_accurate(count in 0usize..100) {
            let mut allocator = WitnessAllocator::new();

            prop_assert_eq!(allocator.witness_count(), 0, "new allocator should have count 0");

            for i in 1..=count {
                allocator.alloc();
                prop_assert_eq!(
                    allocator.witness_count(),
                    i as u32,
                    "count should increment with each allocation"
                );
            }
        }
    }
}
