use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_field::Field;
use strum_macros::EnumCount;

use super::executor::{HintExecutor, NonPrimitiveExecutor};
use crate::CircuitError;
use crate::circuit::PreprocessedColumns;
use crate::types::{NonPrimitiveOpId, WitnessId};

/// ALU operation kinds for the unified arithmetic table.
///
/// This enum defines the different arithmetic operations that can be performed
/// in a single ALU row, selected by preprocessed selectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AluOpKind {
    /// Addition: out = a + b
    Add,
    /// Multiplication: out = a * b
    Mul,
    /// Boolean check: a * (a - 1) = 0, out = a
    BoolCheck,
    /// Fused multiply-add: out = a * b + c
    MulAdd,
    /// Row-chained Horner accumulator: out = prev_row_out * b + c - a
    ///
    /// Uses the `out` column of the previous ALU row (same lane) as an implicit
    /// accumulator input. Fuses multiply-add-subtract into a single row.
    /// HornerAcc ops within the same chain must be placed in consecutive ALU rows
    /// of the same lane.
    HornerAcc,
}

impl AluOpKind {
    /// Returns the four ALU selector values:
    /// ```text
    ///     [sel_add_vs_mul, sel_bool, sel_muladd, sel_horner]
    /// ```
    pub const fn selectors<F: Field>(&self) -> [F; 4] {
        match self {
            Self::Add => [F::ONE, F::ZERO, F::ZERO, F::ZERO],
            Self::Mul => [F::ZERO; 4],
            Self::BoolCheck => [F::ZERO, F::ONE, F::ZERO, F::ZERO],
            Self::MulAdd => [F::ZERO, F::ZERO, F::ONE, F::ZERO],
            Self::HornerAcc => [F::ZERO, F::ZERO, F::ZERO, F::ONE],
        }
    }
}

/// Circuit operations.
///
/// Operations are distinguised as primitive, non-primitive, and hints:
///
/// - Primitive ops (`Const`, `Public`, `Alu`) are the basic arithmetic building blocks
/// - Non-primitive ops (`NonPrimitiveOpWithExecutor`) are table-backed plugin operations
/// - Hint ops (`Hint`) are non-deterministic witness assignments that do NOT have tables,
///   AIR, or traces; they are purely a convenience for filling witnesses.
#[derive(Debug)]
pub enum Op<F> {
    /// Load a constant value into the witness table
    ///
    /// Sets `witness[out] = val`. Used for literal constants and
    /// supports constant pooling optimization where identical constants
    /// reuse the same witness slot.
    Const { out: WitnessId, val: F },

    /// Load a public input value into the witness table
    ///
    /// Sets `witness[out] = public_inputs[public_pos]`. Public inputs
    /// are values known to both prover and verifier, typically used
    /// for circuit inputs and expected outputs.
    Public { out: WitnessId, public_pos: usize },

    /// Unified ALU operation supporting multiple arithmetic operations.
    ///
    /// The `kind` field determines the operation:
    /// - `Add`: out = a + b
    /// - `Mul`: out = a * b
    /// - `BoolCheck`: a * (a - 1) = 0, out = a
    /// - `MulAdd`: out = a * b + c
    /// - `HornerAcc`: out = acc * b + c - a (acc from previous row's out, same lane)
    Alu {
        kind: AluOpKind,
        a: WitnessId,
        b: WitnessId,
        /// Third operand, used for MulAdd and HornerAcc
        c: Option<WitnessId>,
        out: WitnessId,
        /// Intermediate output for MulAdd: stores a * b when fused from separate mul + add.
        /// For HornerAcc: stores the accumulator WitnessId (previous Horner step's out).
        /// The runner sets this witness value so dependent operations still work.
        intermediate_out: Option<WitnessId>,
    },

    /// Hint operation: non-deterministically fills witness values via a user-provided closure.
    ///
    /// Hints are NOT table-backed:
    /// - they do not have an AIR
    /// - they do not participate in non-primitive traces
    /// - they do not have private data or configs
    ///
    /// They are used for things like bit decompositions and extension-field decompositions.
    Hint {
        /// Input witnesses read by the hint.
        inputs: Vec<WitnessId>,
        /// Output witnesses written by the hint.
        outputs: Vec<WitnessId>,
        /// User-provided executor that implements the hint logic.
        executor: Box<dyn HintExecutor<F>>,
    },

    /// Non-primitive operation with executor-based dispatch
    NonPrimitiveOpWithExecutor {
        inputs: Vec<Vec<WitnessId>>,
        outputs: Vec<Vec<WitnessId>>,
        executor: Box<dyn NonPrimitiveExecutor<F>>,
        /// For private data lookup and error reporting
        op_id: NonPrimitiveOpId,
    },
}

impl<F> Op<F> {
    /// Create an addition operation (convenience wrapper for Op::Alu with AluOpKind::Add).
    pub const fn add(a: WitnessId, b: WitnessId, out: WitnessId) -> Self {
        Self::Alu {
            kind: AluOpKind::Add,
            a,
            b,
            c: None,
            out,
            intermediate_out: None,
        }
    }

    /// Create a multiplication operation (convenience wrapper for Op::Alu with AluOpKind::Mul).
    pub const fn mul(a: WitnessId, b: WitnessId, out: WitnessId) -> Self {
        Self::Alu {
            kind: AluOpKind::Mul,
            a,
            b,
            c: None,
            out,
            intermediate_out: None,
        }
    }

    /// Create a fused multiply-add operation: out = a * b + c.
    pub const fn mul_add(a: WitnessId, b: WitnessId, c: WitnessId, out: WitnessId) -> Self {
        Self::Alu {
            kind: AluOpKind::MulAdd,
            a,
            b,
            c: Some(c),
            out,
            intermediate_out: None,
        }
    }

    /// Create a Horner accumulator: out = acc * b + c - a.
    ///
    /// `acc` is the previous Horner step's output (used by runner for computation).
    /// In the AIR, the accumulator comes implicitly from the previous row's `out` column.
    pub const fn horner_acc(
        a: WitnessId,
        b: WitnessId,
        c: WitnessId,
        out: WitnessId,
        acc: WitnessId,
    ) -> Self {
        Self::Alu {
            kind: AluOpKind::HornerAcc,
            a,
            b,
            c: Some(c),
            out,
            intermediate_out: Some(acc),
        }
    }

    /// Create a boolean check operation: a * (a - 1) = 0.
    pub const fn bool_check(a: WitnessId, b: WitnessId, out: WitnessId) -> Self {
        Self::Alu {
            kind: AluOpKind::BoolCheck,
            a,
            b,
            c: None,
            out,
            intermediate_out: None,
        }
    }

    /// Check if this is an ALU operation of the given kind.
    pub fn is_alu_kind(&self, kind: AluOpKind) -> bool {
        matches!(self, Self::Alu { kind: k, .. } if *k == kind)
    }

    /// Check if this is an addition operation.
    pub fn is_add(&self) -> bool {
        self.is_alu_kind(AluOpKind::Add)
    }

    /// Check if this is a multiplication operation.
    pub fn is_mul(&self) -> bool {
        self.is_alu_kind(AluOpKind::Mul)
    }

    /// Rewrite witness IDs in place using the given map (follows chains to canonical ID).
    /// Used by the optimizer to apply ALU dedup without re-boxing non-primitive executors.
    pub fn apply_witness_rewrite(&mut self, rewrite: &HashMap<WitnessId, WitnessId>) {
        if rewrite.is_empty() {
            return;
        }
        match self {
            Self::Const { out, .. } => *out = out.resolve(rewrite),
            Self::Public { out, .. } => *out = out.resolve(rewrite),
            Self::Alu {
                a,
                b,
                c,
                out,
                intermediate_out,
                ..
            } => {
                *a = a.resolve(rewrite);
                *b = b.resolve(rewrite);
                *c = c.map(|id| id.resolve(rewrite));
                *out = out.resolve(rewrite);
                *intermediate_out = intermediate_out.map(|id| id.resolve(rewrite));
            }
            Self::Hint {
                inputs, outputs, ..
            } => {
                for w in inputs.iter_mut() {
                    *w = w.resolve(rewrite);
                }
                for w in outputs.iter_mut() {
                    *w = w.resolve(rewrite);
                }
            }
            Self::NonPrimitiveOpWithExecutor {
                inputs, outputs, ..
            } => {
                for g in inputs.iter_mut() {
                    for w in g.iter_mut() {
                        *w = w.resolve(rewrite);
                    }
                }
                for g in outputs.iter_mut() {
                    for w in g.iter_mut() {
                        *w = w.resolve(rewrite);
                    }
                }
            }
        }
    }

    /// Preprocess a single operation, updating `defined` and `preprocessed`.
    ///
    /// Each variant handles its own column layout and reader/creator tracking:
    /// - `Const`/`Public`: store D-scaled output index, mark output as defined.
    /// - `Alu`: store selectors + operand indices + reader/creator flags.
    /// - `NonPrimitiveOpWithExecutor`: delegate to executor, track duplicate outputs.
    /// - `Hint`: no-op (hints have no tables or AIR).
    pub(crate) fn preprocess(
        &self,
        defined: &mut Vec<bool>,
        preprocessed: &mut PreprocessedColumns<F>,
    ) -> Result<(), CircuitError>
    where
        F: Field,
    {
        match self {
            Self::Const { out, .. } => {
                let idx = preprocessed.witness_index_as_field(*out);
                preprocessed.primitive[PrimitiveOpType::Const as usize].push(idx);
                out.mark_defined(defined);
            }
            Self::Public { out, .. } => {
                let idx = preprocessed.witness_index_as_field(*out);
                preprocessed.primitive[PrimitiveOpType::Public as usize].push(idx);
                out.mark_defined(defined);
            }
            Self::Alu {
                kind, a, b, c, out, ..
            } => {
                let [sel_add_vs_mul, sel_bool, sel_muladd, sel_horner] = kind.selectors();
                let c_wid = c.unwrap_or(WitnessId(0));
                let d_u32 = preprocessed.d as u32;

                let out_already_defined = out.is_defined(defined);
                let b_already_defined = b.is_defined(defined);
                let a_is_reader = a.is_defined(defined);
                let c_is_reader = c_wid.is_defined(defined);

                let (b_is_creator, out_is_creator) = if !out_already_defined {
                    (false, true) // forward: out is new creator
                } else if !b_already_defined {
                    (true, false) // backward: b is new creator
                } else {
                    (false, false) // all-reader: no new creator
                };

                preprocessed.primitive[PrimitiveOpType::Alu as usize].extend([
                    sel_add_vs_mul,
                    sel_bool,
                    sel_muladd,
                    sel_horner,
                    F::from_u32(a.0 * d_u32),
                    F::from_u32(b.0 * d_u32),
                    F::from_u32(c_wid.0 * d_u32),
                    F::from_u32(out.0 * d_u32),
                    if a_is_reader { F::ONE } else { F::ZERO },
                    if b_is_creator { F::ONE } else { F::ZERO },
                    if c_is_reader { F::ONE } else { F::ZERO },
                    if out_is_creator { F::ONE } else { F::ZERO },
                ]);

                // Mark the new creator as defined.
                if out_is_creator {
                    out.mark_defined(defined);
                } else if b_is_creator {
                    b.mark_defined(defined);
                }

                // Collect readers: non-creator operands that are constrained witnesses.
                // - b is a reader unless it's the creator (forward + all-reader cases).
                // - out is a reader unless it's the creator (backward + all-reader cases).
                // - a and c are readers only if they were previously defined.
                let mut readers = vec![];
                if !b_is_creator {
                    readers.push(*b);
                }
                if !out_is_creator && out_already_defined {
                    readers.push(*out);
                }
                if a_is_reader {
                    readers.push(*a);
                }
                if c_is_reader {
                    readers.push(c_wid);
                }
                preprocessed.increment_ext_reads(&readers);
            }
            Self::NonPrimitiveOpWithExecutor {
                executor,
                inputs,
                outputs,
                ..
            } => {
                executor.preprocess(inputs, outputs, preprocessed)?;

                // Track duplicate non-primitive outputs: first occurrence is a creator,
                // subsequent occurrences are treated as readers on WitnessChecks.
                let op_type = executor.op_type();
                let n_exposed = executor.num_exposed_outputs().unwrap_or(outputs.len());
                for wid in outputs.iter().take(n_exposed).flatten() {
                    if wid.is_defined(defined) {
                        let dup = preprocessed
                            .dup_npo_outputs
                            .entry(op_type.clone())
                            .or_default();
                        let wid_idx = wid.0 as usize;
                        if wid_idx >= dup.len() {
                            dup.resize(wid_idx + 1, false);
                        }
                        dup[wid_idx] = true;
                        preprocessed.increment_ext_reads(&[*wid]);
                    } else {
                        wid.mark_defined(defined);
                    }
                }
            }
            Self::Hint { .. } => {}
        }
        Ok(())
    }
}

#[derive(EnumCount, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveOpType {
    Const = 0,
    Public = 1,
    /// Unified ALU table (combines Add, Mul, BoolCheck, MulAdd)
    Alu = 2,
}

#[allow(clippy::fallible_impl_from)]
impl From<usize> for PrimitiveOpType {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::Const,
            1 => Self::Public,
            2 => Self::Alu,
            _ => panic!("Invalid PrimitiveOpType value: {}", value),
        }
    }
}

impl<F: Field + Clone> Clone for Op<F> {
    fn clone(&self) -> Self {
        match self {
            Self::Const { out, val } => Self::Const {
                out: *out,
                val: *val,
            },
            Self::Public { out, public_pos } => Self::Public {
                out: *out,
                public_pos: *public_pos,
            },
            Self::Alu {
                kind,
                a,
                b,
                c,
                out,
                intermediate_out,
            } => Self::Alu {
                kind: *kind,
                a: *a,
                b: *b,
                c: *c,
                out: *out,
                intermediate_out: *intermediate_out,
            },
            Self::Hint {
                inputs,
                outputs,
                executor,
            } => Self::Hint {
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                executor: executor.boxed(),
            },
            Self::NonPrimitiveOpWithExecutor {
                inputs,
                outputs,
                executor,
                op_id,
            } => Self::NonPrimitiveOpWithExecutor {
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                executor: executor.boxed(),
                op_id: *op_id,
            },
        }
    }
}

impl<F: Field + PartialEq> PartialEq for Op<F> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Const { out: o1, val: v1 }, Self::Const { out: o2, val: v2 }) => {
                o1 == o2 && v1 == v2
            }
            (
                Self::Public {
                    out: o1,
                    public_pos: p1,
                },
                Self::Public {
                    out: o2,
                    public_pos: p2,
                },
            ) => o1 == o2 && p1 == p2,
            (
                Self::Alu {
                    kind: k1,
                    a: a1,
                    b: b1,
                    c: c1,
                    out: o1,
                    intermediate_out: io1,
                },
                Self::Alu {
                    kind: k2,
                    a: a2,
                    b: b2,
                    c: c2,
                    out: o2,
                    intermediate_out: io2,
                },
            ) => k1 == k2 && a1 == a2 && b1 == b2 && c1 == c2 && o1 == o2 && io1 == io2,
            (
                Self::Hint {
                    inputs: i1,
                    outputs: o1,
                    executor: _,
                },
                Self::Hint {
                    inputs: i2,
                    outputs: o2,
                    executor: _,
                },
            ) => {
                // Compare by value layout only; executors are opaque closures.
                i1 == i2 && o1 == o2
            }
            (
                Self::NonPrimitiveOpWithExecutor {
                    inputs: i1,
                    outputs: o1,
                    executor: e1,
                    op_id: id1,
                },
                Self::NonPrimitiveOpWithExecutor {
                    inputs: i2,
                    outputs: o2,
                    executor: e2,
                    op_id: id2,
                },
            ) => i1 == i2 && o1 == o2 && e1.op_type() == e2.op_type() && id1 == id2,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};
    use strum::EnumCount;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_op_partial_eq_different_variants() {
        let const_op = Op::Const {
            out: WitnessId(0),
            val: F::from_u64(5),
        };
        let alu_op = Op::add(WitnessId(0), WitnessId(1), WitnessId(2));
        assert_ne!(const_op, alu_op);
    }

    #[test]
    fn test_op_partial_eq_same_variant_different_values() {
        let alu_op1: Op<F> = Op::add(WitnessId(0), WitnessId(1), WitnessId(2));
        let alu_op2: Op<F> = Op::add(WitnessId(3), WitnessId(4), WitnessId(5));
        assert_ne!(alu_op1, alu_op2);
    }

    #[test]
    fn test_op_partial_eq_alu_add_same_values() {
        let a = WitnessId(0);
        let b = WitnessId(1);
        let out = WitnessId(2);
        let alu_op1: Op<F> = Op::add(a, b, out);
        let alu_op2: Op<F> = Op::add(a, b, out);
        assert_eq!(alu_op1, alu_op2);
    }

    #[test]
    fn test_op_partial_eq_const_different_values() {
        let out = WitnessId(0);
        let const_op1: Op<F> = Op::Const {
            out,
            val: F::from_u64(10),
        };
        let const_op2: Op<F> = Op::Const {
            out,
            val: F::from_u64(20),
        };
        assert_ne!(const_op1, const_op2);
    }

    #[test]
    fn test_op_partial_eq_const_different_outputs() {
        let val = F::from_u64(42);
        let const_op1: Op<F> = Op::Const {
            out: WitnessId(0),
            val,
        };
        let const_op2: Op<F> = Op::Const {
            out: WitnessId(1),
            val,
        };
        assert_ne!(const_op1, const_op2);
    }

    #[test]
    fn test_op_partial_eq_const_same_values() {
        let out = WitnessId(0);
        let val = F::from_u64(99);
        let const_op1: Op<F> = Op::Const { out, val };
        let const_op2: Op<F> = Op::Const { out, val };
        assert_eq!(const_op1, const_op2);
    }

    #[test]
    fn test_op_partial_eq_public_different_positions() {
        let out = WitnessId(0);
        let public_op1: Op<F> = Op::Public { out, public_pos: 0 };
        let public_op2: Op<F> = Op::Public { out, public_pos: 1 };
        assert_ne!(public_op1, public_op2);
    }

    #[test]
    fn test_op_partial_eq_public_same_values() {
        let out = WitnessId(5);
        let public_pos = 3;
        let public_op1: Op<F> = Op::Public { out, public_pos };
        let public_op2: Op<F> = Op::Public { out, public_pos };
        assert_eq!(public_op1, public_op2);
    }

    #[test]
    fn test_op_partial_eq_alu_mul_different_values() {
        let mul_op1: Op<F> = Op::mul(WitnessId(0), WitnessId(1), WitnessId(2));
        let mul_op2: Op<F> = Op::mul(WitnessId(10), WitnessId(11), WitnessId(12));
        assert_ne!(mul_op1, mul_op2);
    }

    #[test]
    fn test_op_partial_eq_alu_mul_same_values() {
        let a = WitnessId(7);
        let b = WitnessId(8);
        let out = WitnessId(9);
        let mul_op1: Op<F> = Op::mul(a, b, out);
        let mul_op2: Op<F> = Op::mul(a, b, out);
        assert_eq!(mul_op1, mul_op2);
    }

    #[test]
    fn test_op_partial_eq_alu_partial_match() {
        let alu_op1: Op<F> = Op::add(WitnessId(0), WitnessId(1), WitnessId(2));
        let alu_op2: Op<F> = Op::add(WitnessId(0), WitnessId(1), WitnessId(99));
        assert_ne!(alu_op1, alu_op2);
    }

    #[test]
    fn test_op_partial_eq_alu_different_kinds() {
        let add_op: Op<F> = Op::add(WitnessId(0), WitnessId(1), WitnessId(2));
        let mul_op: Op<F> = Op::mul(WitnessId(0), WitnessId(1), WitnessId(2));
        assert_ne!(add_op, mul_op);
    }

    #[test]
    fn test_op_partial_eq_alu_muladd() {
        let muladd_op1: Op<F> = Op::mul_add(WitnessId(0), WitnessId(1), WitnessId(2), WitnessId(3));
        let muladd_op2: Op<F> = Op::mul_add(WitnessId(0), WitnessId(1), WitnessId(2), WitnessId(3));
        assert_eq!(muladd_op1, muladd_op2);
    }

    #[test]
    #[should_panic(expected = "Invalid PrimitiveOpType value")]
    fn test_primitive_op_type_invalid_conversion() {
        let _ = PrimitiveOpType::from(999);
    }

    #[test]
    fn test_preprocess_const_stores_index_and_marks_defined() {
        let op: Op<F> = Op::Const {
            out: WitnessId(3),
            val: F::from_u64(42),
        };
        let mut defined = vec![false; 4];
        let mut prep = PreprocessedColumns::new();

        op.preprocess(&mut defined, &mut prep).unwrap();

        assert_eq!(
            prep.primitive[PrimitiveOpType::Const as usize],
            vec![F::from_u32(3)]
        );
        assert_eq!(defined, vec![false, false, false, true]);
        // Other primitive columns untouched.
        assert!(prep.primitive[PrimitiveOpType::Public as usize].is_empty());
        assert!(prep.primitive[PrimitiveOpType::Alu as usize].is_empty());
    }

    #[test]
    fn test_preprocess_const_with_d_scaling() {
        let op: Op<F> = Op::Const {
            out: WitnessId(2),
            val: F::from_u64(7),
        };
        let mut defined = vec![];
        let mut prep = PreprocessedColumns::new_with_d(4);

        op.preprocess(&mut defined, &mut prep).unwrap();

        // WitnessId(2) * d=4 → 8
        assert_eq!(
            prep.primitive[PrimitiveOpType::Const as usize],
            vec![F::from_u32(8)]
        );
        assert_eq!(defined, vec![false, false, true]);
    }

    #[test]
    fn test_preprocess_public_stores_index_and_marks_defined() {
        let op: Op<F> = Op::Public {
            out: WitnessId(1),
            public_pos: 0,
        };
        let mut defined = vec![false; 2];
        let mut prep = PreprocessedColumns::new();

        op.preprocess(&mut defined, &mut prep).unwrap();

        assert_eq!(
            prep.primitive[PrimitiveOpType::Public as usize],
            vec![F::from_u32(1)]
        );
        assert_eq!(defined, vec![false, true]);
        assert!(prep.primitive[PrimitiveOpType::Const as usize].is_empty());
    }

    #[test]
    fn test_preprocess_alu_forward_add() {
        // Forward case: out not yet defined → out_is_creator=1, b_is_creator=0.
        // - a(0) defined,
        // - b(1) undefined,
        // - out(2) undefined.
        let op: Op<F> = Op::add(WitnessId(0), WitnessId(1), WitnessId(2));
        let mut defined = vec![true, false, false];
        let mut prep = PreprocessedColumns::new();

        op.preprocess(&mut defined, &mut prep).unwrap();

        let f = F::from_u32;
        assert_eq!(
            prep.primitive[PrimitiveOpType::Alu as usize],
            vec![
                F::ONE,  // sel_add_vs_mul
                F::ZERO, // sel_bool
                F::ZERO, // sel_muladd
                F::ZERO, // sel_horner
                f(0),    // a_idx
                f(1),    // b_idx
                f(0),    // c_idx (None → WitnessId(0))
                f(2),    // out_idx
                F::ONE,  // a_is_reader (a=0 defined)
                F::ZERO, // b_is_creator
                F::ONE,  // c_is_reader (c defaults to WitnessId(0), which is defined)
                F::ONE,  // out_is_creator
            ]
        );
        // out(2) now defined.
        assert_eq!(defined, vec![true, false, true]);
        // Readers: b(1), a(0), c(0).
        assert_eq!(prep.ext_reads, vec![2, 1]);
    }

    #[test]
    fn test_preprocess_alu_forward_mul() {
        // Forward Mul: all selectors zero.
        let op: Op<F> = Op::mul(WitnessId(0), WitnessId(1), WitnessId(2));
        let mut defined = vec![false; 3];
        let mut prep = PreprocessedColumns::new();

        op.preprocess(&mut defined, &mut prep).unwrap();

        let alu = &prep.primitive[PrimitiveOpType::Alu as usize];
        // First 4 values are selectors — all zero for Mul.
        assert_eq!(&alu[..4], &[F::ZERO; 4]);
        assert_eq!(defined, vec![false, false, true]);
    }

    #[test]
    fn test_preprocess_alu_backward() {
        // Backward case: out(2) already defined, b(1) not yet defined.
        // → b_is_creator=1, out_is_creator=0.
        let op: Op<F> = Op::add(WitnessId(0), WitnessId(1), WitnessId(2));
        let mut defined = vec![false, false, true];
        let mut prep = PreprocessedColumns::new();

        op.preprocess(&mut defined, &mut prep).unwrap();

        let alu = &prep.primitive[PrimitiveOpType::Alu as usize];
        // b_is_creator=1 (index 9), out_is_creator=0 (index 11).
        assert_eq!(alu[9], F::ONE);
        assert_eq!(alu[11], F::ZERO);
        // b(1) now defined.
        assert_eq!(defined, vec![false, true, true]);
        // Readers: out(2) always, a(0) not defined → not a reader.
        assert_eq!(prep.ext_reads, vec![0, 0, 1]);
    }

    #[test]
    fn test_preprocess_alu_all_reader() {
        // All-reader case: both out(2) and b(1) already defined.
        // → b_is_creator=0, out_is_creator=0.
        let op: Op<F> = Op::add(WitnessId(0), WitnessId(1), WitnessId(2));
        let mut defined = vec![true, true, true];
        let mut prep = PreprocessedColumns::new();

        op.preprocess(&mut defined, &mut prep).unwrap();

        let alu = &prep.primitive[PrimitiveOpType::Alu as usize];
        // b_is_creator=0 (index 9), out_is_creator=0 (index 11).
        assert_eq!(alu[9], F::ZERO);
        assert_eq!(alu[11], F::ZERO);
        // a_is_reader=1 (index 8), c_is_reader=1 (index 10, c defaults to WitnessId(0) which is defined).
        assert_eq!(alu[8], F::ONE);
        assert_eq!(alu[10], F::ONE);
        // defined unchanged.
        assert_eq!(defined, vec![true, true, true]);
        // Readers: b(1), out(2), a(0), c(0).
        assert_eq!(prep.ext_reads, vec![2, 1, 1]);
    }

    #[test]
    fn test_preprocess_alu_muladd_selectors() {
        let op: Op<F> = Op::mul_add(WitnessId(0), WitnessId(1), WitnessId(2), WitnessId(3));
        let mut defined = vec![false; 4];
        let mut prep = PreprocessedColumns::new();

        op.preprocess(&mut defined, &mut prep).unwrap();

        let alu = &prep.primitive[PrimitiveOpType::Alu as usize];
        assert_eq!(
            &alu[..4],
            &[F::ZERO, F::ZERO, F::ONE, F::ZERO] // sel_muladd=1
        );
        // c_idx = WitnessId(2) * d=1 → 2.
        assert_eq!(alu[6], F::from_u32(2));
    }

    #[test]
    fn test_preprocess_alu_bool_check_selectors() {
        let op: Op<F> = Op::bool_check(WitnessId(0), WitnessId(1), WitnessId(2));
        let mut defined = vec![false; 3];
        let mut prep = PreprocessedColumns::new();

        op.preprocess(&mut defined, &mut prep).unwrap();

        let alu = &prep.primitive[PrimitiveOpType::Alu as usize];
        assert_eq!(
            &alu[..4],
            &[F::ZERO, F::ONE, F::ZERO, F::ZERO] // sel_bool=1
        );
    }

    #[test]
    fn test_preprocess_alu_d_scaling() {
        // With d=4, witness indices are multiplied by 4.
        let op: Op<F> = Op::add(WitnessId(2), WitnessId(3), WitnessId(5));
        let mut defined = vec![false; 6];
        let mut prep = PreprocessedColumns::new_with_d(4);

        op.preprocess(&mut defined, &mut prep).unwrap();

        let alu = &prep.primitive[PrimitiveOpType::Alu as usize];
        let f = F::from_u32;
        assert_eq!(alu[4], f(8)); // a=2*4
        assert_eq!(alu[5], f(12)); // b=3*4
        assert_eq!(alu[6], f(0)); // c=0*4
        assert_eq!(alu[7], f(20)); // out=5*4
    }

    #[test]
    fn test_preprocess_alu_a_and_c_undefined_not_readers() {
        // a(10) and c not defined → a_is_reader=0, c_is_reader=0.
        // Only b is a reader in forward case.
        let op: Op<F> = Op::add(WitnessId(10), WitnessId(5), WitnessId(3));
        let mut defined = vec![false; 11];
        let mut prep = PreprocessedColumns::new();

        op.preprocess(&mut defined, &mut prep).unwrap();

        let alu = &prep.primitive[PrimitiveOpType::Alu as usize];
        assert_eq!(alu[8], F::ZERO); // a_is_reader=0
        assert_eq!(alu[10], F::ZERO); // c_is_reader=0
        // Only b(5) is a reader.
        assert_eq!(prep.ext_reads, vec![0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn test_preprocess_selectors_all_kinds() {
        let cases: [(AluOpKind, [F; 4]); 5] = [
            (AluOpKind::Add, [F::ONE, F::ZERO, F::ZERO, F::ZERO]),
            (AluOpKind::Mul, [F::ZERO, F::ZERO, F::ZERO, F::ZERO]),
            (AluOpKind::BoolCheck, [F::ZERO, F::ONE, F::ZERO, F::ZERO]),
            (AluOpKind::MulAdd, [F::ZERO, F::ZERO, F::ONE, F::ZERO]),
            (AluOpKind::HornerAcc, [F::ZERO, F::ZERO, F::ZERO, F::ONE]),
        ];
        for (kind, expected) in &cases {
            assert_eq!(
                kind.selectors::<F>(),
                *expected,
                "selectors mismatch for {kind:?}"
            );
        }
    }

    #[test]
    fn test_preprocess_preserves_other_primitive_columns() {
        // Preprocessing a Const should not touch Public or Alu columns, and vice versa.
        let mut defined = vec![false; 3];
        let mut prep = PreprocessedColumns::new();

        Op::<F>::Const {
            out: WitnessId(0),
            val: F::from_u64(1),
        }
        .preprocess(&mut defined, &mut prep)
        .unwrap();
        Op::<F>::Public {
            out: WitnessId(1),
            public_pos: 0,
        }
        .preprocess(&mut defined, &mut prep)
        .unwrap();
        Op::<F>::add(WitnessId(0), WitnessId(1), WitnessId(2))
            .preprocess(&mut defined, &mut prep)
            .unwrap();

        assert_eq!(prep.primitive.len(), PrimitiveOpType::COUNT);
        assert_eq!(prep.primitive[PrimitiveOpType::Const as usize].len(), 1);
        assert_eq!(prep.primitive[PrimitiveOpType::Public as usize].len(), 1);
        assert_eq!(prep.primitive[PrimitiveOpType::Alu as usize].len(), 12);
    }
}
