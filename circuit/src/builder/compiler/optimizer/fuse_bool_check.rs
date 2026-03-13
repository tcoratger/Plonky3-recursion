use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_field::Field;

use super::analysis::OpDef;
use crate::ops::{AluOpKind, Op};
use crate::types::WitnessId;

/// Detects `b * (b - 1) = 0` patterns and rewrites them as `BoolCheck` ops.
///
/// The pattern appears as:
/// 1. `neg_one_val = mul(one, const_neg_1)`
/// 2. `b_minus_1 = add(b, neg_one_val)`
/// 3. `product = mul(b, b_minus_1)` — **this** mul becomes a BoolCheck.
pub(super) struct BoolCheckFusion<F> {
    defs: HashMap<WitnessId, OpDef<F>>,
}

impl<F: Field> BoolCheckFusion<F> {
    /// Scans ops to build a definition map.
    pub(super) fn new(ops: &[Op<F>]) -> Self {
        let mut defs = HashMap::with_capacity(ops.len());

        for op in ops {
            match op {
                Op::Const { out, val } => {
                    defs.insert(*out, OpDef::Const(*val));
                }
                Op::Alu {
                    kind: AluOpKind::Add,
                    a,
                    b,
                    out,
                    ..
                } => {
                    defs.insert(*out, OpDef::Add { a: *a, b: *b });
                }
                Op::Alu {
                    kind: AluOpKind::Mul,
                    a,
                    b,
                    out,
                    ..
                } => {
                    defs.insert(*out, OpDef::Mul { a: *a, b: *b });
                }
                Op::Alu { out, .. } | Op::Public { out, .. } => {
                    defs.insert(*out, OpDef::Other);
                }
                Op::NonPrimitiveOpWithExecutor { .. } | Op::Hint { .. } => {}
            }
        }

        Self { defs }
    }

    /// Rewrites matching muls into BoolCheck ops; passes everything else through.
    pub(super) fn run(self, ops: Vec<Op<F>>) -> Vec<Op<F>> {
        ops.into_iter()
            .map(|op| {
                if let Op::Alu {
                    kind: AluOpKind::Mul,
                    a,
                    b,
                    c: None,
                    out,
                    ..
                } = &op
                    && let Some(input) = self.detect_pattern(*a, *b)
                {
                    return Op::bool_check(input, *b, *out);
                }
                op
            })
            .collect()
    }

    /// Checks whether `mul(a, b)` matches `X * (X - 1)`. Returns `Some(X)`.
    fn detect_pattern(&self, mul_a: WitnessId, mul_b: WitnessId) -> Option<WitnessId> {
        self.is_x_times_x_minus_one(mul_a, mul_b)
            .or_else(|| self.is_x_times_x_minus_one(mul_b, mul_a))
    }

    /// Returns `Some(x)` when `candidate == x` and `other == add(x, -1)`.
    fn is_x_times_x_minus_one(&self, x: WitnessId, other: WitnessId) -> Option<WitnessId> {
        let OpDef::Add { a, b } = self.defs.get(&other)? else {
            return None;
        };
        (*a == x && self.evaluates_to_neg_one(*b)).then_some(x)
    }

    /// Whether `id` is known to hold the value `-1` (direct constant or `1 * (-1)`).
    fn evaluates_to_neg_one(&self, id: WitnessId) -> bool {
        match self.defs.get(&id) {
            Some(OpDef::Const(val)) => *val == -F::ONE,
            Some(OpDef::Mul { a, b }) => self.is_const_product_neg_one(*a, *b),
            _ => false,
        }
    }

    /// Whether `a * b` is a constant multiplication that equals `-1`.
    fn is_const_product_neg_one(&self, a: WitnessId, b: WitnessId) -> bool {
        self.const_value(a)
            .zip(self.const_value(b))
            .is_some_and(|(x, y)| *x * *y == -F::ONE)
    }

    fn const_value(&self, id: WitnessId) -> Option<&F> {
        self.defs.get(&id).and_then(OpDef::const_value)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};

    use super::*;
    use crate::ops::Op;

    type F = BabyBear;

    #[test]
    fn test_bool_check_fusion() {
        let b = WitnessId(0);
        let one = WitnessId(1);
        let neg_one = WitnessId(2);
        let neg_one_val = WitnessId(3);
        let b_minus_one = WitnessId(4);
        let product = WitnessId(5);

        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: one,
                val: F::ONE,
            },
            Op::Const {
                out: neg_one,
                val: -F::ONE,
            },
            Op::mul(one, neg_one, neg_one_val),
            Op::add(b, neg_one_val, b_minus_one),
            Op::mul(b, b_minus_one, product),
        ];

        let fused = BoolCheckFusion::new(&ops).run(ops);

        let count = fused
            .iter()
            .filter(|op| op.is_alu_kind(AluOpKind::BoolCheck))
            .count();
        assert_eq!(count, 1, "Expected 1 BoolCheck");
    }

    #[test]
    fn test_no_false_positive_bool_check() {
        let ops: Vec<Op<F>> = vec![Op::mul(WitnessId(0), WitnessId(1), WitnessId(2))];

        let fused = BoolCheckFusion::new(&ops).run(ops.clone());
        assert_eq!(fused, ops);
    }

    #[test]
    fn test_symmetric_pattern() {
        // mul(b_minus_one, b) instead of mul(b, b_minus_one)
        let b = WitnessId(0);
        let neg_one = WitnessId(1);
        let b_minus_one = WitnessId(2);
        let product = WitnessId(3);

        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: neg_one,
                val: -F::ONE,
            },
            Op::add(b, neg_one, b_minus_one),
            Op::mul(b_minus_one, b, product), // reversed operand order
        ];

        let fused = BoolCheckFusion::new(&ops).run(ops);
        assert_eq!(
            fused
                .iter()
                .filter(|op| op.is_alu_kind(AluOpKind::BoolCheck))
                .count(),
            1
        );
    }

    #[test]
    fn test_direct_const_neg_one() {
        // b - 1 via direct -1 const (no mul intermediary)
        let b = WitnessId(0);
        let neg_one = WitnessId(1);
        let b_minus_one = WitnessId(2);
        let product = WitnessId(3);

        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: neg_one,
                val: -F::ONE,
            },
            Op::add(b, neg_one, b_minus_one),
            Op::mul(b, b_minus_one, product),
        ];

        let fused = BoolCheckFusion::new(&ops).run(ops);
        assert_eq!(
            fused
                .iter()
                .filter(|op| op.is_alu_kind(AluOpKind::BoolCheck))
                .count(),
            1
        );
    }

    #[test]
    fn test_non_alu_passthrough() {
        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: WitnessId(0),
                val: F::ONE,
            },
            Op::add(WitnessId(0), WitnessId(1), WitnessId(2)),
        ];

        let fused = BoolCheckFusion::new(&ops).run(ops.clone());
        assert_eq!(fused, ops);
    }

    #[test]
    fn test_empty_input() {
        let ops: Vec<Op<F>> = vec![];
        let fused = BoolCheckFusion::new(&ops).run(ops);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_add_not_neg_one_no_fusion() {
        // b + 2, then mul(b, b+2) — should NOT be fused
        let b = WitnessId(0);
        let two = WitnessId(1);
        let b_plus_two = WitnessId(2);
        let product = WitnessId(3);

        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: two,
                val: F::TWO,
            },
            Op::add(b, two, b_plus_two),
            Op::mul(b, b_plus_two, product),
        ];

        let fused = BoolCheckFusion::new(&ops).run(ops);
        assert_eq!(
            fused
                .iter()
                .filter(|op| op.is_alu_kind(AluOpKind::BoolCheck))
                .count(),
            0
        );
    }
}
