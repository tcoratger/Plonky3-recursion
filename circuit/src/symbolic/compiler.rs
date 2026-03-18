//! Symbolic constraint compiler.

use alloc::vec::Vec;

use p3_air::{BaseLeaf, ExtLeaf, SymbolicExpr, SymbolicExprNode, SymbolicExpressionExt};
use p3_field::{ExtensionField, Field};
use p3_uni_stark::SymbolicExpression;

use super::targets::{ColumnsTargets, RowSelectorsTargets};
use crate::{CircuitBuilder, ExprId};

/// Compiles symbolic AIR constraints into circuit operations.
///
/// Holds the row selectors and column targets that remain constant
/// across all constraint compilations within a single AIR evaluation.
///
/// # Algorithm
///
/// Builds a [`SymbolicExprDag`] from expression trees, then iterates
/// the topologically-sorted nodes in a single linear pass. Shared
/// sub-expressions are automatically deduplicated by the DAG construction.
pub struct SymbolicCompiler<'a> {
    /// Lagrange selector targets (first row, last row, transition).
    row_selectors: RowSelectorsTargets,
    /// Column targets for every trace category.
    columns: &'a ColumnsTargets<'a>,
}

impl<'a> SymbolicCompiler<'a> {
    /// Create a compiler bound to the given selectors and column targets.
    pub const fn new(row_selectors: RowSelectorsTargets, columns: &'a ColumnsTargets<'a>) -> Self {
        Self {
            row_selectors,
            columns,
        }
    }

    /// Compile a base-field symbolic expression into circuit operations.
    ///
    /// Constants are lifted from the base field into the extension field.
    /// When both fields are the same, this is the identity.
    ///
    /// The pointer-keyed cache should be shared across calls so that
    /// sub-expressions appearing in multiple constraints are compiled only once.
    pub fn compile_base<CF: Field, EF: ExtensionField<CF>>(
        &self,
        expr: &SymbolicExpression<CF>,
        circuit: &mut CircuitBuilder<EF>,
        cache: &mut hashbrown::HashMap<*const SymbolicExpression<CF>, ExprId>,
    ) -> ExprId {
        // Fast path: already compiled this exact expression (by pointer).
        let root_key = expr as *const _;
        if let Some(&cached) = cache.get(&root_key) {
            return cached;
        }

        // Build a topologically-sorted DAG from the expression tree.
        // This deduplicates shared sub-expressions via Arc pointer identity.
        let dag = SymbolicExpr::flatten_to_dag(core::slice::from_ref(expr));

        // Single linear pass: map each DAG node to a circuit ExprId.
        let mut ids: Vec<ExprId> = Vec::with_capacity(dag.nodes.len());
        for node in &dag.nodes {
            let id = match node {
                SymbolicExprNode::Leaf(BaseLeaf::Constant(c)) => {
                    circuit.define_const(EF::from(*c))
                }
                SymbolicExprNode::Leaf(BaseLeaf::Variable(v)) => {
                    self.columns.resolve_base_var(&v.entry, v.index)
                }
                SymbolicExprNode::Leaf(BaseLeaf::IsFirstRow) => self.row_selectors.is_first_row,
                SymbolicExprNode::Leaf(BaseLeaf::IsLastRow) => self.row_selectors.is_last_row,
                SymbolicExprNode::Leaf(BaseLeaf::IsTransition) => self.row_selectors.is_transition,
                SymbolicExprNode::Add { left, right, .. } => circuit.add(ids[*left], ids[*right]),
                SymbolicExprNode::Sub { left, right, .. } => circuit.sub(ids[*left], ids[*right]),
                SymbolicExprNode::Neg { idx, .. } => {
                    let zero = circuit.define_const(EF::ZERO);
                    circuit.sub(zero, ids[*idx])
                }
                SymbolicExprNode::Mul { left, right, .. } => circuit.mul(ids[*left], ids[*right]),
            };
            ids.push(id);
        }

        let result = ids[dag.constraint_idx[0]];
        cache.insert(root_key, result);
        result
    }

    /// Compile an extension-field symbolic expression into circuit operations.
    ///
    /// Extension expressions may contain three kinds of leaves:
    /// - Base-field sub-trees, compiled via the base-field path.
    /// - Extension-field variables (permutation columns and challenges).
    /// - Extension-field constants.
    ///
    /// Two separate caches are needed: one for base-field nodes
    /// and one for extension-field nodes.
    pub fn compile_ext<F: Field, EF: ExtensionField<F>>(
        &self,
        expr: &SymbolicExpressionExt<F, EF>,
        circuit: &mut CircuitBuilder<EF>,
        base_cache: &mut hashbrown::HashMap<*const SymbolicExpression<F>, ExprId>,
        ext_cache: &mut hashbrown::HashMap<*const SymbolicExpressionExt<F, EF>, ExprId>,
    ) -> ExprId {
        // Fast path: already compiled this exact expression (by pointer).
        let root_key = expr as *const _;
        if let Some(&cached) = ext_cache.get(&root_key) {
            return cached;
        }

        // Build a topologically-sorted DAG from the extension expression tree.
        let dag = SymbolicExpr::flatten_to_dag(core::slice::from_ref(expr));

        // Single linear pass: map each DAG node to a circuit ExprId.
        let mut ids: Vec<ExprId> = Vec::with_capacity(dag.nodes.len());
        for node in &dag.nodes {
            let id = match node {
                SymbolicExprNode::Leaf(ExtLeaf::Base(base_expr)) => {
                    self.compile_base(base_expr, circuit, base_cache)
                }
                SymbolicExprNode::Leaf(ExtLeaf::ExtVariable(v)) => {
                    self.columns.resolve_ext_var(&v.entry, v.index)
                }
                SymbolicExprNode::Leaf(ExtLeaf::ExtConstant(c)) => circuit.define_const(*c),
                SymbolicExprNode::Add { left, right, .. } => circuit.add(ids[*left], ids[*right]),
                SymbolicExprNode::Sub { left, right, .. } => circuit.sub(ids[*left], ids[*right]),
                SymbolicExprNode::Neg { idx, .. } => {
                    let zero = circuit.define_const(EF::ZERO);
                    circuit.sub(zero, ids[*idx])
                }
                SymbolicExprNode::Mul { left, right, .. } => circuit.mul(ids[*left], ids[*right]),
            };
            ids.push(id);
        }

        let result = ids[dag.constraint_idx[0]];
        ext_cache.insert(root_key, result);
        result
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::{
        Air, AirLayout, BaseAir, BaseEntry, BaseLeaf, ExtEntry, ExtLeaf, RowWindow,
        SymbolicExpressionExt, SymbolicVariable, SymbolicVariableExt,
    };
    use p3_matrix::dense::RowMajorMatrixView;
    use p3_matrix::stack::VerticalPair;
    use p3_test_utils::baby_bear_params::*;
    use p3_uni_stark::{SymbolicExpression, VerifierConstraintFolder, get_symbolic_constraints};
    use rand::rngs::SmallRng;
    use rand::{Rng, RngExt, SeedableRng};

    use super::*;
    use crate::test_utils::{FibonacciAir, NUM_FIBONACCI_COLS};
    use crate::{CircuitBuilder, CircuitError};

    /// Verify that symbolic compilation of AIR constraints produces
    /// the same folded value as direct evaluation with concrete field elements.
    ///
    /// # Algorithm
    ///
    /// - Evaluate the constraints with random trace values and selectors
    ///   to get a reference folded value.
    /// - Compile the same constraints symbolically into a circuit.
    /// - Run the circuit with identical public inputs and assert that
    ///   the output matches the reference value.
    #[test]
    fn test_symbolic_compiler() -> Result<(), CircuitError> {
        let mut rng = SmallRng::seed_from_u64(1);
        let x = 21;

        // Public inputs for the Fibonacci AIR: fib(0)=0, fib(1)=1, target=21.
        let pis = vec![F::ZERO, F::ONE, F::from_u64(x)];
        let pis_ext = pis
            .iter()
            .map(|c| Challenge::from_prime_subfield(*c))
            .collect::<Vec<_>>();

        let air = FibonacciAir {};
        // Random linear-combination challenge for constraint folding.
        let alpha = Challenge::from_u64(rng.next_u64());

        // Generate random trace values for a single row pair.
        let width = <FibonacciAir as BaseAir<F>>::width(&air);
        let mut trace_local = Vec::with_capacity(width);
        let mut trace_next = Vec::with_capacity(width);
        for _ in 0..width {
            trace_local.push(Challenge::from_prime_subfield(rng.random()));
            trace_next.push(Challenge::from_prime_subfield(rng.random()));
        }
        let main = VerticalPair::new(
            RowMajorMatrixView::new_row(&trace_local),
            RowMajorMatrixView::new_row(&trace_next),
        );

        // Random selector values (is_first_row, is_last_row, is_transition).
        let sels = [
            Challenge::from_u64(rng.next_u64()),
            Challenge::from_u64(rng.next_u64()),
            Challenge::from_u64(rng.next_u64()),
        ];

        let preprocessed = VerticalPair::new(
            RowMajorMatrixView::new(&[], 0),
            RowMajorMatrixView::new(&[], 0),
        );
        let preprocessed_window =
            RowWindow::from_two_rows(preprocessed.top.values, preprocessed.bottom.values);

        // Compute the reference folded constraint value by direct evaluation.
        let mut folder: VerifierConstraintFolder<'_, MyConfig> = VerifierConstraintFolder {
            main,
            preprocessed,
            preprocessed_window,
            public_values: &pis,
            is_first_row: sels[0],
            is_last_row: sels[1],
            is_transition: sels[2],
            alpha,
            accumulator: Challenge::ZERO,
        };
        air.eval(&mut folder);
        let folded_constraints = folder.accumulator;

        let layout = AirLayout {
            preprocessed_width: 3,
            main_width: BaseAir::<F>::width(&air),
            num_public_values: BaseAir::<F>::num_public_values(&air),
            permutation_width: 0,
            num_permutation_challenges: 0,
            num_permutation_values: 0,
            num_periodic_columns: 0,
        };

        // Extract symbolic constraint trees from the AIR definition.
        let symbolic_constraints: Vec<SymbolicExpression<Challenge>> =
            get_symbolic_constraints(&air, layout);

        // Fold all symbolic constraints into a single expression: c_0 + alpha * c_1 + ...
        let folded_symbolic_constraints = {
            let mut acc =
                SymbolicExpression::<Challenge>::Leaf(BaseLeaf::Constant(Challenge::ZERO));
            let ch = SymbolicExpression::Leaf(BaseLeaf::Constant(alpha));
            for s_c in symbolic_constraints.iter() {
                acc = ch.clone() * acc;
                acc += s_c.clone();
            }
            acc
        };

        // Build a circuit with public inputs for selectors, public values, and trace columns.
        let mut circuit = CircuitBuilder::new();
        let circuit_sels = [
            circuit.public_input(),
            circuit.public_input(),
            circuit.public_input(),
        ];
        let circuit_public_values = [
            circuit.public_input(),
            circuit.public_input(),
            circuit.public_input(),
        ];
        let mut circuit_local_values = Vec::with_capacity(NUM_FIBONACCI_COLS);
        let mut circuit_next_values = Vec::with_capacity(NUM_FIBONACCI_COLS);
        for _ in 0..NUM_FIBONACCI_COLS {
            circuit_local_values.push(circuit.public_input());
            circuit_next_values.push(circuit.public_input());
        }

        let row_selectors = RowSelectorsTargets {
            is_first_row: circuit_sels[0],
            is_last_row: circuit_sels[1],
            is_transition: circuit_sels[2],
        };
        let columns = ColumnsTargets {
            challenges: &[],
            public_values: &circuit_public_values,
            permutation_local_values: &[],
            permutation_next_values: &[],
            permutation_values: &[],
            local_prep_values: &[],
            next_prep_values: &[],
            local_values: &circuit_local_values,
            next_values: &circuit_next_values,
        };

        // Compile the folded symbolic expression into circuit gates.
        let compiler = SymbolicCompiler::new(row_selectors, &columns);
        let mut cache = hashbrown::HashMap::new();
        let sum = compiler.compile_base(&folded_symbolic_constraints, &mut circuit, &mut cache);

        // Assert that the circuit output equals the reference folded value.
        let final_result_const = circuit.define_const(folded_constraints);
        circuit.connect(final_result_const, sum);

        // Assemble all public inputs in the order expected by the circuit.
        let mut all_public_values = sels.to_vec();
        all_public_values.extend_from_slice(&pis_ext);
        for i in 0..NUM_FIBONACCI_COLS {
            all_public_values.push(trace_local[i]);
            all_public_values.push(trace_next[i]);
        }

        let builder = circuit.build().unwrap();
        let mut builder = builder.runner();
        builder.set_public_inputs(&all_public_values).unwrap();
        let _ = builder.run()?;

        Ok(())
    }

    /// Helper: build a minimal compiler with the given number of main columns
    /// and public values. Returns (circuit, local_ids, next_ids, pub_ids, sel_ids).
    #[allow(clippy::type_complexity)]
    fn setup(
        n_cols: usize,
        n_pubs: usize,
    ) -> (
        CircuitBuilder<Challenge>,
        Vec<ExprId>,
        Vec<ExprId>,
        Vec<ExprId>,
        [ExprId; 3],
    ) {
        let mut circuit = CircuitBuilder::<Challenge>::new();
        let sels = [
            circuit.public_input(),
            circuit.public_input(),
            circuit.public_input(),
        ];
        let pubs: Vec<_> = (0..n_pubs).map(|_| circuit.public_input()).collect();
        let local: Vec<_> = (0..n_cols).map(|_| circuit.public_input()).collect();
        let next: Vec<_> = (0..n_cols).map(|_| circuit.public_input()).collect();
        (circuit, local, next, pubs, sels)
    }

    /// Compile and run a single base-field expression, returning Ok if the
    /// circuit output matches `expected`.
    fn compile_and_check_base(
        expr: &SymbolicExpression<Challenge>,
        expected: Challenge,
        public_inputs: &[Challenge],
    ) -> Result<(), CircuitError> {
        let n_cols = 2;
        let n_pubs = 1;
        let (mut circuit, local, next, pubs, sels) = setup(n_cols, n_pubs);

        let row_selectors = RowSelectorsTargets {
            is_first_row: sels[0],
            is_last_row: sels[1],
            is_transition: sels[2],
        };
        let columns = ColumnsTargets {
            challenges: &[],
            public_values: &pubs,
            permutation_local_values: &[],
            permutation_next_values: &[],
            permutation_values: &[],
            local_prep_values: &[],
            next_prep_values: &[],
            local_values: &local,
            next_values: &next,
        };

        let compiler = SymbolicCompiler::new(row_selectors, &columns);
        let mut cache = hashbrown::HashMap::new();
        let result = compiler.compile_base(expr, &mut circuit, &mut cache);

        let expected_id = circuit.define_const(expected);
        circuit.connect(result, expected_id);

        let built = circuit.build().unwrap();
        let mut runner = built.runner();
        runner.set_public_inputs(public_inputs).unwrap();
        runner.run()?;
        Ok(())
    }

    #[test]
    fn compile_base_negation() -> Result<(), CircuitError> {
        // -5 should produce the additive inverse.
        let five = Challenge::from_u64(5);
        let expr = -SymbolicExpression::Leaf(BaseLeaf::Constant(five));

        // Public inputs: 3 selectors + 1 pub + 2 local + 2 next = 8
        let zeros = vec![Challenge::ZERO; 8];
        compile_and_check_base(&expr, -five, &zeros)
    }

    #[test]
    fn compile_base_all_ops() -> Result<(), CircuitError> {
        // (a + b) * (a - b) where a=7, b=3 => (10)*(4) = 40
        let a = SymbolicExpression::Leaf(BaseLeaf::Constant(Challenge::from_u64(7)));
        let b = SymbolicExpression::Leaf(BaseLeaf::Constant(Challenge::from_u64(3)));
        let expr = (a.clone() + b.clone()) * (a - b);

        let zeros = vec![Challenge::ZERO; 8];
        compile_and_check_base(&expr, Challenge::from_u64(40), &zeros)
    }

    #[test]
    fn compile_base_selectors() -> Result<(), CircuitError> {
        // is_first_row * is_last_row + is_transition
        let expr = SymbolicExpression::Leaf(BaseLeaf::IsFirstRow)
            * SymbolicExpression::Leaf(BaseLeaf::IsLastRow)
            + SymbolicExpression::Leaf(BaseLeaf::IsTransition);

        let s0 = Challenge::from_u64(3);
        let s1 = Challenge::from_u64(5);
        let s2 = Challenge::from_u64(11);
        // expected: 3*5 + 11 = 26
        // inputs: [s0, s1, s2, pub(0), local(0), local(1), next(0), next(1)]
        let inputs = vec![
            s0,
            s1,
            s2,
            Challenge::ZERO,
            Challenge::ZERO,
            Challenge::ZERO,
            Challenge::ZERO,
            Challenge::ZERO,
        ];
        compile_and_check_base(&expr, s0 * s1 + s2, &inputs)
    }

    #[test]
    fn compile_base_variable_resolution() -> Result<(), CircuitError> {
        // local[0] + next[1]
        let v_local = SymbolicVariable::<Challenge>::new(BaseEntry::Main { offset: 0 }, 0);
        let v_next = SymbolicVariable::<Challenge>::new(BaseEntry::Main { offset: 1 }, 1);
        let expr = SymbolicExpression::Leaf(BaseLeaf::Variable(v_local))
            + SymbolicExpression::Leaf(BaseLeaf::Variable(v_next));

        let a = Challenge::from_u64(13);
        let b = Challenge::from_u64(29);
        // inputs: [s0, s1, s2, pub(0), local(0), local(1), next(0), next(1)]
        let inputs = vec![
            Challenge::ZERO,
            Challenge::ZERO,
            Challenge::ZERO,
            Challenge::ZERO,
            a,
            Challenge::ZERO,
            Challenge::ZERO,
            b,
        ];
        compile_and_check_base(&expr, a + b, &inputs)
    }

    #[test]
    fn compile_base_cache_deduplication() -> Result<(), CircuitError> {
        use alloc::sync::Arc;

        // Build expr = shared * shared where both sides of the Mul
        // point to the *same* Arc allocation, so the DAG deduplicates.
        // Use variables (not constants) to avoid sym_add constant folding.
        //
        //         Mul (root)        ← 1 node
        //        /         \
        //     shared     shared     ← 1 node (same Arc, deduplicated by DAG)
        //      / \
        //  local[0] local[1]        ← 2 nodes (variable leaves)
        //
        // Total DAG nodes: 4.
        let v0 = SymbolicVariable::<Challenge>::new(BaseEntry::Main { offset: 0 }, 0);
        let v1 = SymbolicVariable::<Challenge>::new(BaseEntry::Main { offset: 0 }, 1);
        let shared = Arc::new(SymbolicExpression::from(v0) + SymbolicExpression::from(v1));
        let expr: SymbolicExpression<Challenge> = SymbolicExpression::Mul {
            x: Arc::clone(&shared),
            y: shared,
            degree_multiple: 2,
        };

        let (mut circuit, local, next, pubs, sels) = setup(2, 1);
        let row_selectors = RowSelectorsTargets {
            is_first_row: sels[0],
            is_last_row: sels[1],
            is_transition: sels[2],
        };
        let columns = ColumnsTargets {
            challenges: &[],
            public_values: &pubs,
            permutation_local_values: &[],
            permutation_next_values: &[],
            permutation_values: &[],
            local_prep_values: &[],
            next_prep_values: &[],
            local_values: &local,
            next_values: &next,
        };

        let compiler = SymbolicCompiler::new(row_selectors, &columns);
        let mut cache = hashbrown::HashMap::new();
        let result = compiler.compile_base(&expr, &mut circuit, &mut cache);

        // The DAG should have 4 nodes: v0, v1, Add, Mul.
        // The shared sub-expression is deduplicated by the DAG itself.
        // The external cache has 1 entry (the root expression pointer).
        assert_eq!(cache.len(), 1);

        // Verify correctness: (a + b) * (a + b) with a=3, b=5 → 64.
        let a = Challenge::from_u64(3);
        let b = Challenge::from_u64(5);
        let expected = circuit.define_const((a + b) * (a + b));
        circuit.connect(result, expected);

        let built = circuit.build().unwrap();
        let mut runner = built.runner();
        // inputs: [s0, s1, s2, pub(0), local(0)=a, local(1)=b, next(0), next(1)]
        runner
            .set_public_inputs(&[
                Challenge::ZERO,
                Challenge::ZERO,
                Challenge::ZERO,
                Challenge::ZERO,
                a,
                b,
                Challenge::ZERO,
                Challenge::ZERO,
            ])
            .unwrap();
        runner.run()?;
        Ok(())
    }

    #[test]
    fn compile_ext_constant_and_base_delegation() -> Result<(), CircuitError> {
        // ext_const(42) + base(7) => 42 + 7 = 49
        let base_expr = SymbolicExpression::<F>::Leaf(BaseLeaf::Constant(F::from_u64(7)));
        let ext_const: SymbolicExpressionExt<F, Challenge> =
            SymbolicExpressionExt::Leaf(ExtLeaf::ExtConstant(Challenge::from_u64(42)));
        let ext_base: SymbolicExpressionExt<F, Challenge> =
            SymbolicExpressionExt::Leaf(ExtLeaf::Base(base_expr));
        let expr = ext_const + ext_base;

        let mut circuit = CircuitBuilder::<Challenge>::new();
        let sels = [
            circuit.public_input(),
            circuit.public_input(),
            circuit.public_input(),
        ];

        let row_selectors = RowSelectorsTargets {
            is_first_row: sels[0],
            is_last_row: sels[1],
            is_transition: sels[2],
        };
        let columns = ColumnsTargets {
            challenges: &[],
            public_values: &[],
            permutation_local_values: &[],
            permutation_next_values: &[],
            permutation_values: &[],
            local_prep_values: &[],
            next_prep_values: &[],
            local_values: &[],
            next_values: &[],
        };

        let compiler = SymbolicCompiler::new(row_selectors, &columns);
        let mut base_cache = hashbrown::HashMap::new();
        let mut ext_cache = hashbrown::HashMap::new();
        let result = compiler.compile_ext(&expr, &mut circuit, &mut base_cache, &mut ext_cache);

        let expected = circuit.define_const(Challenge::from_u64(49));
        circuit.connect(result, expected);

        let built = circuit.build().unwrap();
        let mut runner = built.runner();
        runner
            .set_public_inputs(&[Challenge::ZERO, Challenge::ZERO, Challenge::ZERO])
            .unwrap();
        runner.run()?;
        Ok(())
    }

    #[test]
    fn compile_ext_negation() -> Result<(), CircuitError> {
        // -(ext_const(13))
        let expr: SymbolicExpressionExt<F, Challenge> =
            -SymbolicExpressionExt::Leaf(ExtLeaf::ExtConstant(Challenge::from_u64(13)));

        let mut circuit = CircuitBuilder::<Challenge>::new();
        let sels = [
            circuit.public_input(),
            circuit.public_input(),
            circuit.public_input(),
        ];

        let row_selectors = RowSelectorsTargets {
            is_first_row: sels[0],
            is_last_row: sels[1],
            is_transition: sels[2],
        };
        let columns = ColumnsTargets {
            challenges: &[],
            public_values: &[],
            permutation_local_values: &[],
            permutation_next_values: &[],
            permutation_values: &[],
            local_prep_values: &[],
            next_prep_values: &[],
            local_values: &[],
            next_values: &[],
        };

        let compiler = SymbolicCompiler::new(row_selectors, &columns);
        let mut base_cache = hashbrown::HashMap::new();
        let mut ext_cache = hashbrown::HashMap::new();
        let result = compiler.compile_ext(&expr, &mut circuit, &mut base_cache, &mut ext_cache);

        let expected = circuit.define_const(-Challenge::from_u64(13));
        circuit.connect(result, expected);

        let built = circuit.build().unwrap();
        let mut runner = built.runner();
        runner
            .set_public_inputs(&[Challenge::ZERO, Challenge::ZERO, Challenge::ZERO])
            .unwrap();
        runner.run()?;
        Ok(())
    }

    #[test]
    fn compile_ext_variable_resolution() -> Result<(), CircuitError> {
        // challenge[0] * permutation_local[0]
        let ch = SymbolicVariableExt::<F, Challenge>::new(ExtEntry::Challenge, 0);
        let perm = SymbolicVariableExt::<F, Challenge>::new(ExtEntry::Permutation { offset: 0 }, 0);
        let expr: SymbolicExpressionExt<F, Challenge> =
            SymbolicExpressionExt::Leaf(ExtLeaf::ExtVariable(ch))
                * SymbolicExpressionExt::Leaf(ExtLeaf::ExtVariable(perm));

        let mut circuit = CircuitBuilder::<Challenge>::new();
        let sels = [
            circuit.public_input(),
            circuit.public_input(),
            circuit.public_input(),
        ];
        let challenge_target = circuit.public_input();
        let perm_local_target = circuit.public_input();

        let row_selectors = RowSelectorsTargets {
            is_first_row: sels[0],
            is_last_row: sels[1],
            is_transition: sels[2],
        };
        let columns = ColumnsTargets {
            challenges: &[challenge_target],
            public_values: &[],
            permutation_local_values: &[perm_local_target],
            permutation_next_values: &[],
            permutation_values: &[],
            local_prep_values: &[],
            next_prep_values: &[],
            local_values: &[],
            next_values: &[],
        };

        let ch_val = Challenge::from_u64(7);
        let perm_val = Challenge::from_u64(11);

        let compiler = SymbolicCompiler::new(row_selectors, &columns);
        let mut base_cache = hashbrown::HashMap::new();
        let mut ext_cache = hashbrown::HashMap::new();
        let result = compiler.compile_ext(&expr, &mut circuit, &mut base_cache, &mut ext_cache);

        let expected = circuit.define_const(ch_val * perm_val);
        circuit.connect(result, expected);

        let built = circuit.build().unwrap();
        let mut runner = built.runner();
        runner
            .set_public_inputs(&[
                Challenge::ZERO,
                Challenge::ZERO,
                Challenge::ZERO,
                ch_val,
                perm_val,
            ])
            .unwrap();
        runner.run()?;
        Ok(())
    }

    #[test]
    fn compile_base_constant_lifting() -> Result<(), CircuitError> {
        // Test CF != EF: compile a base-field (F) expression into an
        // extension-field (Challenge) circuit with constant lifting.
        let a = F::from_u64(17);
        let b = F::from_u64(5);
        let expr = SymbolicExpression::Leaf(BaseLeaf::Constant(a))
            * SymbolicExpression::Leaf(BaseLeaf::Constant(b));

        let mut circuit = CircuitBuilder::<Challenge>::new();
        let sels = [
            circuit.public_input(),
            circuit.public_input(),
            circuit.public_input(),
        ];

        let row_selectors = RowSelectorsTargets {
            is_first_row: sels[0],
            is_last_row: sels[1],
            is_transition: sels[2],
        };
        let columns = ColumnsTargets {
            challenges: &[],
            public_values: &[],
            permutation_local_values: &[],
            permutation_next_values: &[],
            permutation_values: &[],
            local_prep_values: &[],
            next_prep_values: &[],
            local_values: &[],
            next_values: &[],
        };

        let compiler = SymbolicCompiler::new(row_selectors, &columns);
        let mut cache = hashbrown::HashMap::new();
        // CF=F, EF=Challenge: constants are lifted via Challenge::from(F).
        let result = compiler.compile_base(&expr, &mut circuit, &mut cache);

        let expected = circuit.define_const(Challenge::from_prime_subfield(a * b));
        circuit.connect(result, expected);

        let built = circuit.build().unwrap();
        let mut runner = built.runner();
        runner
            .set_public_inputs(&[Challenge::ZERO, Challenge::ZERO, Challenge::ZERO])
            .unwrap();
        runner.run()?;
        Ok(())
    }
}
