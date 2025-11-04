use core::marker::PhantomData;

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, CanSample, DuplexChallenger, FieldChallenger};
use p3_circuit::CircuitBuilder;
use p3_circuit::utils::RowSelectorsTargets;
use p3_commit::testing::TrivialPcs;
use p3_commit::{Pcs, PolynomialSpace};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_recursion::challenger::CircuitChallenger;
use p3_recursion::traits::{ComsWithOpeningsTargets, Recursive, RecursivePcs};
use p3_recursion::types::{OpenedValuesTargets, ProofTargets};
use p3_recursion::verifier::{ObservableCommitment, VerificationError, verify_circuit};
use p3_recursion::{RecursiveLagrangeSelectors, Target};
use p3_uni_stark::{Domain, StarkConfig, StarkGenericConfig, Val, prove};
use p3_util::log2_strict_usize;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Default rate for the `CircuitChallenger`
const DEFAULT_CHALLENGER_RATE: usize = 8;

/// Number of repetitions of the multiplication constraint (must be < 255 to fit in u8)
const REPETITIONS: usize = 20;

/// Total trace width: 3 columns per repetition (a, b, c)
const TRACE_WIDTH: usize = REPETITIONS * 3;

/// Dummy commitment type for `TrivialPcs` (stores polynomial coefficients directly)
#[derive(Clone)]
#[allow(dead_code)]
struct DummyCom<F>(Vec<Vec<F>>);

impl<F: Field> ObservableCommitment for DummyCom<F> {
    fn to_observation_targets(&self) -> Vec<Target> {
        // For dummy/trivial commitments, return empty targets
        // The commitment is the polynomial itself, so no hash to observe
        vec![]
    }
}

impl<F: Field, EF: ExtensionField<F>> Recursive<EF> for DummyCom<F> {
    type Input = Vec<Vec<F>>;

    fn new(_circuit: &mut CircuitBuilder<EF>, _input: &Self::Input) -> Self {
        // For TrivialPcs, we don't need to allocate targets for commitments
        DummyCom(vec![])
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        // Convert all polynomial coefficients to extension field elements
        input.iter().flatten().map(|v| EF::from(*v)).collect()
    }
}

/// Empty target type for TrivialPcs opening proof (unit type)
#[derive(Clone, Copy)]
struct EmptyTarget;

impl<F: Field> Recursive<F> for EmptyTarget {
    type Input = ();

    fn new(_circuit: &mut CircuitBuilder<F>, _input: &Self::Input) -> Self {
        EmptyTarget
    }

    fn get_values(_input: &Self::Input) -> Vec<F> {
        vec![]
    }
}

/// Wrapper around `TrivialPcs` to satisfy orphan rule
struct TestPcs<Val: TwoAdicField, Dft: TwoAdicSubgroupDft<Val>>(TrivialPcs<Val, Dft>);

impl<Val: TwoAdicField, Challenge: ExtensionField<Val>, Challenger, Dft: TwoAdicSubgroupDft<Val>>
    Pcs<Challenge, Challenger> for TestPcs<Val, Dft>
where
    Challenger: CanSample<Challenge>,
    Vec<Vec<Val>>: Serialize + for<'de> Deserialize<'de>,
{
    type Domain = <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::Domain;
    type Commitment = <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::Commitment;
    type ProverData = <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::ProverData;
    type EvaluationsOnDomain<'a> =
        <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::EvaluationsOnDomain<'a>;
    type Proof = <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::Proof;
    type Error = <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::Error;
    const ZK: bool = <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::ZK;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &self.0, degree,
        )
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::commit(&self.0, evaluations)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::get_evaluations_on_domain(
            &self.0,
            prover_data,
            idx,
            domain,
        )
    }

    fn open(
        &self,
        rounds: Vec<(&Self::ProverData, Vec<Vec<Challenge>>)>,
        challenger: &mut Challenger,
    ) -> (p3_commit::OpenedValues<Challenge>, Self::Proof) {
        <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::open(&self.0, rounds, challenger)
    }

    fn verify(
        &self,
        rounds: Vec<(
            Self::Commitment,
            Vec<(Self::Domain, Vec<(Challenge, Vec<Challenge>)>)>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        <TrivialPcs<Val, Dft> as Pcs<Challenge, Challenger>>::verify(
            &self.0, rounds, proof, challenger,
        )
    }
}

impl<SC: StarkGenericConfig, Comm: Recursive<SC::Challenge>, Dft>
    RecursivePcs<SC, EmptyTarget, EmptyTarget, Comm, TwoAdicMultiplicativeCoset<Val<SC>>>
    for TestPcs<Val<SC>, Dft>
where
    Domain<SC>: PolynomialSpace,
    Val<SC>: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val<SC>>,
{
    type VerifierParams = ();
    type RecursiveProof = EmptyTarget;

    fn get_challenges_circuit<const RATE: usize>(
        _circuit: &mut CircuitBuilder<<SC as StarkGenericConfig>::Challenge>,
        _challenger: &mut CircuitChallenger<RATE>,
        _proof_targets: &ProofTargets<SC, Comm, EmptyTarget>,
        _opened_values: &OpenedValuesTargets<SC>,
        _params: &Self::VerifierParams,
    ) -> Vec<Target> {
        // TrivialPcs has no additional challenges beyond the base STARK challenges
        vec![]
    }

    fn verify_circuit(
        &self,
        _circuit: &mut CircuitBuilder<<SC as StarkGenericConfig>::Challenge>,
        _challenges: &[Target],
        _commitments_with_opening_points: &ComsWithOpeningsTargets<
            Comm,
            TwoAdicMultiplicativeCoset<Val<SC>>,
        >,
        _opening_proof: &EmptyTarget,
        _params: &Self::VerifierParams,
    ) -> Result<(), VerificationError> {
        // TrivialPcs doesn't verify anything - it's for testing only
        Ok(())
    }

    fn selectors_at_point_circuit(
        &self,
        circuit: &mut CircuitBuilder<SC::Challenge>,
        domain: &TwoAdicMultiplicativeCoset<Val<SC>>,
        point: &Target,
    ) -> RecursiveLagrangeSelectors {
        // Constants that we will need
        let shift_inv = circuit.add_const(SC::Challenge::from(domain.shift_inverse()));
        let one = circuit.add_const(SC::Challenge::from(Val::<SC>::ONE));
        let subgroup_gen_inv =
            circuit.add_const(SC::Challenge::from(domain.subgroup_generator().inverse()));

        // Compute unshifted point and vanishing polynomial Z_H(point) = point^n - 1
        let unshifted_point = circuit.mul(shift_inv, *point);
        let us_exp = circuit.exp_power_of_2(unshifted_point, domain.log_size());
        let z_h = circuit.sub(us_exp, one);

        // Compute denominators for Lagrange basis evaluation
        let us_minus_one = circuit.sub(unshifted_point, one);
        let us_minus_gen_inv = circuit.sub(unshifted_point, subgroup_gen_inv);

        // Compute row selectors:
        // - is_first_row: L_0(point) = Z_H(point) / (point - 1)
        // - is_last_row: L_{n-1}(point) = Z_H(point) / (point - g^{-1})
        // - is_transition: (point - g^{-1}) - used for transition constraints
        let is_first_row = circuit.div(z_h, us_minus_one);
        let is_last_row = circuit.div(z_h, us_minus_gen_inv);
        let is_transition = us_minus_gen_inv;
        let inv_vanishing = circuit.div(one, z_h);

        let row_selectors = RowSelectorsTargets {
            is_first_row,
            is_last_row,
            is_transition,
        };

        RecursiveLagrangeSelectors {
            row_selectors,
            inv_vanishing,
        }
    }

    fn create_disjoint_domain(
        &self,
        trace_domain: TwoAdicMultiplicativeCoset<Val<SC>>,
        degree: usize,
    ) -> TwoAdicMultiplicativeCoset<Val<SC>> {
        trace_domain.create_disjoint_domain(degree)
    }

    fn split_domains(
        &self,
        trace_domain: &TwoAdicMultiplicativeCoset<Val<SC>>,
        degree: usize,
    ) -> Vec<TwoAdicMultiplicativeCoset<Val<SC>>> {
        trace_domain.split_domains(degree)
    }

    fn size(&self, trace_domain: &TwoAdicMultiplicativeCoset<Val<SC>>) -> usize {
        trace_domain.size()
    }

    fn log_size(&self, trace_domain: &TwoAdicMultiplicativeCoset<Val<SC>>) -> usize {
        trace_domain.log_size()
    }

    fn first_point(&self, trace_domain: &TwoAdicMultiplicativeCoset<Val<SC>>) -> SC::Challenge {
        SC::Challenge::from(trace_domain.first_point())
    }
}

/// A test AIR that enforces multiplication constraints: a^(degree-1) * b = c
///
/// # Constraints
/// For each of REPETITIONS triples (a, b, c):
/// 1. Multiplication: a^(degree-1) * b = c
/// 2. First row: a^2 + 1 = b
/// 3. Transition: a' = a + REPETITIONS (where a' is next row's a)
///
/// # Trace Layout
/// The trace has TRACE_WIDTH = REPETITIONS * 3 columns:
/// [a_0, b_0, c_0, a_1, b_1, c_1, ..., a_19, b_19, c_19]
pub struct MulAir {
    /// Degree of the polynomial constraint (a^(degree-1) * b = c)
    degree: u64,
}

impl Default for MulAir {
    fn default() -> Self {
        Self { degree: 3 }
    }
}

impl MulAir {
    /// Generate a random valid (or invalid) trace for testing
    ///
    /// # Parameters
    /// - `rows`: Number of rows in the trace
    /// - `valid`: If true, generates a valid trace; if false, makes it invalid
    pub fn random_valid_trace<F: Field>(&self, rows: usize, valid: bool) -> RowMajorMatrix<F>
    where
        StandardUniform: Distribution<F>,
    {
        let mut rng = SmallRng::seed_from_u64(1);
        let mut trace_values = F::zero_vec(rows * TRACE_WIDTH);

        for (i, (a, b, c)) in trace_values.iter_mut().tuples().enumerate() {
            let row = i / REPETITIONS;
            *a = F::from_usize(i);

            // First row: b = a^2 + 1
            // Other rows: random b
            *b = if row == 0 {
                a.square() + F::ONE
            } else {
                rng.random()
            };

            // Compute c = a^(degree-1) * b
            *c = a.exp_u64(self.degree - 1) * *b;

            if !valid {
                // Make the trace invalid by corrupting c
                *c *= F::TWO;
            }
        }

        RowMajorMatrix::new(trace_values, TRACE_WIDTH)
    }
}

impl<F> BaseAir<F> for MulAir {
    fn width(&self) -> usize {
        TRACE_WIDTH
    }
}

impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.row_slice(0).expect("Matrix is empty?");
        let main_next = main.row_slice(1).expect("Matrix only has 1 row?");

        for i in 0..REPETITIONS {
            let start = i * 3;
            let a = main_local[start].clone();
            let b = main_local[start + 1].clone();
            let c = main_local[start + 2].clone();

            // Constraint 1: a^(degree-1) * b = c
            builder.assert_zero(a.clone().into().exp_u64(self.degree - 1) * b.clone() - c);

            // Constraint 2: On first row, b = a^2 + 1
            builder
                .when_first_row()
                .assert_eq(a.clone() * a.clone() + AB::Expr::ONE, b);

            // Constraint 3: On transition rows, a' = a + REPETITIONS
            let next_a = main_next[start].clone();
            builder
                .when_transition()
                .assert_eq(a + AB::Expr::from_u8(REPETITIONS as u8), next_a);
        }
    }
}

/// Generate challenges for STARK verification without PCS challenges
///
/// This mimics the Fiat-Shamir transform used during proving:
/// 1. Observe instance data (degree)
/// 2. Observe trace commitment
/// 3. Sample alpha
/// 4. Observe quotient chunks commitment
/// 5. Observe random commitment (if ZK)
/// 6. Sample zeta and compute zeta_next
///
/// # Returns
/// `Vec<Challenge>` containing `[alpha, zeta, zeta_next]`
fn get_challenges<SC: StarkGenericConfig>(
    config: &SC,
    degree: usize,
    trace_commit: &<SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
    quotient_chunks: &<SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
    random: &Option<<SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment>,
) -> Vec<SC::Challenge> {
    let log_degree = log2_strict_usize(degree);
    let log_ext_degree = log_degree + config.is_zk();

    let pcs = config.pcs();
    let init_trace_domain = pcs.natural_domain_for_degree(degree >> (config.is_zk()));

    // Initialize the challenger
    let mut challenger = config.initialise_challenger();

    // Observe the instance (degree information)
    challenger.observe(Val::<SC>::from_u8(log_ext_degree as u8));
    challenger.observe(Val::<SC>::from_u8(log_degree as u8));

    // Observe the trace commitment
    challenger.observe(trace_commit.clone());

    // Sample alpha for constraint folding
    let alpha: SC::Challenge = challenger.sample_algebra_element();

    // Observe quotient chunks commitment
    challenger.observe(quotient_chunks.clone());

    // Observe random commitment if ZK is enabled
    if let Some(r_commit) = random.clone() {
        challenger.observe(r_commit);
    }

    // Sample out-of-domain evaluation point zeta
    let zeta = challenger.sample_algebra_element();
    let zeta_next = init_trace_domain.next_point(zeta).unwrap();

    vec![alpha, zeta, zeta_next]
}

#[test]
fn test_mul_verifier_circuit() -> Result<(), String> {
    // Configuration
    let log_n = 8;
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Dft = Radix2DitParallel<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Pcs = TestPcs<Val, Dft>;
    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

    // Initialize RNG and permutation
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    // Create PCS and config
    let dft = Dft::default();
    let pcs = TestPcs(TrivialPcs {
        dft,
        log_n,
        _phantom: PhantomData,
    });
    let challenger = Challenger::new(perm);
    let config = MyConfig::new(pcs, challenger);

    // Create AIR and generate valid trace
    let air = MulAir { degree: 2 };
    let trace = air.random_valid_trace(1 << log_n, true);

    // Generate proof
    let mut proof = prove(&config, &air, trace, &vec![]);

    // Generate challenges
    let challenges = get_challenges(
        &config,
        1 << log_n,
        &proof.commitments.trace,
        &proof.commitments.quotient_chunks,
        &proof.commitments.random,
    );

    // Clear commitments (TrivialPcs stores them inline)
    proof.commitments.random = None;
    proof.commitments.quotient_chunks = vec![];
    proof.commitments.trace = vec![];

    // Build verification circuit
    let mut circuit_builder = CircuitBuilder::new();
    let proof_targets =
        ProofTargets::<MyConfig, DummyCom<Val>, EmptyTarget>::new(&mut circuit_builder, &proof);

    let proof_values = ProofTargets::<MyConfig, DummyCom<Val>, EmptyTarget>::get_values(&proof);

    // Combine proof values and challenges as public inputs
    let pvs = proof_values
        .iter()
        .chain(&challenges)
        .copied()
        .collect::<Vec<_>>();

    // Add verification constraints to circuit
    verify_circuit::<_, _, _, _, _, DEFAULT_CHALLENGER_RATE>(
        &config,
        &air,
        &mut circuit_builder,
        &proof_targets,
        &[],
        &(),
    )
    .map_err(|e| format!("{e:?}"))?;

    // Build and run circuit
    let circuit = circuit_builder.build().unwrap();
    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&pvs)
        .map_err(|e| format!("{e:?}"))?;

    let _traces = runner.run().map_err(|e| format!("{e:?}"))?;

    Ok(())
}
