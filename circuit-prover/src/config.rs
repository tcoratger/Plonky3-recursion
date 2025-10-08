//! STARK proving configurations.
//!
//! This module provides STARK configurations for different prime fields.
//!
//! # Quick Start
//!
//! ```ignore
//! use p3_circuit_prover::config;
//!
//! // Use a preconfigured setup
//! let config = config::baby_bear()
//!     .build();
//! ```

use p3_baby_bear::{BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_16};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use p3_fri::{TwoAdicFriPcs, create_benchmark_fri_params};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear, default_koalabear_poseidon2_16};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{
    CryptographicPermutation, PaddingFreeSponge, PseudoCompressionFunction, TruncatedPermutation,
};
use p3_uni_stark::StarkConfig;

/// Compression function arity (number of inputs per compression).
const COMPRESS_ARITY: usize = 2;

/// A STARK configuration with all cryptographic primitives specified.
pub type Config<
    F,
    Perm,
    const PERM_WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
    const COMPRESS_CHUNK: usize,
    const CHALLENGE_DEGREE: usize,
> = StarkConfig<
    TwoAdicFriPcs<
        F,
        Radix2DitParallel<F>,
        MerkleTreeMmcs<
            F,
            F,
            PaddingFreeSponge<Perm, PERM_WIDTH, RATE, OUT>,
            TruncatedPermutation<Perm, COMPRESS_ARITY, COMPRESS_CHUNK, PERM_WIDTH>,
            COMPRESS_CHUNK,
        >,
        ExtensionMmcs<
            F,
            BinomialExtensionField<F, CHALLENGE_DEGREE>,
            MerkleTreeMmcs<
                F,
                F,
                PaddingFreeSponge<Perm, PERM_WIDTH, RATE, OUT>,
                TruncatedPermutation<Perm, COMPRESS_ARITY, COMPRESS_CHUNK, PERM_WIDTH>,
                COMPRESS_CHUNK,
            >,
        >,
    >,
    BinomialExtensionField<F, CHALLENGE_DEGREE>,
    DuplexChallenger<F, Perm, PERM_WIDTH, RATE>,
>;

/// Configuration builder for STARK provers.
pub struct ConfigBuilder<
    F,
    Perm,
    const PERM_WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
    const COMPRESS_CHUNK: usize,
    const CHALLENGE_DEGREE: usize,
> {
    perm: Perm,
    _phantom: core::marker::PhantomData<F>,
}

impl<
    F,
    Perm,
    const PERM_WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
    const COMPRESS_CHUNK: usize,
    const CHALLENGE_DEGREE: usize,
> ConfigBuilder<F, Perm, PERM_WIDTH, RATE, OUT, COMPRESS_CHUNK, CHALLENGE_DEGREE>
where
    F: Field,
    Perm: Clone + CryptographicPermutation<[F; PERM_WIDTH]>,
{
    const fn new(perm: Perm) -> Self {
        Self {
            perm,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Builds the final STARK configuration.
    pub fn build(self) -> Config<F, Perm, PERM_WIDTH, RATE, OUT, COMPRESS_CHUNK, CHALLENGE_DEGREE> {
        type Hash<Perm, const PERM_WIDTH: usize, const RATE: usize, const OUT: usize> =
            PaddingFreeSponge<Perm, PERM_WIDTH, RATE, OUT>;
        type Compress<Perm, const PERM_WIDTH: usize, const COMPRESS_CHUNK: usize> =
            TruncatedPermutation<Perm, COMPRESS_ARITY, COMPRESS_CHUNK, PERM_WIDTH>;

        let hash = Hash::<Perm, PERM_WIDTH, RATE, OUT>::new(self.perm.clone());
        let compress = Compress::<Perm, PERM_WIDTH, COMPRESS_CHUNK>::new(self.perm.clone());
        let val_mmcs = MerkleTreeMmcs::new(hash, compress);
        let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());
        let dft = Radix2DitParallel::default();
        let fri_params = create_benchmark_fri_params(challenge_mmcs);
        let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
        let challenger = DuplexChallenger::new(self.perm);

        StarkConfig::new(pcs, challenger)
    }

    /// Creates the compression function for this configuration.
    pub fn compression_function(
        &self,
    ) -> TruncatedPermutation<Perm, COMPRESS_ARITY, COMPRESS_CHUNK, PERM_WIDTH> {
        TruncatedPermutation::new(self.perm.clone())
    }
}

/// Creates a standard BabyBear configuration.
///
/// BabyBear is a 31-bit prime field (2^31 - 2^27 + 1).
///
/// # Parameters
/// - **Permutation width**: 16 (appropriate for 32-bit fields)
/// - **Rate**: 8 (256 bits / 32 bits per element)
/// - **Output size**: 8 (256 bits / 32 bits per element)
/// - **Challenge degree**: 4
///
/// # Examples
///
/// ```ignore
/// let config = config::baby_bear().build();
/// let prover = MultiTableProver::new(config);
/// ```
#[inline]
pub fn baby_bear() -> ConfigBuilder<BabyBear, Poseidon2BabyBear<16>, 16, 8, 8, 8, 4> {
    ConfigBuilder::new(default_babybear_poseidon2_16())
}

/// Creates the standard BabyBear compression function.
#[inline]
pub fn baby_bear_compression() -> impl PseudoCompressionFunction<[BabyBear; 8], 2> {
    baby_bear().compression_function()
}

/// Creates a standard KoalaBear configuration.
///
/// KoalaBear is a 31-bit prime field (2^31 - 2^24 + 1).
///
/// # Parameters
/// - **Permutation width**: 16 (appropriate for 32-bit fields)
/// - **Rate**: 8 (256 bits / 32 bits per element)
/// - **Output size**: 8 (256 bits / 32 bits per element)
/// - **Challenge degree**: 4
///
/// # Examples
///
/// ```ignore
/// let config = config::koala_bear().build();
/// let prover = MultiTableProver::new(config);
/// ```
#[inline]
pub fn koala_bear() -> ConfigBuilder<KoalaBear, Poseidon2KoalaBear<16>, 16, 8, 8, 8, 4> {
    ConfigBuilder::new(default_koalabear_poseidon2_16())
}

/// Creates the standard KoalaBear compression function.
#[inline]
pub fn koala_bear_compression() -> impl PseudoCompressionFunction<[KoalaBear; 8], 2> {
    koala_bear().compression_function()
}

/// Creates a standard Goldilocks configuration.
///
/// Goldilocks is a 64-bit prime field (2^64 - 2^32 + 1).
///
/// # Parameters
/// - **Permutation width**: 8 (appropriate for 64-bit fields)
/// - **Rate**: 4 (256 bits / 64 bits per element)
/// - **Output size**: 4 (256 bits / 64 bits per element)
/// - **Challenge degree**: 2
///
/// # Examples
///
/// ```ignore
/// let config = config::goldilocks().build();
/// let prover = MultiTableProver::new(config);
/// ```
#[inline]
pub fn goldilocks() -> ConfigBuilder<Goldilocks, Poseidon2Goldilocks<8>, 8, 4, 4, 4, 2> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
    let perm = p3_goldilocks::Poseidon2Goldilocks::<8>::new_from_rng_128(&mut rng);
    ConfigBuilder::new(perm)
}

/// Creates the standard Goldilocks compression function.
#[inline]
pub fn goldilocks_compression() -> impl PseudoCompressionFunction<[Goldilocks; 4], 2> {
    goldilocks().compression_function()
}

/// Type alias for BabyBear STARK configuration.
pub type BabyBearConfig = Config<BabyBear, Poseidon2BabyBear<16>, 16, 8, 8, 8, 4>;

/// Type alias for KoalaBear STARK configuration.
pub type KoalaBearConfig = Config<KoalaBear, Poseidon2KoalaBear<16>, 16, 8, 8, 8, 4>;

/// Type alias for Goldilocks STARK configuration.
pub type GoldilocksConfig = Config<Goldilocks, Poseidon2Goldilocks<8>, 8, 4, 4, 4, 2>;

/// Trait bounds for STARK-compatible fields.
pub trait StarkField: Field + PrimeCharacteristicRing + TwoAdicField + PrimeField64 {}

impl<F> StarkField for F where F: Field + PrimeCharacteristicRing + TwoAdicField + PrimeField64 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_fields_configs_compile() {
        let _bb: BabyBearConfig = baby_bear().build();
        let _kb: KoalaBearConfig = koala_bear().build();
        let _gl: GoldilocksConfig = goldilocks().build();
    }

    #[test]
    fn compression_function_works() {
        let _compress = baby_bear_compression();
        let _compress = koala_bear_compression();
        let _compress = goldilocks_compression();
    }
}
