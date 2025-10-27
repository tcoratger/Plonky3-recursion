//! Prelude module for common imports.
//!
//! This module re-exports the most commonly used items from the crate,
//! allowing users to write:
//!
//! ```ignore
//! use p3_recursion::prelude::*;
//! ```
//!
//! Instead of importing each item individually.

// Core type
pub use crate::Target;
// Challenger
pub use crate::challenger::CircuitChallenger;
// Generation
pub use crate::generation::{GenerationError, PcsGeneration, generate_challenges};
// PCS
pub use crate::pcs::fri::{FriVerifierParams, MAX_QUERY_INDEX_BITS};
// Public inputs
pub use crate::public_inputs::{
    CommitmentOpening, FriVerifierInputs, PublicInputBuilder, StarkVerifierInputs,
    StarkVerifierInputsBuilder, construct_stark_verifier_inputs,
};
// Core traits
pub use crate::traits::{
    ComsWithOpeningsTargets, Recursive, RecursiveAir, RecursiveChallenger, RecursiveExtensionMmcs,
    RecursiveMmcs, RecursivePcs,
};
// Key types
pub use crate::types::{
    CommitmentTargets, OpenedValuesTargets, ProofTargets, RecursiveLagrangeSelectors,
    StarkChallenges,
};
// Verifier
pub use crate::verifier::{ObservableCommitment, VerificationError, verify_circuit};
