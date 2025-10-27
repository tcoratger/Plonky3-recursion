#![no_std]
//! Recursive proof verification for Plonky3 STARKs.
//!
//! This crate provides the infrastructure for verifying STARK proofs within circuits,
//! enabling recursive proof composition. The verification process is split into two phases:
//!
//! 1. **Circuit Building**: Allocate targets and add verification constraints
//! 2. **Execution**: Populate public inputs and execute the circuit
//!
//! # Architecture
//!
//! The crate is organized into the following modules:
//!
//! - [`traits`]: Core traits (`Recursive`, `RecursiveChallenger`, `RecursiveAir`, `RecursivePcs`)
//! - [`types`]: Core types (`ProofTargets`, `StarkChallenges`, `Target`)
//! - [`challenger`]: Fiat-Shamir challenger implementations
//! - [`verifier`]: STARK verification logic
//! - [`pcs`]: Polynomial commitment scheme implementations (FRI, etc.)
//! - [`generation`]: Challenge and proof generation utilities
//! - [`public_inputs`]: Public input builders and helpers
//! - [`prelude`]: Commonly used imports
//!
//! # Example
//!
//! ```ignore
//! use p3_recursion::prelude::*;
//!
//! // Phase 1: Build the verification circuit
//! let mut circuit = CircuitBuilder::new();
//! let proof_targets = ProofTargets::new(&mut circuit, &proof);
//! let public_targets = circuit.alloc_public_inputs(num_public, "AIR public values");
//!
//! verify_circuit::<_, _, _, _, _, 8>(
//!     &config,
//!     &air,
//!     &mut circuit,
//!     &proof_targets,
//!     &public_targets,
//!     &fri_params,
//! )?;
//!
//! // Phase 2: Execute with actual values
//! let challenges = generate_challenges(&air, &config, &proof, &public_values, None)?;
//! let public_inputs = construct_stark_verifier_inputs(
//!     &public_values,
//!     &ProofTargets::get_values(&proof),
//!     &challenges,
//!     num_queries,
//! );
//!
//! circuit.set_public_inputs(&public_inputs)?;
//! circuit.execute()?;
//! ```

extern crate alloc;

// ================================
// Module declarations
// ================================

pub mod challenger;
pub mod generation;
pub mod pcs;
pub mod prelude;
pub mod public_inputs;
pub mod traits;
pub mod types;
pub mod verifier;

// ================================
// Public API re-exports
// ================================

// Core type
// Challenger
pub use challenger::CircuitChallenger;
// Generation
pub use generation::{GenerationError, PcsGeneration, generate_challenges};
// PCS
pub use pcs::fri::{FriVerifierParams, MAX_QUERY_INDEX_BITS};
// Public inputs
pub use public_inputs::{
    CommitmentOpening, FriVerifierInputs, PublicInputBuilder, StarkVerifierInputs,
    StarkVerifierInputsBuilder, construct_stark_verifier_inputs,
};
// Core traits
pub use traits::{
    Recursive, RecursiveAir, RecursiveChallenger, RecursiveExtensionMmcs, RecursiveMmcs,
    RecursivePcs,
};
pub use types::Target;
// Key types
pub use types::{
    CommitmentTargets, OpenedValuesTargets, ProofTargets, RecursiveLagrangeSelectors,
    StarkChallenges,
};
// Verifier
pub use verifier::{ObservableCommitment, VerificationError, verify_circuit};
