#![no_std]
extern crate alloc;
#[cfg(feature = "debugging")]
pub mod alloc_entry;

pub mod builder;
pub mod circuit;
pub mod errors;
pub mod expr;
pub mod ops;
pub mod symbolic;
pub mod tables;
pub mod test_utils;
pub mod types;

// Re-export public API
#[cfg(feature = "debugging")]
pub use alloc_entry::{AllocationEntry, AllocationType, dump_expr_ids};
pub use builder::{
    CircuitBuilder, CircuitBuilderError, NonPrimitiveOperationData, NpoCircuitPlugin,
    NpoLoweringContext,
};
pub use circuit::{Circuit, PreprocessedColumns};
pub use errors::CircuitError;
pub use expr::{Expr, ExpressionGraph};
pub use ops::{AluOpKind, HintExecutor, NonPrimitiveOpPrivateData, NpoPrivateData, NpoTypeId, Op};
pub use tables::{CircuitRunner, Traces};
pub use types::{ExprId, NonPrimitiveOpId, WitnessAllocator, WitnessId};
