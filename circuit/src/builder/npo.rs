use alloc::boxed::Box;
use alloc::vec::Vec;
use core::marker::PhantomData;

use hashbrown::HashMap;
use p3_field::Field;

use crate::CircuitBuilderError;
use crate::op::{HintExecutor, NpoConfig, NpoTypeId, Op};
use crate::tables::TraceGeneratorFn;
use crate::types::{ExprId, NonPrimitiveOpId, WitnessId};

/// Per-op extra parameters that are not encoded in the op type.
#[derive(Debug)]
pub enum NonPrimitiveOpParams<F> {
    Poseidon2Perm { new_start: bool, merkle_path: bool },
    Unconstrained { executor: Box<dyn HintExecutor<F>> },
    Recompose,
}

impl<F: Field> Clone for NonPrimitiveOpParams<F> {
    fn clone(&self) -> Self {
        match self {
            Self::Poseidon2Perm {
                new_start,
                merkle_path,
            } => Self::Poseidon2Perm {
                new_start: *new_start,
                merkle_path: *merkle_path,
            },
            Self::Unconstrained { executor } => Self::Unconstrained {
                executor: executor.boxed(),
            },
            Self::Recompose => Self::Recompose,
        }
    }
}

/// The non-primitive operation id, type, the vectors of the expressions representing its inputs
/// and outputs, and any per-op parameters.
#[derive(Debug, Clone)]
pub struct NonPrimitiveOperationData<F: Field> {
    pub op_id: NonPrimitiveOpId,
    pub op_type: NpoTypeId,
    /// Input expressions (e.g., for Poseidon2Perm: [in0, in1, in2, in3, mmcs_index_sum, mmcs_bit])
    pub input_exprs: Vec<Vec<ExprId>>,
    /// Output expressions (e.g., for Poseidon2Perm: [out0, out1])
    pub output_exprs: Vec<Vec<ExprId>>,
    pub params: Option<NonPrimitiveOpParams<F>>,
}

/// Lowering context passed to `NpoCircuitPlugin::lower`, providing access to the
/// expression-to-witness map and witness allocation function.
pub struct NpoLoweringContext<'a, F> {
    pub expr_to_widx: &'a mut HashMap<ExprId, WitnessId>,
    pub alloc_witness_id: &'a mut dyn FnMut(usize) -> WitnessId,
    /// Phantom to keep `F` in the type, even though we only carry witness IDs here.
    _phantom: PhantomData<F>,
}

impl<'a, F> NpoLoweringContext<'a, F> {
    pub fn new(
        expr_to_widx: &'a mut HashMap<ExprId, WitnessId>,
        alloc_witness_id: &'a mut dyn FnMut(usize) -> WitnessId,
    ) -> Self {
        Self {
            expr_to_widx,
            alloc_witness_id,
            _phantom: PhantomData,
        }
    }
}

/// Circuit-layer plugin interface for non-primitive operations.
///
/// Implementors are responsible for:
/// - Lowering their high-level operation description into a single `Op<F>`
/// - Providing a trace generator for their dedicated table
/// - Exposing their configuration as an `NpoConfig`
pub trait NpoCircuitPlugin<F: Field>: Send + Sync {
    /// Unique type identifier for this NPO (e.g. "poseidon2_perm/baby_bear_d4_w16").
    fn type_id(&self) -> NpoTypeId;

    /// Convert a high-level NPO operation into a concrete `Op<F>`.
    ///
    /// The lowering context gives access to the expression→witness mapping and
    /// witness allocation for any new outputs.
    fn lower(
        &self,
        data: &NonPrimitiveOperationData<F>,
        output_exprs: &[(u32, ExprId)],
        ctx: &mut NpoLoweringContext<'_, F>,
    ) -> Result<Op<F>, CircuitBuilderError>;

    /// Produce the trace generator for this NPO.
    fn trace_generator(&self) -> TraceGeneratorFn<F>;

    /// Return plugin-specific configuration for this NPO.
    fn config(&self) -> NpoConfig;
}
