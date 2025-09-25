use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

/// Scaffolding for future implementation of bit extraction AIR table
///
/// This AIR table will implement the circuit version of `challenger.sample_bits(n)`
/// operation used in FRI verification. The operation extracts the lowest `bits` bits
/// from a field element, equivalent to `input & ((1 << bits) - 1)`.
///
/// ## Usage in Recursive Circuit:
///
/// During FRI verification, the challenger samples random field elements and needs
/// to extract specific bit ranges for query indices. Each `sample_bits(n)` call
/// will add one row to this AIR table, with the circuit builder managing the
/// witness assignments and constraint generation.
#[derive(Debug, Clone)]
pub struct SampleBitsAir<F, const BIT_WIDTH: usize> {
    /// Expected number of sample_bits operations in the circuit
    ///
    /// This determines the trace height for this AIR table
    pub height: usize,
    _phantom: core::marker::PhantomData<F>,
}

impl<F: Field, const BIT_WIDTH: usize> SampleBitsAir<F, BIT_WIDTH> {
    /// Create a new SampleBitsAir instance
    ///
    /// # Arguments
    /// * `height` - Number of sample_bits operations expected in the circuit
    pub const fn new(height: usize) -> Self {
        Self {
            height,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Convert sample_bits operations to trace matrix
    ///
    /// # Future Implementation Notes:
    /// - Each operation provides: input, output, num_bits
    /// - Prover must compute: full bit decomposition of input
    /// - Trace construction: pack all data into row-major format
    /// - Padding: extend to power-of-two height with repeated last row
    pub fn operations_to_matrix(
        _operations: &[SampleBitsOperation<F, BIT_WIDTH>],
    ) -> RowMajorMatrix<F> {
        unimplemented!("SampleBitsAir::operations_to_matrix - convert operations to trace matrix")
    }
}

impl<F: Field, const BIT_WIDTH: usize> BaseAir<F> for SampleBitsAir<F, BIT_WIDTH> {
    /// Number of columns in the trace matrix
    ///
    /// Layout: [input, output, num_bits, bit_0, bit_1, ..., bit_{BIT_WIDTH-1}, index]
    /// Total: 1 + 1 + 1 + BIT_WIDTH + 1 = BIT_WIDTH + 4 columns
    fn width(&self) -> usize {
        BIT_WIDTH + 4
    }
}

impl<AB: AirBuilder, const BIT_WIDTH: usize> Air<AB> for SampleBitsAir<AB::F, BIT_WIDTH>
where
    AB::F: Field,
{
    /// Implement the constraints for bit extraction verification
    ///
    /// This will enforce:
    /// 1. Bit decomposition correctness
    /// 2. Binary range constraints (for each of the BIT_WIDTH bits)
    /// 3. Output extraction correctness
    /// 4. Transparent index monotonicity
    fn eval(&self, _builder: &mut AB) {
        unimplemented!("SampleBitsAir::eval - implement bit extraction constraints")
    }
}

/// Represents a single sample_bits operation for trace generation
///
/// This struct will hold all the data needed to generate one row of the
/// SampleBitsAir trace matrix, including both public interface values
/// and private witness data.
#[derive(Debug, Clone)]
pub struct SampleBitsOperation<F, const BIT_WIDTH: usize> {
    /// Input field element from challenger sampling
    pub input: F,

    /// Output: extracted bits as field element (range [0, 2^num_bits - 1])
    pub output: F,

    /// Number of bits to extract (1 ≤ num_bits ≤ BIT_WIDTH)
    pub num_bits: usize,

    /// Private witness: bit decomposition of input
    /// Length should be BIT_WIDTH (field element bit width)
    /// Each element must be 0 or 1
    pub bit_decomposition: Vec<F>,
}
