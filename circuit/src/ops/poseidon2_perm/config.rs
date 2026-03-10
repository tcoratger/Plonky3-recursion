//! Poseidon2 configuration types and execution closures.

use alloc::boxed::Box;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

/// Poseidon2 configuration used as a stable operation key and parameter source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Poseidon2Config {
    /// BabyBear with extension degree D=1 (base field challenges), width 16.
    BabyBearD1Width16,
    BabyBearD4Width16,
    BabyBearD4Width24,
    /// KoalaBear with extension degree D=1 (base field challenges), width 16.
    KoalaBearD1Width16,
    KoalaBearD4Width16,
    KoalaBearD4Width24,
    /// Goldilocks with extension degree D=2, width 8 (matches Poseidon2Goldilocks<8>).
    GoldilocksD2Width8,
}

impl Poseidon2Config {
    pub const fn d(self) -> usize {
        match self {
            Self::BabyBearD1Width16 | Self::KoalaBearD1Width16 => 1,
            Self::GoldilocksD2Width8 => 2,
            Self::BabyBearD4Width16
            | Self::BabyBearD4Width24
            | Self::KoalaBearD4Width16
            | Self::KoalaBearD4Width24 => 4,
        }
    }

    pub const fn width(self) -> usize {
        match self {
            Self::BabyBearD1Width16
            | Self::BabyBearD4Width16
            | Self::KoalaBearD1Width16
            | Self::KoalaBearD4Width16 => 16,
            Self::BabyBearD4Width24 | Self::KoalaBearD4Width24 => 24,
            Self::GoldilocksD2Width8 => 8,
        }
    }

    /// Rate in extension field elements (WIDTH / D for D=4, or WIDTH for D=1).
    pub const fn rate_ext(self) -> usize {
        match self {
            Self::BabyBearD1Width16 | Self::KoalaBearD1Width16 => 8,
            Self::BabyBearD4Width16 | Self::KoalaBearD4Width16 => 2,
            Self::BabyBearD4Width24 | Self::KoalaBearD4Width24 => 4,
            Self::GoldilocksD2Width8 => 2,
        }
    }

    pub const fn rate(self) -> usize {
        self.rate_ext() * self.d()
    }

    /// Capacity in extension field elements.
    pub const fn capacity_ext(self) -> usize {
        match self {
            Self::BabyBearD1Width16 | Self::KoalaBearD1Width16 => 8,
            Self::BabyBearD4Width16
            | Self::BabyBearD4Width24
            | Self::KoalaBearD4Width16
            | Self::KoalaBearD4Width24 => 2,
            Self::GoldilocksD2Width8 => 2,
        }
    }

    pub const fn sbox_degree(self) -> u64 {
        match self {
            Self::BabyBearD1Width16 | Self::BabyBearD4Width16 | Self::BabyBearD4Width24 => 7,
            Self::KoalaBearD1Width16 | Self::KoalaBearD4Width16 | Self::KoalaBearD4Width24 => 3,
            Self::GoldilocksD2Width8 => 7,
        }
    }

    pub const fn sbox_registers(self) -> usize {
        match self {
            Self::BabyBearD1Width16
            | Self::BabyBearD4Width16
            | Self::BabyBearD4Width24
            | Self::GoldilocksD2Width8 => 1,
            Self::KoalaBearD1Width16 | Self::KoalaBearD4Width16 | Self::KoalaBearD4Width24 => 0,
        }
    }

    pub const fn half_full_rounds(self) -> usize {
        match self {
            Self::BabyBearD1Width16
            | Self::BabyBearD4Width16
            | Self::BabyBearD4Width24
            | Self::KoalaBearD1Width16
            | Self::KoalaBearD4Width16
            | Self::KoalaBearD4Width24
            | Self::GoldilocksD2Width8 => 4,
        }
    }

    pub const fn partial_rounds(self) -> usize {
        match self {
            Self::BabyBearD1Width16 | Self::BabyBearD4Width16 => 13,
            Self::BabyBearD4Width24 => 21,
            Self::KoalaBearD1Width16 | Self::KoalaBearD4Width16 => 20,
            Self::KoalaBearD4Width24 => 23,
            Self::GoldilocksD2Width8 => 22,
        }
    }

    pub const fn width_ext(self) -> usize {
        self.rate_ext() + self.capacity_ext()
    }

    /// Stable string name for this config variant, used to build `NpoTypeId`.
    pub const fn variant_name(self) -> &'static str {
        match self {
            Self::BabyBearD1Width16 => "baby_bear_d1_w16",
            Self::BabyBearD4Width16 => "baby_bear_d4_w16",
            Self::BabyBearD4Width24 => "baby_bear_d4_w24",
            Self::KoalaBearD1Width16 => "koala_bear_d1_w16",
            Self::KoalaBearD4Width16 => "koala_bear_d4_w16",
            Self::KoalaBearD4Width24 => "koala_bear_d4_w24",
            Self::GoldilocksD2Width8 => "goldilocks_d2_w8",
        }
    }

    /// Parse a `Poseidon2Config` from a variant name string.
    pub fn from_variant_name(name: &str) -> Option<Self> {
        match name {
            "baby_bear_d1_w16" => Some(Self::BabyBearD1Width16),
            "baby_bear_d4_w16" => Some(Self::BabyBearD4Width16),
            "baby_bear_d4_w24" => Some(Self::BabyBearD4Width24),
            "koala_bear_d1_w16" => Some(Self::KoalaBearD1Width16),
            "koala_bear_d4_w16" => Some(Self::KoalaBearD4Width16),
            "koala_bear_d4_w24" => Some(Self::KoalaBearD4Width24),
            "goldilocks_d2_w8" => Some(Self::GoldilocksD2Width8),
            _ => None,
        }
    }
}

/// Poseidon2 permutation execution closure.
///
/// Takes `width_ext` field elements and returns `width_ext` output elements.
/// For D=1 mode, `width_ext == width` and the elements are base field values.
pub type Poseidon2PermExec<F> = Box<dyn Fn(&[F]) -> Vec<F> + Send + Sync>;

/// Config data stored inside `NpoConfig` for Poseidon2 operations.
///
/// Stored behind `NpoConfig(Arc<dyn Any>)`, so cloning happens at the Arc level.
pub struct Poseidon2PermConfigData<F> {
    pub config: Poseidon2Config,
    pub exec: Poseidon2PermExec<F>,
}
