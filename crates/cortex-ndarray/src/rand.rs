//! Random number generation utilities for cortex-ndarray

#[cfg(not(feature = "std"))]
use rand::rngs::SmallRng;
#[cfg(feature = "std")]
use rand::rngs::StdRng;

/// Type alias for the RNG used by cortex-ndarray
#[cfg(feature = "std")]
pub type NdArrayRng = StdRng;
#[cfg(not(feature = "std"))]
pub type NdArrayRng = SmallRng;

#[cfg(not(feature = "std"))]
use rand::SeedableRng;

/// Get a seeded random number generator
///
/// For std builds, uses OS entropy.
/// For no_std builds, uses a compile-time random seed.
#[cfg(feature = "std")]
pub fn get_seeded_rng() -> NdArrayRng {
    // Use the standard implementation from cortex-std
    cortex_std::rand::get_seeded_rng()
}

/// Get a seeded random number generator
///
/// For std builds, uses OS entropy.
/// For no_std builds, uses a compile-time random seed.
#[cfg(not(feature = "std"))]
pub fn get_seeded_rng() -> NdArrayRng {
    // Use compile-time random seed for no_std
    const SEED: u64 = const_random::const_random!(u64);
    SmallRng::seed_from_u64(SEED)
}
