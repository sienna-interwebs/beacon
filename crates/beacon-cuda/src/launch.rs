use crate::device::Stream;
use crate::error::{LaunchError, LaunchResult};

pub const MAX_THREADS_PER_BLOCK: u32 = 1024;
pub const MAX_DYNAMIC_SMEM_BYTES: u32 = 227 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Dim3 { x, y, z }
    }

    pub const fn linear(x: u32) -> Self {
        Dim3 { x, y: 1, z: 1 }
    }

    pub const fn total(&self) -> u64 {
        (self.x as u64) * (self.y as u64) * (self.z as u64)
    }
}

impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Dim3::linear(x)
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from((x, y): (u32, u32)) -> Self {
        Dim3 { x, y, z: 1 }
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Dim3 { x, y, z }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LaunchParams {
    pub grid: Dim3,
    pub block: Dim3,
    pub shared_mem_bytes: u32,
    pub stream: Stream,
}

impl LaunchParams {
    pub fn new(grid: impl Into<Dim3>, block: impl Into<Dim3>) -> Self {
        LaunchParams {
            grid: grid.into(),
            block: block.into(),
            shared_mem_bytes: 0,
            stream: Stream::default_stream(),
        }
    }

    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    pub fn on_stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    pub fn total_threads(&self) -> u64 {
        self.grid.total() * self.block.total()
    }

    pub fn validate(&self) -> LaunchResult<()> {
        if self.grid.total() == 0 || self.block.total() == 0 {
            return Err(LaunchError::InvalidLaunchConfig(
                "grid and block dimensions must be nonzero".into(),
            ));
        }
        let block_threads = self.block.total();
        if block_threads > MAX_THREADS_PER_BLOCK as u64 {
            return Err(LaunchError::InvalidLaunchConfig(format!(
                "block has {block_threads} threads, max is {MAX_THREADS_PER_BLOCK}"
            )));
        }
        if self.shared_mem_bytes > MAX_DYNAMIC_SMEM_BYTES {
            return Err(LaunchError::InvalidLaunchConfig(format!(
                "dynamic shared memory {} bytes exceeds H100 max {MAX_DYNAMIC_SMEM_BYTES}",
                self.shared_mem_bytes
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim3_conversions_and_total() {
        assert_eq!(Dim3::from(8), Dim3::new(8, 1, 1));
        assert_eq!(Dim3::from((4, 4)), Dim3::new(4, 4, 1));
        assert_eq!(Dim3::from((2, 3, 4)), Dim3::new(2, 3, 4));
        assert_eq!(Dim3::new(2, 3, 4).total(), 24);
    }

    #[test]
    fn params_builders_and_defaults() {
        let p = LaunchParams::new(128u32, 256u32);
        assert_eq!(p.shared_mem_bytes, 0);
        assert!(p.stream.is_default());
        assert_eq!(p.total_threads(), 128 * 256);
        let p = p.with_shared_mem(48 * 1024);
        assert_eq!(p.shared_mem_bytes, 48 * 1024);
    }

    #[test]
    fn validate_ok() {
        let p = LaunchParams::new((32u32, 8u32, 1u32), (64u32, 16u32, 1u32))
            .with_shared_mem(100 * 1024);
        assert_eq!(p.block.total(), 1024);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn validate_rejects_too_many_threads() {
        let p = LaunchParams::new(1u32, (32u32, 33u32, 1u32));
        assert!(p.block.total() > MAX_THREADS_PER_BLOCK as u64);
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_rejects_oversized_smem() {
        let p = LaunchParams::new(1u32, 256u32).with_shared_mem(MAX_DYNAMIC_SMEM_BYTES + 1);
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_dims() {
        let p = LaunchParams::new(0u32, 256u32);
        assert!(p.validate().is_err());
    }
}
