#![allow(dead_code)]

pub mod device;
pub mod error;
pub mod launch;
pub mod launcher;

pub use device::{Device, DeviceId, Stream, StreamHandle};
pub use error::{LaunchError, LaunchResult};
pub use launch::{Dim3, LaunchParams, MAX_DYNAMIC_SMEM_BYTES, MAX_THREADS_PER_BLOCK};
pub use launcher::{Access, KernelArg, KernelId, KernelLauncher};
