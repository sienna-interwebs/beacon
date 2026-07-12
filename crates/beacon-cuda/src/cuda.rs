pub use cudarc::cublas::safe::CudaBlas;
pub use cudarc::cublaslt::safe::CudaBlasLT;
pub use cudarc::driver::safe::{CudaContext, CudaSlice, CudaStream};
pub use cudarc::driver::DriverError;
pub use cudarc::nvrtc::Ptx;

use crate::error::LaunchError;

pub fn map_driver_error(err: DriverError) -> LaunchError {
    LaunchError::Cuda(err.to_string())
}

pub fn map_cublas_error(err: cudarc::cublas::result::CublasError) -> LaunchError {
    LaunchError::Cublas(err.to_string())
}

pub fn map_cublaslt_error(err: cudarc::cublaslt::result::CublasError) -> LaunchError {
    LaunchError::Cublas(err.to_string())
}
