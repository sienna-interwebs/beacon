use crate::device::{Device, Stream};
use crate::error::LaunchResult;

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::cuda::{map_cublas_error, map_cublaslt_error, CudaBlas, CudaBlasLT, CudaStream};

pub struct BlasHandles {
    #[cfg(feature = "cuda")]
    cublas: CudaBlas,
    #[cfg(feature = "cuda")]
    cublaslt: CudaBlasLT,
}

impl BlasHandles {
    pub fn new(device: &Device, stream: &Stream) -> LaunchResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let cuda_stream = stream.cuda_stream(device)?;
            let cublas = CudaBlas::new(cuda_stream.clone()).map_err(map_cublas_error)?;
            let cublaslt = CudaBlasLT::new(cuda_stream).map_err(map_cublaslt_error)?;
            Ok(BlasHandles { cublas, cublaslt })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (device, stream);
            Ok(BlasHandles {})
        }
    }

    pub fn bind_stream(&mut self, device: &Device, stream: &Stream) -> LaunchResult<()> {
        #[cfg(feature = "cuda")]
        {
            let cuda_stream = stream.cuda_stream(device)?;
            unsafe {
                self.cublas
                    .set_stream(cuda_stream.clone())
                    .map_err(map_cublas_error)?;
            }
            self.cublaslt = CudaBlasLT::new(cuda_stream).map_err(map_cublaslt_error)?;
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (device, stream);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn cublas(&self) -> &CudaBlas {
        &self.cublas
    }

    #[cfg(feature = "cuda")]
    pub fn cublaslt(&self) -> &CudaBlasLT {
        &self.cublaslt
    }

    #[cfg(feature = "cuda")]
    pub fn bound_stream(&self) -> &Arc<CudaStream> {
        self.cublaslt.stream()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    fn cuda_driver_available() -> bool {
        std::env::var("BEACON_CUDA_TEST").ok().as_deref() == Some("1")
    }

    fn device_or_skip() -> Option<Device> {
        #[cfg(feature = "cuda")]
        if !cuda_driver_available() {
            return None;
        }
        Device::new(0).ok()
    }

    #[test]
    fn host_stub_constructs() {
        let Some(device) = device_or_skip() else {
            return;
        };
        let stream = device.default_stream();
        let mut handles = BlasHandles::new(&device, &stream).unwrap();
        handles.bind_stream(&device, &stream).unwrap();
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn handles_bind_to_default_stream() {
        let Some(device) = device_or_skip() else {
            return;
        };
        let stream = device.default_stream();
        let handles = BlasHandles::new(&device, &stream).unwrap();
        assert!(Arc::ptr_eq(
            handles.bound_stream(),
            device.context().default_stream()
        ));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn bind_stream_follows_owned_stream() {
        let Some(device) = device_or_skip() else {
            return;
        };
        let owned = device.new_stream().unwrap();
        let mut handles = BlasHandles::new(&device, &device.default_stream()).unwrap();
        handles.bind_stream(&device, &owned).unwrap();
        let cuda = owned.cuda_stream(&device).unwrap();
        assert!(Arc::ptr_eq(handles.bound_stream(), &cuda));
    }
}
