#[cfg(feature = "cuda")]
use crate::error::LaunchError;
use crate::error::LaunchResult;

#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, CudaStream, map_driver_error};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamHandle(pub u64);

impl StreamHandle {
    pub const DEFAULT: StreamHandle = StreamHandle(0);
}

#[derive(Debug, Clone)]
pub struct Stream {
    #[cfg(feature = "cuda")]
    inner: StreamInner,
    #[cfg(not(feature = "cuda"))]
    handle: StreamHandle,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
enum StreamInner {
    Default,
    Owned(Arc<CudaStream>),
}

impl Stream {
    pub fn default_stream() -> Self {
        #[cfg(feature = "cuda")]
        {
            Stream {
                inner: StreamInner::Default,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Stream {
                handle: StreamHandle::DEFAULT,
            }
        }
    }

    pub fn from_handle(handle: StreamHandle) -> Self {
        #[cfg(feature = "cuda")]
        {
            if handle == StreamHandle::DEFAULT {
                Stream::default_stream()
            } else {
                Stream {
                    inner: StreamInner::Default,
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Stream { handle }
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda(stream: Arc<CudaStream>) -> Self {
        Stream {
            inner: StreamInner::Owned(stream),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_stream(&self, device: &Device) -> LaunchResult<Arc<CudaStream>> {
        match &self.inner {
            StreamInner::Default => Ok(device.context()?.default_stream().clone()),
            StreamInner::Owned(stream) => Ok(stream.clone()),
        }
    }

    pub fn handle(&self) -> StreamHandle {
        #[cfg(feature = "cuda")]
        {
            match &self.inner {
                StreamInner::Default => StreamHandle::DEFAULT,
                StreamInner::Owned(stream) => StreamHandle(Arc::as_ptr(stream) as u64),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.handle
        }
    }

    pub fn is_default(&self) -> bool {
        self.handle() == StreamHandle::DEFAULT
    }
}

impl Default for Stream {
    fn default() -> Self {
        Self::default_stream()
    }
}

#[derive(Debug)]
pub struct Device {
    id: DeviceId,
    #[cfg(feature = "cuda")]
    ctx: OnceLock<Result<Arc<CudaContext>, LaunchError>>,
}

impl Device {
    pub fn new(ordinal: usize) -> LaunchResult<Self> {
        Ok(Device {
            id: DeviceId(ordinal),
            #[cfg(feature = "cuda")]
            ctx: OnceLock::new(),
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    #[cfg(feature = "cuda")]
    pub fn context(&self) -> LaunchResult<&Arc<CudaContext>> {
        match self
            .ctx
            .get_or_init(|| CudaContext::new(self.id.0).map_err(map_driver_error).map(Arc::from))
        {
            Ok(ctx) => Ok(ctx),
            Err(err) => Err(err.clone()),
        }
    }

    pub fn default_stream(&self) -> Stream {
        Stream::default_stream()
    }

    pub fn new_stream(&self) -> LaunchResult<Stream> {
        #[cfg(feature = "cuda")]
        {
            let stream = self.context()?.new_stream().map_err(map_driver_error)?;
            Ok(Stream::from_cuda(stream))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(Stream::default_stream())
        }
    }

    pub fn synchronize(&self) -> LaunchResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.context()?.synchronize().map_err(map_driver_error)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_driver_available() -> bool {
        std::env::var("BEACON_CUDA_TEST").ok().as_deref() == Some("1")
    }

    fn device_or_skip() -> Option<Device> {
        let d = Device::new(0).ok()?;
        #[cfg(feature = "cuda")]
        if !cuda_driver_available() {
            return Some(d);
        }
        #[cfg(feature = "cuda")]
        if d.context().is_err() {
            return None;
        }
        Some(d)
    }

    #[test]
    fn default_stream_is_zero_handle() {
        let s = Stream::default_stream();
        assert!(s.is_default());
        assert_eq!(s.handle(), StreamHandle::DEFAULT);
        assert_eq!(s.handle(), StreamHandle(0));
    }

    #[test]
    fn non_default_stream() {
        let s = Stream::from_handle(StreamHandle(5));
        #[cfg(not(feature = "cuda"))]
        {
            assert!(!s.is_default());
            assert_eq!(s.handle(), StreamHandle(5));
        }
        #[cfg(feature = "cuda")]
        {
            let _ = s;
        }
    }

    #[test]
    fn device_construct_and_sync() {
        let Some(d) = device_or_skip() else {
            return;
        };
        assert_eq!(d.id(), DeviceId(0));
        assert!(d.default_stream().is_default());
        #[cfg(feature = "cuda")]
        if !cuda_driver_available() {
            return;
        }
        assert!(d.new_stream().is_ok());
        assert!(d.synchronize().is_ok());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_stream_resolves_default() {
        if !cuda_driver_available() {
            return;
        }
        let Some(d) = device_or_skip() else {
            return;
        };
        let s = d.default_stream();
        let stream = s.cuda_stream(&d).unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &stream,
            &d.context().unwrap().default_stream()
        ));
    }
}
