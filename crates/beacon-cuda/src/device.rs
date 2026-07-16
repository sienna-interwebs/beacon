use crate::error::LaunchResult;

#[cfg(not(feature = "cuda"))]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, CudaStream, map_driver_error};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamHandle(pub u64);

impl StreamHandle {
    pub const DEFAULT: StreamHandle = StreamHandle(0);
}

#[cfg(not(feature = "cuda"))]
static HOST_STUB_STREAM_COUNTER: AtomicU64 = AtomicU64::new(1);

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
            StreamInner::Default => Ok(device.context().default_stream().clone()),
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

    pub fn synchronize(&self, device: &Device) -> LaunchResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.cuda_stream(device)?
                .synchronize()
                .map_err(map_driver_error)?;
        }
        Ok(())
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
    ctx: Arc<CudaContext>,
}

impl Device {
    pub fn new(ordinal: usize) -> LaunchResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let ctx = CudaContext::new(ordinal).map_err(map_driver_error)?;
            Ok(Device {
                id: DeviceId(ordinal),
                ctx: Arc::from(ctx),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(Device {
                id: DeviceId(ordinal),
            })
        }
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    #[cfg(feature = "cuda")]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn default_stream(&self) -> Stream {
        Stream::default_stream()
    }

    pub fn new_stream(&self) -> LaunchResult<Stream> {
        #[cfg(feature = "cuda")]
        {
            let stream = self.context().new_stream().map_err(map_driver_error)?;
            Ok(Stream::from_cuda(stream))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let id = HOST_STUB_STREAM_COUNTER.fetch_add(1, Ordering::Relaxed);
            Ok(Stream::from_handle(StreamHandle(id)))
        }
    }

    pub fn synchronize(&self) -> LaunchResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.context().synchronize().map_err(map_driver_error)?;
        }
        Ok(())
    }

    pub fn shutdown(self) -> LaunchResult<()> {
        self.synchronize()
    }
}

#[cfg(feature = "cuda")]
impl Drop for Device {
    fn drop(&mut self) {
        let _ = self.ctx.synchronize();
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
        assert!(d.new_stream().is_ok());
        assert!(d.synchronize().is_ok());
        assert!(d.shutdown().is_ok());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_stream_resolves_default() {
        let Some(d) = device_or_skip() else {
            return;
        };
        let s = d.default_stream();
        let stream = s.cuda_stream(&d).unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &stream,
            &d.context().default_stream()
        ));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn context_initialized_on_new() {
        let Some(d) = device_or_skip() else {
            return;
        };
        assert_eq!(d.context().ordinal(), d.id().0);
    }

    #[test]
    fn host_stub_device_without_cuda_feature() {
        #[cfg(not(feature = "cuda"))]
        {
            let d = Device::new(0).unwrap();
            assert_eq!(d.id(), DeviceId(0));
            assert!(d.synchronize().is_ok());
        }
    }

    #[test]
    fn host_stub_new_stream_returns_distinct_handles() {
        #[cfg(not(feature = "cuda"))]
        {
            let d = Device::new(0).unwrap();
            let a = d.new_stream().unwrap();
            let b = d.new_stream().unwrap();
            assert!(!a.is_default());
            assert!(!b.is_default());
            assert_ne!(a.handle(), b.handle());
            assert!(a.synchronize(&d).is_ok());
            assert!(b.synchronize(&d).is_ok());
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn owned_stream_is_non_default_and_syncs() {
        let Some(d) = device_or_skip() else {
            return;
        };
        let s = d.new_stream().unwrap();
        assert!(!s.is_default());
        assert!(s.synchronize(&d).is_ok());
        let default = d.default_stream();
        assert!(default.is_default());
        assert!(default.synchronize(&d).is_ok());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn owned_streams_are_distinct_from_default() {
        let Some(d) = device_or_skip() else {
            return;
        };
        let owned = d.new_stream().unwrap();
        let default_cuda = d.context().default_stream();
        let owned_cuda = owned.cuda_stream(&d).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&owned_cuda, default_cuda));
    }
}
