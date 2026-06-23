use crate::error::LaunchResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamHandle(pub u64);

impl StreamHandle {
    pub const DEFAULT: StreamHandle = StreamHandle(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stream {
    handle: StreamHandle,
}

impl Stream {
    pub fn default_stream() -> Self {
        Stream {
            handle: StreamHandle::DEFAULT,
        }
    }

    pub fn from_handle(handle: StreamHandle) -> Self {
        Stream { handle }
    }

    pub fn handle(&self) -> StreamHandle {
        self.handle
    }

    pub fn is_default(&self) -> bool {
        self.handle == StreamHandle::DEFAULT
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
    raw: u64,
}

impl Device {
    pub fn new(ordinal: usize) -> LaunchResult<Self> {
        Ok(Device {
            id: DeviceId(ordinal),
            #[cfg(feature = "cuda")]
            raw: 0,
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn default_stream(&self) -> Stream {
        Stream::default_stream()
    }

    pub fn new_stream(&self) -> LaunchResult<Stream> {
        Ok(Stream::default_stream())
    }

    pub fn synchronize(&self) -> LaunchResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(!s.is_default());
        assert_eq!(s.handle(), StreamHandle(5));
    }

    #[test]
    fn device_construct_and_sync() {
        let d = Device::new(0).unwrap();
        assert_eq!(d.id(), DeviceId(0));
        assert!(d.default_stream().is_default());
        assert!(d.new_stream().is_ok());
        assert!(d.synchronize().is_ok());
    }
}
