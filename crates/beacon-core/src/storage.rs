#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    Weight,
    Activation,
    Gradient,
}

#[derive(Debug)]
pub struct HostBuffer {
    bytes: Vec<u8>,
}

impl HostBuffer {
    pub fn zeroed(len: usize) -> Self {
        HostBuffer {
            bytes: vec![0u8; len],
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    pub fn slice(&self, offset: usize, len: usize) -> &[u8] {
        &self.bytes[offset..offset + len]
    }

    pub fn slice_mut(&mut self, offset: usize, len: usize) -> &mut [u8] {
        &mut self.bytes[offset..offset + len]
    }
}
