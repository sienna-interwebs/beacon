use crate::dtype::Dtype;
use crate::shape::Shape;
use crate::storage::Region;
use core::marker::PhantomData;

#[derive(Clone, Copy, Debug)]
pub struct Tensor<T: Dtype, S: Shape> {
    pub region: Region,
    pub offset: usize,
    _pd: PhantomData<(T, S)>,
}

impl<T: Dtype, S: Shape> Tensor<T, S> {
    pub const NUMEL: usize = S::NUMEL;
    pub const NBYTES: usize = S::NUMEL * T::SIZE_BYTES;

    pub fn from_offset(region: Region, offset: usize) -> Self {
        Tensor {
            region,
            offset,
            _pd: PhantomData,
        }
    }

    pub const fn numel(&self) -> usize {
        Self::NUMEL
    }

    pub const fn nbytes(&self) -> usize {
        Self::NBYTES
    }

    pub fn dims(&self) -> &'static [usize] {
        S::DIMS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{Bf16, F32};
    use crate::shape::{S1, S2};

    #[test]
    fn const_byte_sizes() {
        assert_eq!(Tensor::<F32, S2<768, 3072>>::NBYTES, 768 * 3072 * 4);
        assert_eq!(Tensor::<Bf16, S2<768, 3072>>::NBYTES, 768 * 3072 * 2);
        assert_eq!(Tensor::<F32, S1<50257>>::NUMEL, 50257);
    }
}
