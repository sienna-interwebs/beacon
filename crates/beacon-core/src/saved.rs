use crate::dtype::Dtype;
use crate::shape::Shape;
use core::marker::PhantomData;

pub struct Saved<T: Dtype, S: Shape> {
    offset: usize,
    _pd: PhantomData<(T, S)>,
}

impl<T: Dtype, S: Shape> Saved<T, S> {
    pub const BYTES: usize = S::NUMEL * T::SIZE_BYTES;

    pub fn at(offset: usize) -> Self {
        Saved {
            offset,
            _pd: PhantomData,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn consume(self) -> usize {
        self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{Bf16, F32};
    use crate::shape::{S1, S2};

    #[test]
    fn bytes_is_const_from_shape() {
        assert_eq!(Saved::<F32, S2<512, 768>>::BYTES, 512 * 768 * 4);
        assert_eq!(Saved::<Bf16, S1<768>>::BYTES, 768 * 2);
    }

    #[test]
    fn consume_returns_offset() {
        let s = Saved::<F32, S2<8, 8>>::at(256);
        assert_eq!(s.offset(), 256);
        assert_eq!(s.consume(), 256);
    }
}
