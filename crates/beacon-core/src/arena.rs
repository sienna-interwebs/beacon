use crate::dtype::Dtype;
use crate::saved::Saved;
use crate::shape::Shape;
use crate::storage::{HostBuffer, Region};
use crate::tensor::Tensor;

pub const ALIGN: usize = 256;

const fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

#[derive(Clone, Copy, Debug)]
pub struct ArenaLayout {
    pub weight_bytes: usize,
    pub activation_bytes: usize,
    pub gradient_bytes: usize,
}

impl ArenaLayout {
    pub fn total(&self) -> usize {
        align_up(self.weight_bytes, ALIGN)
            + align_up(self.activation_bytes, ALIGN)
            + align_up(self.gradient_bytes, ALIGN)
    }
}

#[derive(Debug)]
struct RegionState {
    base: usize,
    cap: usize,
    cursor: usize,
}

impl RegionState {
    fn bump(&mut self, nbytes: usize) -> usize {
        let start = align_up(self.cursor, ALIGN);
        let end = start + nbytes;
        assert!(
            end <= self.cap,
            "arena region overflow: need {end} bytes, capacity {}",
            self.cap
        );
        self.cursor = end;
        self.base + start
    }
}

#[derive(Debug)]
pub struct Arena {
    buf: HostBuffer,
    weight: RegionState,
    activation: RegionState,
    gradient: RegionState,
}

impl Arena {
    pub fn new(layout: ArenaLayout) -> Self {
        let w = align_up(layout.weight_bytes, ALIGN);
        let a = align_up(layout.activation_bytes, ALIGN);
        let g = align_up(layout.gradient_bytes, ALIGN);
        Arena {
            buf: HostBuffer::zeroed(w + a + g),
            weight: RegionState {
                base: 0,
                cap: w,
                cursor: 0,
            },
            activation: RegionState {
                base: w,
                cap: a,
                cursor: 0,
            },
            gradient: RegionState {
                base: w + a,
                cap: g,
                cursor: 0,
            },
        }
    }

    fn region_mut(&mut self, region: Region) -> &mut RegionState {
        match region {
            Region::Weight => &mut self.weight,
            Region::Activation => &mut self.activation,
            Region::Gradient => &mut self.gradient,
        }
    }

    pub fn alloc<T: Dtype, S: Shape>(&mut self, region: Region) -> Tensor<T, S> {
        let offset = self.region_mut(region).bump(Tensor::<T, S>::NBYTES);
        Tensor::from_offset(region, offset)
    }

    pub fn alloc_saved<T: Dtype, S: Shape>(&mut self) -> Saved<T, S> {
        let offset = self.gradient.bump(Saved::<T, S>::BYTES);
        Saved::at(offset)
    }

    pub fn reset_activations(&mut self) {
        self.activation.cursor = 0;
    }

    pub fn used(&self, region: Region) -> usize {
        match region {
            Region::Weight => self.weight.cursor,
            Region::Activation => self.activation.cursor,
            Region::Gradient => self.gradient.cursor,
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.buf.len()
    }

    pub fn tensor_bytes<T: Dtype, S: Shape>(&self, t: &Tensor<T, S>) -> &[u8] {
        self.buf.slice(t.offset, Tensor::<T, S>::NBYTES)
    }

    pub fn tensor_bytes_mut<T: Dtype, S: Shape>(&mut self, t: &Tensor<T, S>) -> &mut [u8] {
        self.buf.slice_mut(t.offset, Tensor::<T, S>::NBYTES)
    }

    pub fn buffer(&self) -> &HostBuffer {
        &self.buf
    }

    pub fn buffer_mut(&mut self) -> &mut HostBuffer {
        &mut self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{Bf16, F32};
    use crate::shape::S2;

    fn small_arena() -> Arena {
        Arena::new(ArenaLayout {
            weight_bytes: 1 << 20,
            activation_bytes: 1 << 20,
            gradient_bytes: 1 << 20,
        })
    }

    #[test]
    fn allocations_are_aligned_and_sequential() {
        let mut a = small_arena();
        let t0 = a.alloc::<F32, S2<16, 16>>(Region::Weight);
        let t1 = a.alloc::<F32, S2<16, 16>>(Region::Weight);
        assert_eq!(t0.offset % ALIGN, 0);
        assert_eq!(t1.offset % ALIGN, 0);
        let first_end = t0.offset + Tensor::<F32, S2<16, 16>>::NBYTES;
        assert!(t1.offset >= first_end);
        assert_eq!(t1.offset % ALIGN, 0);
    }

    #[test]
    fn regions_are_disjoint() {
        let mut a = small_arena();
        let w = a.alloc::<F32, S2<16, 16>>(Region::Weight);
        let act = a.alloc::<Bf16, S2<16, 16>>(Region::Activation);
        let g = a.alloc_saved::<F32, S2<16, 16>>();
        assert!(w.offset < act.offset);
        assert!(act.offset < g.offset());
    }

    #[test]
    fn activation_reset_rewinds_cursor() {
        let mut a = small_arena();
        let first = a.alloc::<F32, S2<64, 64>>(Region::Activation);
        assert!(a.used(Region::Activation) > 0);
        a.reset_activations();
        assert_eq!(a.used(Region::Activation), 0);
        let again = a.alloc::<F32, S2<64, 64>>(Region::Activation);
        assert_eq!(again.offset, first.offset);
    }

    #[test]
    fn gradient_buffer_offsets_match_saved_bytes() {
        let mut a = small_arena();
        let s0 = a.alloc_saved::<F32, S2<8, 8>>();
        let s1 = a.alloc_saved::<F32, S2<8, 8>>();
        assert_eq!(s1.offset() - s0.offset(), Saved::<F32, S2<8, 8>>::BYTES);
    }
}
