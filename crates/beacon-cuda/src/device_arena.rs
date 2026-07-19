use beacon_core::arena::ALIGN;
use beacon_core::{ArenaLayout, Dtype, Region, Saved, Shape, Tensor};
use crate::device::Device;
use crate::error::{LaunchError, LaunchResult};

#[cfg(feature = "cuda")]
use crate::cuda::{map_driver_error, CudaSlice, CudaView, CudaViewMut};

const fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

#[derive(Debug)]
struct RegionState {
    base: usize,
    cap: usize,
    cursor: usize,
}

impl RegionState {
    fn bump(&mut self, nbytes: usize) -> LaunchResult<usize> {
        let start = align_up(self.cursor, ALIGN);
        let end = start + nbytes;
        if end > self.cap {
            return Err(LaunchError::OutOfMemory {
                requested: nbytes,
                available: self.cap.saturating_sub(start),
            });
        }
        self.cursor = end;
        Ok(self.base + start)
    }
}

pub struct DeviceArena {
    #[cfg(feature = "cuda")]
    buf: CudaSlice<u8>,
    #[cfg(not(feature = "cuda"))]
    buf: Vec<u8>,
    weight: RegionState,
    activation: RegionState,
    gradient: RegionState,
}

impl DeviceArena {
    pub fn new(device: &Device, layout: ArenaLayout) -> LaunchResult<Self> {
        let w = align_up(layout.weight_bytes, ALIGN);
        let a = align_up(layout.activation_bytes, ALIGN);
        let g = align_up(layout.gradient_bytes, ALIGN);
        let total = w + a + g;
        #[cfg(feature = "cuda")]
        let buf = device
            .context()
            .default_stream()
            .alloc_zeros::<u8>(total)
            .map_err(map_driver_error)?;
        #[cfg(not(feature = "cuda"))]
        let buf = vec![0u8; total];
        Ok(DeviceArena {
            buf,
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
        })
    }

    fn region_mut(&mut self, region: Region) -> &mut RegionState {
        match region {
            Region::Weight => &mut self.weight,
            Region::Activation => &mut self.activation,
            Region::Gradient => &mut self.gradient,
        }
    }

    pub fn alloc<T: Dtype, S: Shape>(&mut self, region: Region) -> LaunchResult<Tensor<T, S>> {
        let offset = self.region_mut(region).bump(Tensor::<T, S>::NBYTES)?;
        Ok(Tensor::from_offset(region, offset))
    }

    pub fn alloc_saved<T: Dtype, S: Shape>(&mut self) -> LaunchResult<Saved<T, S>> {
        let offset = self.gradient.bump(Saved::<T, S>::BYTES)?;
        Ok(Saved::at(offset))
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

    #[cfg(not(feature = "cuda"))]
    pub fn host_bytes(&self, offset: usize, len: usize) -> &[u8] {
        &self.buf[offset..offset + len]
    }

    #[cfg(not(feature = "cuda"))]
    pub fn host_bytes_mut(&mut self, offset: usize, len: usize) -> &mut [u8] {
        &mut self.buf[offset..offset + len]
    }

    #[cfg(feature = "cuda")]
    pub fn device_buf(&self) -> &CudaSlice<u8> {
        &self.buf
    }

    #[cfg(feature = "cuda")]
    pub fn device_slice(&self, offset: usize, len: usize) -> CudaView<'_, u8> {
        self.buf.slice(offset..offset + len)
    }

    #[cfg(feature = "cuda")]
    pub fn device_slice_mut(&mut self, offset: usize, len: usize) -> CudaViewMut<'_, u8> {
        self.buf.slice_mut(offset..offset + len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use beacon_core::dtype::{Bf16, F32};
    use beacon_core::shape::S2;

    fn small_layout() -> ArenaLayout {
        ArenaLayout {
            weight_bytes: 1 << 20,
            activation_bytes: 1 << 20,
            gradient_bytes: 1 << 20,
        }
    }

    fn open_arena() -> Option<(Device, DeviceArena)> {
        #[cfg(feature = "cuda")]
        if std::env::var("BEACON_CUDA_TEST").ok().as_deref() != Some("1") {
            return None;
        }
        let device = Device::new(0).ok()?;
        let arena = DeviceArena::new(&device, small_layout()).ok()?;
        Some((device, arena))
    }

    #[test]
    fn allocations_are_aligned_and_sequential() {
        let Some((_device, mut a)) = open_arena() else {
            return;
        };
        let t0 = a
            .alloc::<F32, S2<16, 16>>(Region::Weight)
            .unwrap();
        let t1 = a
            .alloc::<F32, S2<16, 16>>(Region::Weight)
            .unwrap();
        assert_eq!(t0.offset % ALIGN, 0);
        assert_eq!(t1.offset % ALIGN, 0);
        let first_end = t0.offset + Tensor::<F32, S2<16, 16>>::NBYTES;
        assert!(t1.offset >= first_end);
        assert_eq!(t1.offset % ALIGN, 0);
    }

    #[test]
    fn regions_are_disjoint() {
        let Some((_device, mut a)) = open_arena() else {
            return;
        };
        let w = a
            .alloc::<F32, S2<16, 16>>(Region::Weight)
            .unwrap();
        let act = a
            .alloc::<Bf16, S2<16, 16>>(Region::Activation)
            .unwrap();
        let g = a.alloc_saved::<F32, S2<16, 16>>().unwrap();
        assert!(w.offset < act.offset);
        assert!(act.offset < g.offset());
    }

    #[test]
    fn activation_reset_rewinds_cursor() {
        let Some((_device, mut a)) = open_arena() else {
            return;
        };
        let first = a
            .alloc::<F32, S2<64, 64>>(Region::Activation)
            .unwrap();
        assert!(a.used(Region::Activation) > 0);
        a.reset_activations();
        assert_eq!(a.used(Region::Activation), 0);
        let again = a
            .alloc::<F32, S2<64, 64>>(Region::Activation)
            .unwrap();
        assert_eq!(again.offset, first.offset);
    }

    #[test]
    fn gradient_buffer_offsets_match_saved_bytes() {
        let Some((_device, mut a)) = open_arena() else {
            return;
        };
        let s0 = a.alloc_saved::<F32, S2<8, 8>>().unwrap();
        let s1 = a.alloc_saved::<F32, S2<8, 8>>().unwrap();
        assert_eq!(s1.offset() - s0.offset(), Saved::<F32, S2<8, 8>>::BYTES);
    }

    #[test]
    fn total_bytes_matches_layout() {
        let Some((_device, a)) = open_arena() else {
            return;
        };
        assert_eq!(a.total_bytes(), small_layout().total());
    }

    #[test]
    fn region_overflow_returns_error() {
        let Some((device, _)) = open_arena() else {
            return;
        };
        let mut a = DeviceArena::new(
            &device,
            ArenaLayout {
                weight_bytes: 64,
                activation_bytes: 64,
                gradient_bytes: 64,
            },
        )
        .unwrap();
        assert!(a
            .alloc::<F32, S2<16, 16>>(Region::Weight)
            .is_err());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn single_device_allocation() {
        let Some((_device, a)) = open_arena() else {
            return;
        };
        assert_eq!(a.total_bytes(), small_layout().total());
        assert_eq!(a.device_buf().len(), a.total_bytes());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn device_slice_covers_tensor_bytes() {
        let Some((_device, mut a)) = open_arena() else {
            return;
        };
        let t = a
            .alloc::<F32, S2<4, 4>>(Region::Activation)
            .unwrap();
        let view = a.device_slice(t.offset, t.nbytes());
        assert_eq!(view.len(), t.nbytes());
    }
}
