use beacon_core::{Arena, Dtype, Shape, Tensor};
use crate::device::{Device, Stream};
use crate::device_arena::DeviceArena;
use crate::error::{LaunchError, LaunchResult};

#[cfg(feature = "cuda")]
use crate::cuda::map_driver_error;

fn repr_bytes<T: Dtype>(data: &[T::Repr]) -> &[u8] {
    let nbytes = data.len() * T::SIZE_BYTES;
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), nbytes) }
}

fn repr_bytes_mut<T: Dtype>(data: &mut [T::Repr]) -> &mut [u8] {
    let nbytes = data.len() * T::SIZE_BYTES;
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), nbytes) }
}

fn check_numel<T: Dtype, S: Shape>(data: &[T::Repr]) -> LaunchResult<()> {
    if data.len() != Tensor::<T, S>::NUMEL {
        return Err(LaunchError::ShapeMismatch {
            expected: format!("{} elements", Tensor::<T, S>::NUMEL),
            found: format!("{} elements", data.len()),
        });
    }
    Ok(())
}

fn check_device_bounds<T: Dtype, S: Shape>(
    arena: &DeviceArena,
    tensor: &Tensor<T, S>,
) -> LaunchResult<()> {
    let end = tensor.offset.saturating_add(tensor.nbytes());
    if end > arena.total_bytes() {
        return Err(LaunchError::InvalidLaunchConfig(format!(
            "tensor byte range [{}, {end}) exceeds device arena size {}",
            tensor.offset,
            arena.total_bytes()
        )));
    }
    Ok(())
}

fn check_host_bounds<T: Dtype, S: Shape>(arena: &Arena, tensor: &Tensor<T, S>) -> LaunchResult<()> {
    let end = tensor.offset.saturating_add(tensor.nbytes());
    if end > arena.total_bytes() {
        return Err(LaunchError::InvalidLaunchConfig(format!(
            "tensor byte range [{}, {end}) exceeds host arena size {}",
            tensor.offset,
            arena.total_bytes()
        )));
    }
    Ok(())
}

pub fn copy_htod<T: Dtype, S: Shape>(
    device: &Device,
    stream: &Stream,
    arena: &mut DeviceArena,
    dst: &Tensor<T, S>,
    src: &[T::Repr],
) -> LaunchResult<()> {
    check_numel::<T, S>(src)?;
    check_device_bounds(arena, dst)?;
    copy_bytes_htod(device, stream, arena, dst.offset, repr_bytes::<T>(src))
}

pub fn copy_dtoh<T: Dtype, S: Shape>(
    device: &Device,
    stream: &Stream,
    arena: &DeviceArena,
    src: &Tensor<T, S>,
    dst: &mut [T::Repr],
) -> LaunchResult<()> {
    check_numel::<T, S>(dst)?;
    check_device_bounds(arena, src)?;
    copy_bytes_dtoh(
        device,
        stream,
        arena,
        src.offset,
        repr_bytes_mut::<T>(dst),
    )
}

pub fn copy_htod_from_host_arena<T: Dtype, S: Shape>(
    device: &Device,
    stream: &Stream,
    device_arena: &mut DeviceArena,
    host_arena: &Arena,
    dst: &Tensor<T, S>,
    src: &Tensor<T, S>,
) -> LaunchResult<()> {
    check_host_bounds(host_arena, src)?;
    check_device_bounds(device_arena, dst)?;
    if src.nbytes() != dst.nbytes() {
        return Err(LaunchError::ShapeMismatch {
            expected: format!("{} bytes", dst.nbytes()),
            found: format!("{} bytes", src.nbytes()),
        });
    }
    copy_bytes_htod(
        device,
        stream,
        device_arena,
        dst.offset,
        host_arena.tensor_bytes(src),
    )
}

pub fn copy_dtoh_to_host_arena<T: Dtype, S: Shape>(
    device: &Device,
    stream: &Stream,
    device_arena: &DeviceArena,
    host_arena: &mut Arena,
    src: &Tensor<T, S>,
    dst: &Tensor<T, S>,
) -> LaunchResult<()> {
    check_host_bounds(host_arena, dst)?;
    check_device_bounds(device_arena, src)?;
    if src.nbytes() != dst.nbytes() {
        return Err(LaunchError::ShapeMismatch {
            expected: format!("{} bytes", dst.nbytes()),
            found: format!("{} bytes", src.nbytes()),
        });
    }
    copy_bytes_dtoh(
        device,
        stream,
        device_arena,
        src.offset,
        host_arena.tensor_bytes_mut(dst),
    )
}

fn copy_bytes_htod(
    device: &Device,
    stream: &Stream,
    arena: &mut DeviceArena,
    offset: usize,
    src: &[u8],
) -> LaunchResult<()> {
    if src.is_empty() {
        return Ok(());
    }
    let end = offset.saturating_add(src.len());
    if end > arena.total_bytes() {
        return Err(LaunchError::InvalidLaunchConfig(format!(
            "copy byte range [{offset}, {end}) exceeds device arena size {}",
            arena.total_bytes()
        )));
    }
    #[cfg(feature = "cuda")]
    {
        let cuda = stream.cuda_stream(device)?;
        cuda.memcpy_htod(src, &mut arena.device_slice_mut(offset, src.len()))
            .map_err(map_driver_error)?;
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (device, stream);
        arena.host_bytes_mut(offset, src.len()).copy_from_slice(src);
    }
    Ok(())
}

fn copy_bytes_dtoh(
    device: &Device,
    stream: &Stream,
    arena: &DeviceArena,
    offset: usize,
    dst: &mut [u8],
) -> LaunchResult<()> {
    if dst.is_empty() {
        return Ok(());
    }
    let end = offset.saturating_add(dst.len());
    if end > arena.total_bytes() {
        return Err(LaunchError::InvalidLaunchConfig(format!(
            "copy byte range [{offset}, {end}) exceeds device arena size {}",
            arena.total_bytes()
        )));
    }
    #[cfg(feature = "cuda")]
    {
        let cuda = stream.cuda_stream(device)?;
        cuda.memcpy_dtoh(&arena.device_slice(offset, dst.len()), dst)
            .map_err(map_driver_error)?;
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (device, stream);
        dst.copy_from_slice(arena.host_bytes(offset, dst.len()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use beacon_core::dtype::F32;
    use beacon_core::shape::S2;
    use beacon_core::{ArenaLayout, Region};

    fn small_layout() -> ArenaLayout {
        ArenaLayout {
            weight_bytes: 1 << 20,
            activation_bytes: 1 << 20,
            gradient_bytes: 1 << 20,
        }
    }

    fn open_arenas() -> Option<(Device, Arena, DeviceArena)> {
        #[cfg(feature = "cuda")]
        if std::env::var("BEACON_CUDA_TEST").ok().as_deref() != Some("1") {
            return None;
        }
        let device = Device::new(0).ok()?;
        let layout = small_layout();
        let host = Arena::new(layout);
        let device_arena = DeviceArena::new(&device, layout).ok()?;
        Some((device, host, device_arena))
    }

    #[test]
    fn htod_dtoh_roundtrip() {
        let Some((device, _host, mut dev)) = open_arenas() else {
            return;
        };
        let stream = device.default_stream();
        let t = dev.alloc::<F32, S2<4, 4>>(Region::Weight).unwrap();
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        copy_htod(&device, &stream, &mut dev, &t, &src).unwrap();
        let mut out = vec![0.0f32; 16];
        copy_dtoh(&device, &stream, &dev, &t, &mut out).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn host_arena_to_device_and_back() {
        let Some((device, mut host, mut dev)) = open_arenas() else {
            return;
        };
        let stream = device.default_stream();
        let ht = host.alloc::<F32, S2<2, 3>>(Region::Weight);
        let ht_out = host.alloc::<F32, S2<2, 3>>(Region::Activation);
        let dt = dev.alloc::<F32, S2<2, 3>>(Region::Weight).unwrap();
        let payload: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        {
            let bytes = host.tensor_bytes_mut(&ht);
            let floats =
                unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), payload.len()) };
            floats.copy_from_slice(&payload);
        }
        copy_htod_from_host_arena(&device, &stream, &mut dev, &host, &dt, &ht).unwrap();
        copy_dtoh_to_host_arena(&device, &stream, &dev, &mut host, &dt, &ht_out).unwrap();
        let host_read = host.tensor_bytes(&ht_out);
        let floats: &[f32] =
            unsafe { std::slice::from_raw_parts(host_read.as_ptr().cast(), payload.len()) };
        assert_eq!(floats, payload.as_slice());
    }

    #[test]
    fn rejects_length_mismatch() {
        let Some((device, _host, mut dev)) = open_arenas() else {
            return;
        };
        let stream = device.default_stream();
        let t = dev.alloc::<F32, S2<4, 4>>(Region::Weight).unwrap();
        assert!(copy_htod(&device, &stream, &mut dev, &t, &[1.0f32]).is_err());
    }
}
