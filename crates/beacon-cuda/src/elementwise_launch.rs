use crate::device::Device;
use crate::error::{LaunchError, LaunchResult};
use crate::launch::LaunchParams;
use crate::launcher::KernelArg;
use crate::modules::ModuleCache;

#[cfg(feature = "cuda")]
use cudarc::driver::LaunchConfig;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaSlice, map_driver_error};
#[cfg(feature = "cuda")]
use crate::device_arena::DeviceArena;
#[cfg(feature = "cuda")]
use crate::elementwise::kid;

pub const ELEMENTWISE_MODULE: &str = "elementwise";

#[cfg(feature = "cuda")]
fn validate_f32_elementwise_triple(
    arena: &CudaSlice<u8>,
    args: &[KernelArg],
    kernel: &str,
) -> LaunchResult<(KernelArg, KernelArg, KernelArg, i32)> {
    if args.len() != 3 {
        return Err(LaunchError::InvalidLaunchConfig(format!(
            "{kernel} expects 3 args, got {}",
            args.len()
        )));
    }
    let a0 = args[0];
    let a1 = args[1];
    let a2 = args[2];
    if a0.len_bytes != a1.len_bytes || a0.len_bytes != a2.len_bytes {
        return Err(LaunchError::ShapeMismatch {
            expected: format!("{} bytes", a0.len_bytes),
            found: format!(
                "arg0={}, arg1={}, arg2={}",
                a0.len_bytes, a1.len_bytes, a2.len_bytes
            ),
        });
    }
    if a0.len_bytes % 4 != 0 {
        return Err(LaunchError::InvalidLaunchConfig(format!(
            "{kernel} byte length must be a multiple of 4"
        )));
    }
    let n = (a0.len_bytes / 4) as i32;
    for (label, arg) in [("arg0", a0), ("arg1", a1), ("arg2", a2)] {
        let end = arg.offset.saturating_add(arg.len_bytes);
        if end > arena.len() {
            return Err(LaunchError::InvalidLaunchConfig(format!(
                "{label} byte range [{}, {end}) exceeds arena size {}",
                arg.offset,
                arena.len()
            )));
        }
    }
    Ok((a0, a1, a2, n))
}

#[cfg(feature = "cuda")]
fn launch_elementwise_kernel(
    device: &Device,
    modules: &mut ModuleCache,
    arena: &CudaSlice<u8>,
    params: &LaunchParams,
    kernel: &str,
    off0: u64,
    off1: u64,
    off2: u64,
    n: i32,
) -> LaunchResult<()> {
    let func = modules.function(device, ELEMENTWISE_MODULE, kernel)?;
    let stream = params.stream.cuda_stream(device)?;
    let cfg = LaunchConfig {
        grid_dim: (params.grid.x, params.grid.y, params.grid.z),
        block_dim: (params.block.x, params.block.y, params.block.z),
        shared_mem_bytes: params.shared_mem_bytes,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(arena)
            .arg(&off0)
            .arg(&off1)
            .arg(&off2)
            .arg(&n)
            .launch(cfg)
            .map_err(map_driver_error)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn launch_residual_add_fwd(
    device: &Device,
    modules: &mut ModuleCache,
    arena: &CudaSlice<u8>,
    params: &LaunchParams,
    args: &[KernelArg],
) -> LaunchResult<()> {
    let (out, a, b, n) = validate_f32_elementwise_triple(arena, args, "residual_add_fwd")?;
    launch_elementwise_kernel(
        device,
        modules,
        arena,
        params,
        kid::RESIDUAL_ADD_FWD.name(),
        out.offset as u64,
        a.offset as u64,
        b.offset as u64,
        n,
    )
}

#[cfg(feature = "cuda")]
pub fn launch_residual_add_bwd(
    device: &Device,
    modules: &mut ModuleCache,
    arena: &CudaSlice<u8>,
    params: &LaunchParams,
    args: &[KernelArg],
) -> LaunchResult<()> {
    let (d_a, d_b, d_out, n) =
        validate_f32_elementwise_triple(arena, args, "residual_add_bwd")?;
    launch_elementwise_kernel(
        device,
        modules,
        arena,
        params,
        kid::RESIDUAL_ADD_BWD.name(),
        d_a.offset as u64,
        d_b.offset as u64,
        d_out.offset as u64,
        n,
    )
}

#[cfg(feature = "cuda")]
pub fn arena_slice_from(arena: &DeviceArena) -> CudaSlice<u8> {
    arena.device_buf().clone()
}
