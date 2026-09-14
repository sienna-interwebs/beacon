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
pub fn launch_residual_add_fwd(
    device: &Device,
    modules: &mut ModuleCache,
    arena: &CudaSlice<u8>,
    params: &LaunchParams,
    args: &[KernelArg],
) -> LaunchResult<()> {
    if args.len() != 3 {
        return Err(LaunchError::InvalidLaunchConfig(format!(
            "residual_add_fwd expects 3 args, got {}",
            args.len()
        )));
    }
    let [out, a, b] = [args[0], args[1], args[2]];
    if out.len_bytes != a.len_bytes || out.len_bytes != b.len_bytes {
        return Err(LaunchError::ShapeMismatch {
            expected: format!("{} bytes", out.len_bytes),
            found: format!("out={}, a={}, b={}", out.len_bytes, a.len_bytes, b.len_bytes),
        });
    }
    if out.len_bytes % 4 != 0 {
        return Err(LaunchError::InvalidLaunchConfig(
            "residual_add_fwd byte length must be a multiple of 4".into(),
        ));
    }
    let n = (out.len_bytes / 4) as i32;
    for (label, arg) in [("out", out), ("a", a), ("b", b)] {
        let end = arg.offset.saturating_add(arg.len_bytes);
        if end > arena.len() {
            return Err(LaunchError::InvalidLaunchConfig(format!(
                "{label} byte range [{}, {end}) exceeds arena size {}",
                arg.offset,
                arena.len()
            )));
        }
    }
    let func = modules.function(device, ELEMENTWISE_MODULE, kid::RESIDUAL_ADD_FWD.name())?;
    let stream = params.stream.cuda_stream(device)?;
    let cfg = LaunchConfig {
        grid_dim: (params.grid.x, params.grid.y, params.grid.z),
        block_dim: (params.block.x, params.block.y, params.block.z),
        shared_mem_bytes: params.shared_mem_bytes,
    };
    let out_off = out.offset as u64;
    let a_off = a.offset as u64;
    let b_off = b.offset as u64;
    unsafe {
        stream
            .launch_builder(&func)
            .arg(arena)
            .arg(&out_off)
            .arg(&a_off)
            .arg(&b_off)
            .arg(&n)
            .launch(cfg)
            .map_err(map_driver_error)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn arena_slice_from(arena: &DeviceArena) -> CudaSlice<u8> {
    arena.device_buf().clone()
}
