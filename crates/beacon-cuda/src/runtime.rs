use crate::device::{Device, DeviceId};
use crate::elementwise::kid;
#[cfg(feature = "cuda")]
use crate::elementwise_launch::{
    launch_embedding_lookup_fwd, launch_residual_add_bwd, launch_residual_add_fwd,
};
use crate::error::{LaunchError, LaunchResult};
use crate::launch::LaunchParams;
use crate::launcher::{KernelArg, KernelId, KernelLauncher};
#[cfg(feature = "cuda")]
use std::cell::RefCell;

#[cfg(feature = "cuda")]
use crate::modules::ModuleCache;

#[cfg(feature = "cuda")]
use crate::cuda::CudaSlice;

pub struct Launcher {
    device: Device,
    #[cfg(feature = "cuda")]
    modules: RefCell<ModuleCache>,
    #[cfg(feature = "cuda")]
    arena: RefCell<Option<CudaSlice<u8>>>,
}

impl Launcher {
    pub fn new(device: Device) -> Self {
        Launcher {
            device,
            #[cfg(feature = "cuda")]
            modules: RefCell::new(ModuleCache::new()),
            #[cfg(feature = "cuda")]
            arena: RefCell::new(None),
        }
    }

    pub fn on_device(ordinal: usize) -> LaunchResult<Self> {
        Ok(Launcher {
            device: Device::new(ordinal)?,
            #[cfg(feature = "cuda")]
            modules: RefCell::new(ModuleCache::new()),
            #[cfg(feature = "cuda")]
            arena: RefCell::new(None),
        })
    }

    pub fn device_ref(&self) -> &Device {
        &self.device
    }

    #[cfg(feature = "cuda")]
    pub fn bind_device_arena(&self, arena: &crate::device_arena::DeviceArena) {
        *self.arena.borrow_mut() = Some(crate::elementwise_launch::arena_slice_from(arena));
    }
}

impl KernelLauncher for Launcher {
    fn device(&self) -> DeviceId {
        self.device.id()
    }

    fn launch(
        &self,
        kernel: KernelId,
        params: LaunchParams,
        args: &[KernelArg],
    ) -> LaunchResult<()> {
        params.validate()?;
        #[cfg(feature = "cuda")]
        {
            match kernel.name() {
                n if n == kid::RESIDUAL_ADD_FWD.name()
                    || n == kid::RESIDUAL_ADD_BWD.name()
                    || n == kid::EMBEDDING_FWD.name() =>
                {
                    let arena = self.arena.borrow();
                    let Some(buf) = arena.as_ref() else {
                        return Err(LaunchError::InvalidLaunchConfig(
                            "device arena not bound on launcher".into(),
                        ));
                    };
                    let mut modules = self.modules.borrow_mut();
                    return match kernel.name() {
                        n if n == kid::RESIDUAL_ADD_FWD.name() => launch_residual_add_fwd(
                            &self.device,
                            &mut modules,
                            buf,
                            &params,
                            args,
                        ),
                        n if n == kid::RESIDUAL_ADD_BWD.name() => launch_residual_add_bwd(
                            &self.device,
                            &mut modules,
                            buf,
                            &params,
                            args,
                        ),
                        _ => launch_embedding_lookup_fwd(
                            &self.device,
                            &mut modules,
                            buf,
                            &params,
                            args,
                        ),
                    };
                }
                _ => {}
            }
        }
        let _ = args;
        Err(LaunchError::Unimplemented(kernel.name()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::AttentionLaunch;
    use crate::elementwise::{CastKind, ElementwiseLaunch};
    use crate::matmul::MatmulLaunch;
    use crate::mlp::SwigluLaunch;
    use crate::norm::{LayerNormLaunch, RmsNormLaunch};

    fn a() -> KernelArg {
        KernelArg::read(0, 256)
    }

    fn launcher() -> Launcher {
        Launcher::on_device(0).unwrap()
    }

    #[test]
    fn device_id_exposed() {
        assert_eq!(launcher().device(), DeviceId(0));
    }

    #[test]
    fn elementwise_methods_are_unimplemented_with_kernel_name() {
        let l = launcher();
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(
                l.residual_add(KernelArg::write(0, 256), a(), a(), 64),
                Err(LaunchError::Unimplemented("residual_add_fwd"))
            );
        }
        #[cfg(feature = "cuda")]
        {
            assert!(matches!(
                l.residual_add(KernelArg::write(0, 256), a(), a(), 64),
                Err(LaunchError::InvalidLaunchConfig(_))
            ));
            assert!(matches!(
                l.residual_add_backward(
                    KernelArg::read_write(0, 256),
                    KernelArg::read_write(256, 256),
                    a(),
                    64
                ),
                Err(LaunchError::InvalidLaunchConfig(_))
            ));
        }
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(
                l.residual_add_backward(
                    KernelArg::read_write(0, 256),
                    KernelArg::read_write(256, 256),
                    a(),
                    64
                ),
                Err(LaunchError::Unimplemented("residual_add_bwd"))
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(
                l.embedding_lookup(KernelArg::write(0, 256), a(), a(), 8, 32),
                Err(LaunchError::Unimplemented("embedding_lookup_fwd"))
            );
        }
        #[cfg(feature = "cuda")]
        {
            assert!(matches!(
                l.embedding_lookup(KernelArg::write(0, 256), a(), a(), 8, 32),
                Err(LaunchError::InvalidLaunchConfig(_))
            ));
        }
        assert_eq!(
            l.cast(KernelArg::write(0, 256), a(), 64, CastKind::F32ToF8E4M3),
            Err(LaunchError::Unimplemented("cast_f32_to_fp8e4m3"))
        );
    }

    #[test]
    fn all_families_bottom_out_in_unimplemented() {
        let l = launcher();
        assert!(matches!(
            l.rmsnorm_fwd(a(), a(), a(), a(), 8, 8),
            Err(LaunchError::Unimplemented("rmsnorm_adjoint_fwd"))
        ));
        assert!(matches!(
            l.layernorm_fwd(a(), a(), a(), a(), a(), a(), 8, 8),
            Err(LaunchError::Unimplemented("layernorm_adjoint_fwd"))
        ));
        assert!(matches!(
            l.swiglu_fwd(a(), a(), a(), a(), a(), 8, 8, 8),
            Err(LaunchError::Unimplemented("swiglu_adjoint_fwd"))
        ));
        assert!(matches!(
            l.attention_fwd(a(), a(), a(), a(), a(), 1, 1, 64, 64, false),
            Err(LaunchError::Unimplemented("attention_adjoint_fwd"))
        ));
        assert!(matches!(
            l.matmul_fwd(a(), a(), a(), 8, 8, 8),
            Err(LaunchError::Unimplemented("matmul_fwd"))
        ));
    }

    #[test]
    fn invalid_config_errors_before_unimplemented() {
        let l = launcher();
        let bad = LaunchParams::new(0u32, 64u32);
        assert!(matches!(
            l.launch(KernelId("x"), bad, &[a()]),
            Err(LaunchError::InvalidLaunchConfig(_))
        ));
    }
}
