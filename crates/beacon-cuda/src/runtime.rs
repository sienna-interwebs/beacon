use crate::device::{Device, DeviceId};
use crate::error::{LaunchError, LaunchResult};
use crate::launch::LaunchParams;
use crate::launcher::{KernelArg, KernelId, KernelLauncher};

pub struct Launcher {
    device: Device,
}

impl Launcher {
    pub fn new(device: Device) -> Self {
        Launcher { device }
    }

    pub fn on_device(ordinal: usize) -> LaunchResult<Self> {
        Ok(Launcher {
            device: Device::new(ordinal)?,
        })
    }

    pub fn device_ref(&self) -> &Device {
        &self.device
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
        let _ = args;
        #[cfg(feature = "cuda")]
        {
            Err(LaunchError::Unimplemented(kernel.name()))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(LaunchError::Unimplemented(kernel.name()))
        }
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
        assert_eq!(
            l.residual_add(KernelArg::write(0, 256), a(), a(), 64),
            Err(LaunchError::Unimplemented("residual_add_fwd"))
        );
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
