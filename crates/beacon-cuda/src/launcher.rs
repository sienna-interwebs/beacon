use crate::device::DeviceId;
use crate::error::LaunchResult;
use crate::launch::LaunchParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelId(pub &'static str);

impl KernelId {
    pub const fn name(&self) -> &'static str {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelArg {
    pub offset: usize,
    pub len_bytes: usize,
    pub access: Access,
}

impl KernelArg {
    pub const fn read(offset: usize, len_bytes: usize) -> Self {
        KernelArg {
            offset,
            len_bytes,
            access: Access::Read,
        }
    }

    pub const fn write(offset: usize, len_bytes: usize) -> Self {
        KernelArg {
            offset,
            len_bytes,
            access: Access::Write,
        }
    }

    pub const fn read_write(offset: usize, len_bytes: usize) -> Self {
        KernelArg {
            offset,
            len_bytes,
            access: Access::ReadWrite,
        }
    }
}

pub trait KernelLauncher {
    fn device(&self) -> DeviceId;

    fn launch(
        &self,
        kernel: KernelId,
        params: LaunchParams,
        args: &[KernelArg],
    ) -> LaunchResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LaunchError;

    struct RecordingLauncher {
        id: DeviceId,
    }

    impl KernelLauncher for RecordingLauncher {
        fn device(&self) -> DeviceId {
            self.id
        }

        fn launch(
            &self,
            kernel: KernelId,
            params: LaunchParams,
            args: &[KernelArg],
        ) -> LaunchResult<()> {
            params.validate()?;
            if args.is_empty() {
                return Err(LaunchError::InvalidLaunchConfig(format!(
                    "{} called with no args",
                    kernel.name()
                )));
            }
            Ok(())
        }
    }

    #[test]
    fn kernel_id_name() {
        assert_eq!(KernelId("rmsnorm_adjoint_fwd").name(), "rmsnorm_adjoint_fwd");
    }

    #[test]
    fn kernel_arg_access_modes() {
        assert_eq!(KernelArg::read(0, 16).access, Access::Read);
        assert_eq!(KernelArg::write(16, 32).access, Access::Write);
        assert_eq!(KernelArg::read_write(48, 8).access, Access::ReadWrite);
    }

    #[test]
    fn launcher_via_trait_object() {
        let l = RecordingLauncher { id: DeviceId(0) };
        let dynamic: &dyn KernelLauncher = &l;
        assert_eq!(dynamic.device(), DeviceId(0));
        let params = LaunchParams::new(8u32, 64u32);
        let args = [KernelArg::read(0, 256), KernelArg::write(256, 256)];
        assert!(dynamic.launch(KernelId("x"), params, &args).is_ok());
    }

    #[test]
    fn launcher_propagates_invalid_config() {
        let l = RecordingLauncher { id: DeviceId(0) };
        let bad = LaunchParams::new(0u32, 64u32);
        assert!(l.launch(KernelId("x"), bad, &[KernelArg::read(0, 1)]).is_err());
    }
}
