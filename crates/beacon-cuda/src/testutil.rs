use crate::device::DeviceId;
use crate::error::LaunchResult;
use crate::launch::LaunchParams;
use crate::launcher::{KernelArg, KernelId, KernelLauncher};
use std::cell::RefCell;

#[derive(Default)]
pub struct Recorder {
    log: RefCell<Vec<(&'static str, usize)>>,
}

impl Recorder {
    pub fn calls(&self) -> Vec<(&'static str, usize)> {
        self.log.borrow().clone()
    }
}

impl KernelLauncher for Recorder {
    fn device(&self) -> DeviceId {
        DeviceId(0)
    }

    fn launch(
        &self,
        kernel: KernelId,
        params: LaunchParams,
        args: &[KernelArg],
    ) -> LaunchResult<()> {
        params.validate()?;
        self.log.borrow_mut().push((kernel.name(), args.len()));
        Ok(())
    }
}
