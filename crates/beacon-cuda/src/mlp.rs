use crate::elementwise::elementwise_grid;
use crate::error::LaunchResult;
use crate::launcher::{KernelArg, KernelLauncher};

pub mod kid {
    use crate::launcher::KernelId;
    pub const SWIGLU_FWD: KernelId = KernelId("swiglu_adjoint_fwd");
    pub const SWIGLU_BWD: KernelId = KernelId("swiglu_adjoint_bwd");
}

pub trait SwigluLaunch: KernelLauncher {
    fn swiglu_fwd(
        &self,
        out: KernelArg,
        x: KernelArg,
        gate_proj: KernelArg,
        up_proj: KernelArg,
        down_proj: KernelArg,
        tokens: usize,
        d_model: usize,
        d_ff: usize,
    ) -> LaunchResult<()> {
        let _ = d_model;
        self.launch(
            kid::SWIGLU_FWD,
            elementwise_grid(tokens * d_ff),
            &[out, x, gate_proj, up_proj, down_proj],
        )
    }

    fn swiglu_bwd(
        &self,
        dx: KernelArg,
        dgate_proj: KernelArg,
        dup_proj: KernelArg,
        ddown_proj: KernelArg,
        dy: KernelArg,
        x: KernelArg,
        gate_proj: KernelArg,
        up_proj: KernelArg,
        down_proj: KernelArg,
        tokens: usize,
        d_model: usize,
        d_ff: usize,
    ) -> LaunchResult<()> {
        let _ = d_model;
        self.launch(
            kid::SWIGLU_BWD,
            elementwise_grid(tokens * d_ff),
            &[dx, dgate_proj, dup_proj, ddown_proj, dy, x, gate_proj, up_proj, down_proj],
        )
    }
}

impl<T: KernelLauncher + ?Sized> SwigluLaunch for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Recorder;

    fn a() -> KernelArg {
        KernelArg::read(0, 256)
    }

    #[test]
    fn swiglu_routes() {
        let r = Recorder::default();
        r.swiglu_fwd(a(), a(), a(), a(), a(), 2048, 768, 3072).unwrap();
        r.swiglu_bwd(a(), a(), a(), a(), a(), a(), a(), a(), a(), 2048, 768, 3072)
            .unwrap();
        assert_eq!(
            r.calls(),
            vec![("swiglu_adjoint_fwd", 5), ("swiglu_adjoint_bwd", 9)]
        );
    }
}
