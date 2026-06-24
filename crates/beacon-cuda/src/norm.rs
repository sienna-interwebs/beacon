use crate::error::LaunchResult;
use crate::launch::LaunchParams;
use crate::launcher::{KernelArg, KernelLauncher};

pub const NORM_BLOCK: u32 = 256;

pub fn rowwise(rows: usize) -> LaunchParams {
    LaunchParams::new((rows as u32).max(1), NORM_BLOCK)
}

pub mod kid {
    use crate::launcher::KernelId;
    pub const RMSNORM_FWD: KernelId = KernelId("rmsnorm_adjoint_fwd");
    pub const RMSNORM_BWD: KernelId = KernelId("rmsnorm_adjoint_bwd");
    pub const LAYERNORM_FWD: KernelId = KernelId("layernorm_adjoint_fwd");
    pub const LAYERNORM_BWD: KernelId = KernelId("layernorm_adjoint_bwd");
}

pub trait RmsNormLaunch: KernelLauncher {
    fn rmsnorm_fwd(
        &self,
        out: KernelArg,
        x: KernelArg,
        weight: KernelArg,
        rms: KernelArg,
        tokens: usize,
        dim: usize,
    ) -> LaunchResult<()> {
        let _ = dim;
        self.launch(kid::RMSNORM_FWD, rowwise(tokens), &[out, x, weight, rms])
    }

    fn rmsnorm_bwd(
        &self,
        dx: KernelArg,
        dweight: KernelArg,
        dy: KernelArg,
        x: KernelArg,
        rms: KernelArg,
        weight: KernelArg,
        tokens: usize,
        dim: usize,
    ) -> LaunchResult<()> {
        let _ = dim;
        self.launch(
            kid::RMSNORM_BWD,
            rowwise(tokens),
            &[dx, dweight, dy, x, rms, weight],
        )
    }
}

impl<T: KernelLauncher + ?Sized> RmsNormLaunch for T {}

pub trait LayerNormLaunch: KernelLauncher {
    fn layernorm_fwd(
        &self,
        out: KernelArg,
        x: KernelArg,
        weight: KernelArg,
        bias: KernelArg,
        mean: KernelArg,
        invstd: KernelArg,
        tokens: usize,
        dim: usize,
    ) -> LaunchResult<()> {
        let _ = dim;
        self.launch(
            kid::LAYERNORM_FWD,
            rowwise(tokens),
            &[out, x, weight, bias, mean, invstd],
        )
    }

    fn layernorm_bwd(
        &self,
        dx: KernelArg,
        dweight: KernelArg,
        dbias: KernelArg,
        dy: KernelArg,
        x: KernelArg,
        mean: KernelArg,
        invstd: KernelArg,
        weight: KernelArg,
        tokens: usize,
        dim: usize,
    ) -> LaunchResult<()> {
        let _ = dim;
        self.launch(
            kid::LAYERNORM_BWD,
            rowwise(tokens),
            &[dx, dweight, dbias, dy, x, mean, invstd, weight],
        )
    }
}

impl<T: KernelLauncher + ?Sized> LayerNormLaunch for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Recorder;

    fn a() -> KernelArg {
        KernelArg::read(0, 256)
    }

    #[test]
    fn rmsnorm_routes() {
        let r = Recorder::default();
        r.rmsnorm_fwd(a(), a(), a(), a(), 512, 768).unwrap();
        r.rmsnorm_bwd(a(), a(), a(), a(), a(), a(), 512, 768).unwrap();
        assert_eq!(
            r.calls(),
            vec![("rmsnorm_adjoint_fwd", 4), ("rmsnorm_adjoint_bwd", 6)]
        );
    }

    #[test]
    fn layernorm_routes() {
        let r = Recorder::default();
        r.layernorm_fwd(a(), a(), a(), a(), a(), a(), 512, 768).unwrap();
        r.layernorm_bwd(a(), a(), a(), a(), a(), a(), a(), a(), 512, 768).unwrap();
        assert_eq!(
            r.calls(),
            vec![("layernorm_adjoint_fwd", 6), ("layernorm_adjoint_bwd", 8)]
        );
    }
}
