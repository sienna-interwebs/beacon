use crate::error::LaunchResult;
use crate::launch::LaunchParams;
use crate::launcher::{KernelArg, KernelLauncher};

pub mod kid {
    use crate::launcher::KernelId;
    pub const MATMUL_FWD: KernelId = KernelId("matmul_fwd");
    pub const MATMUL_BWD: KernelId = KernelId("matmul_bwd");
    pub const LINEAR_FWD: KernelId = KernelId("linear_fwd");
    pub const LINEAR_BWD: KernelId = KernelId("linear_bwd");
}

fn cublas_params() -> LaunchParams {
    LaunchParams::new(1u32, 1u32)
}

pub trait MatmulLaunch: KernelLauncher {
    fn matmul_fwd(
        &self,
        out: KernelArg,
        lhs: KernelArg,
        rhs: KernelArg,
        m: usize,
        k: usize,
        n: usize,
    ) -> LaunchResult<()> {
        let _ = (m, k, n);
        self.launch(kid::MATMUL_FWD, cublas_params(), &[out, lhs, rhs])
    }

    fn matmul_bwd(
        &self,
        dlhs: KernelArg,
        drhs: KernelArg,
        dout: KernelArg,
        lhs: KernelArg,
        rhs: KernelArg,
        m: usize,
        k: usize,
        n: usize,
    ) -> LaunchResult<()> {
        let _ = (m, k, n);
        self.launch(kid::MATMUL_BWD, cublas_params(), &[dlhs, drhs, dout, lhs, rhs])
    }

    fn linear_fwd(
        &self,
        out: KernelArg,
        x: KernelArg,
        weight: KernelArg,
        bias: KernelArg,
        m: usize,
        k: usize,
        n: usize,
    ) -> LaunchResult<()> {
        let _ = (m, k, n);
        self.launch(kid::LINEAR_FWD, cublas_params(), &[out, x, weight, bias])
    }

    fn linear_bwd(
        &self,
        dx: KernelArg,
        dweight: KernelArg,
        dbias: KernelArg,
        dout: KernelArg,
        x: KernelArg,
        weight: KernelArg,
        m: usize,
        k: usize,
        n: usize,
    ) -> LaunchResult<()> {
        let _ = (m, k, n);
        self.launch(
            kid::LINEAR_BWD,
            cublas_params(),
            &[dx, dweight, dbias, dout, x, weight],
        )
    }
}

impl<T: KernelLauncher + ?Sized> MatmulLaunch for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Recorder;

    fn a() -> KernelArg {
        KernelArg::read(0, 256)
    }

    #[test]
    fn matmul_and_linear_route() {
        let r = Recorder::default();
        r.matmul_fwd(a(), a(), a(), 2048, 768, 3072).unwrap();
        r.matmul_bwd(a(), a(), a(), a(), a(), 2048, 768, 3072).unwrap();
        r.linear_fwd(a(), a(), a(), a(), 2048, 768, 50257).unwrap();
        r.linear_bwd(a(), a(), a(), a(), a(), a(), 2048, 768, 50257).unwrap();
        assert_eq!(
            r.calls(),
            vec![
                ("matmul_fwd", 3),
                ("matmul_bwd", 5),
                ("linear_fwd", 4),
                ("linear_bwd", 6),
            ]
        );
    }
}
