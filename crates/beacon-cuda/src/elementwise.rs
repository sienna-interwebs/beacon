use crate::error::LaunchResult;
use crate::launch::LaunchParams;
use crate::launcher::{KernelArg, KernelId, KernelLauncher};

pub const ELEMENTWISE_BLOCK: u32 = 256;

pub fn elementwise_grid(n: usize) -> LaunchParams {
    let block = ELEMENTWISE_BLOCK;
    let grid = ((n as u64 + block as u64 - 1) / block as u64).max(1) as u32;
    LaunchParams::new(grid, block)
}

pub mod kid {
    use crate::launcher::KernelId;
    pub const RESIDUAL_ADD_FWD: KernelId = KernelId("residual_add_fwd");
    pub const RESIDUAL_ADD_BWD: KernelId = KernelId("residual_add_bwd");
    pub const EMBEDDING_FWD: KernelId = KernelId("embedding_lookup_fwd");
    pub const EMBEDDING_BWD: KernelId = KernelId("embedding_lookup_bwd");
    pub const DROPOUT_FWD: KernelId = KernelId("dropout_fwd");
    pub const DROPOUT_BWD: KernelId = KernelId("dropout_bwd");
    pub const CAST_F32_TO_FP8E4M3: KernelId = KernelId("cast_f32_to_fp8e4m3");
    pub const CAST_FP8E4M3_TO_F32: KernelId = KernelId("cast_fp8e4m3_to_f32");
    pub const CAST_F32_TO_BF16: KernelId = KernelId("cast_f32_to_bf16");
    pub const CAST_BF16_TO_F32: KernelId = KernelId("cast_bf16_to_f32");
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    F32ToF8E4M3,
    F8E4M3ToF32,
    F32ToBf16,
    Bf16ToF32,
}

impl CastKind {
    pub const fn kernel(self) -> KernelId {
        match self {
            CastKind::F32ToF8E4M3 => kid::CAST_F32_TO_FP8E4M3,
            CastKind::F8E4M3ToF32 => kid::CAST_FP8E4M3_TO_F32,
            CastKind::F32ToBf16 => kid::CAST_F32_TO_BF16,
            CastKind::Bf16ToF32 => kid::CAST_BF16_TO_F32,
        }
    }
}

pub trait ElementwiseLaunch: KernelLauncher {
    fn residual_add(
        &self,
        out: KernelArg,
        a: KernelArg,
        b: KernelArg,
        n: usize,
    ) -> LaunchResult<()> {
        self.launch(kid::RESIDUAL_ADD_FWD, elementwise_grid(n), &[out, a, b])
    }

    fn residual_add_backward(
        &self,
        d_a: KernelArg,
        d_b: KernelArg,
        d_out: KernelArg,
        n: usize,
    ) -> LaunchResult<()> {
        self.launch(kid::RESIDUAL_ADD_BWD, elementwise_grid(n), &[d_a, d_b, d_out])
    }

    fn embedding_lookup(
        &self,
        out: KernelArg,
        table: KernelArg,
        indices: KernelArg,
        tokens: usize,
        dim: usize,
    ) -> LaunchResult<()> {
        self.launch(
            kid::EMBEDDING_FWD,
            elementwise_grid(tokens * dim),
            &[out, table, indices],
        )
    }

    fn embedding_backward(
        &self,
        d_table: KernelArg,
        d_out: KernelArg,
        indices: KernelArg,
        tokens: usize,
        dim: usize,
    ) -> LaunchResult<()> {
        self.launch(
            kid::EMBEDDING_BWD,
            elementwise_grid(tokens * dim),
            &[d_table, d_out, indices],
        )
    }

    fn dropout_forward(
        &self,
        out: KernelArg,
        input: KernelArg,
        mask: KernelArg,
        n: usize,
        p: f32,
        seed: u64,
    ) -> LaunchResult<()> {
        let _ = (p, seed);
        self.launch(kid::DROPOUT_FWD, elementwise_grid(n), &[out, input, mask])
    }

    fn dropout_backward(
        &self,
        d_in: KernelArg,
        d_out: KernelArg,
        mask: KernelArg,
        n: usize,
    ) -> LaunchResult<()> {
        self.launch(kid::DROPOUT_BWD, elementwise_grid(n), &[d_in, d_out, mask])
    }

    fn cast(
        &self,
        dst: KernelArg,
        src: KernelArg,
        n: usize,
        kind: CastKind,
    ) -> LaunchResult<()> {
        self.launch(kind.kernel(), elementwise_grid(n), &[dst, src])
    }
}

impl<T: KernelLauncher + ?Sized> ElementwiseLaunch for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceId;
    use crate::launcher::Access;
    use std::cell::RefCell;

    #[derive(Default)]
    struct Recorder {
        calls: RefCell<Vec<(&'static str, usize)>>,
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
            self.calls.borrow_mut().push((kernel.name(), args.len()));
            Ok(())
        }
    }

    fn arg() -> KernelArg {
        KernelArg::read(0, 256)
    }

    #[test]
    fn grid_is_nonzero_even_for_zero_elements() {
        assert!(elementwise_grid(0).grid.total() >= 1);
        assert_eq!(elementwise_grid(513).grid.x, 3);
    }

    #[test]
    fn cast_kind_maps_to_kernels() {
        assert_eq!(CastKind::F32ToF8E4M3.kernel(), kid::CAST_F32_TO_FP8E4M3);
        assert_eq!(CastKind::Bf16ToF32.kernel(), kid::CAST_BF16_TO_F32);
    }

    #[test]
    fn families_route_to_expected_kernels() {
        let r = Recorder::default();
        r.residual_add(KernelArg::write(0, 256), arg(), arg(), 64).unwrap();
        r.residual_add_backward(KernelArg::read_write(0, 256), KernelArg::read_write(256, 256), arg(), 64)
            .unwrap();
        r.embedding_lookup(KernelArg::write(0, 256), arg(), arg(), 8, 32).unwrap();
        r.embedding_backward(KernelArg::read_write(0, 256), arg(), arg(), 8, 32).unwrap();
        r.dropout_forward(KernelArg::write(0, 256), arg(), KernelArg::write(256, 256), 64, 0.0, 7)
            .unwrap();
        r.dropout_backward(KernelArg::write(0, 256), arg(), arg(), 64).unwrap();
        r.cast(KernelArg::write(0, 256), arg(), 64, CastKind::F32ToF8E4M3).unwrap();

        let calls = r.calls.borrow();
        assert_eq!(
            calls.as_slice(),
            &[
                ("residual_add_fwd", 3),
                ("residual_add_bwd", 3),
                ("embedding_lookup_fwd", 3),
                ("embedding_lookup_bwd", 3),
                ("dropout_fwd", 3),
                ("dropout_bwd", 3),
                ("cast_f32_to_fp8e4m3", 2),
            ]
        );
    }

    #[test]
    fn access_modes_preserved() {
        let w = KernelArg::write(0, 256);
        assert_eq!(w.access, Access::Write);
    }
}
