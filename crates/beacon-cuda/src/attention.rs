use crate::error::LaunchResult;
use crate::launch::LaunchParams;
use crate::launcher::{KernelArg, KernelLauncher};

pub const ATTN_Q_TILE: usize = 64;
pub const ATTN_BLOCK: u32 = 128;

pub fn attention_grid(batch: usize, heads: usize, seqlen: usize) -> LaunchParams {
    let q_blocks = ((seqlen + ATTN_Q_TILE - 1) / ATTN_Q_TILE).max(1) as u32;
    let bh = (batch * heads).max(1) as u32;
    LaunchParams::new((bh, q_blocks), ATTN_BLOCK)
}

pub mod kid {
    use crate::launcher::KernelId;
    pub const ATTENTION_FWD: KernelId = KernelId("attention_adjoint_fwd");
    pub const ATTENTION_BWD: KernelId = KernelId("attention_adjoint_bwd");
}

pub trait AttentionLaunch: KernelLauncher {
    fn attention_fwd(
        &self,
        out: KernelArg,
        q: KernelArg,
        k: KernelArg,
        v: KernelArg,
        lse: KernelArg,
        batch: usize,
        heads: usize,
        seqlen: usize,
        head_dim: usize,
        causal: bool,
    ) -> LaunchResult<()> {
        let _ = (head_dim, causal);
        self.launch(
            kid::ATTENTION_FWD,
            attention_grid(batch, heads, seqlen),
            &[out, q, k, v, lse],
        )
    }

    fn attention_bwd(
        &self,
        dq: KernelArg,
        dk: KernelArg,
        dv: KernelArg,
        dout: KernelArg,
        q: KernelArg,
        k: KernelArg,
        v: KernelArg,
        lse: KernelArg,
        batch: usize,
        heads: usize,
        seqlen: usize,
        head_dim: usize,
        causal: bool,
    ) -> LaunchResult<()> {
        let _ = (head_dim, causal);
        self.launch(
            kid::ATTENTION_BWD,
            attention_grid(batch, heads, seqlen),
            &[dq, dk, dv, dout, q, k, v, lse],
        )
    }
}

impl<T: KernelLauncher + ?Sized> AttentionLaunch for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Recorder;

    fn a() -> KernelArg {
        KernelArg::read(0, 256)
    }

    #[test]
    fn attention_grid_tiles_q() {
        let g = attention_grid(4, 12, 512);
        assert_eq!(g.grid.x, 48);
        assert_eq!(g.grid.y, 8);
    }

    #[test]
    fn attention_routes() {
        let r = Recorder::default();
        r.attention_fwd(a(), a(), a(), a(), a(), 4, 12, 512, 64, true).unwrap();
        r.attention_bwd(a(), a(), a(), a(), a(), a(), a(), a(), 4, 12, 512, 64, true)
            .unwrap();
        assert_eq!(
            r.calls(),
            vec![("attention_adjoint_fwd", 5), ("attention_adjoint_bwd", 8)]
        );
    }
}
