#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

pub mod attention;
pub mod device;
pub mod elementwise;
pub mod error;
pub mod launch;
pub mod launcher;
pub mod matmul;
pub mod mlp;
pub mod norm;
pub mod runtime;

#[cfg(test)]
mod testutil;

pub use attention::AttentionLaunch;
pub use device::{Device, DeviceId, Stream, StreamHandle};
pub use elementwise::{CastKind, ElementwiseLaunch};
pub use error::{LaunchError, LaunchResult};
pub use launch::{Dim3, LaunchParams, MAX_DYNAMIC_SMEM_BYTES, MAX_THREADS_PER_BLOCK};
pub use launcher::{Access, KernelArg, KernelId, KernelLauncher};
pub use matmul::MatmulLaunch;
pub use mlp::SwigluLaunch;
pub use norm::{LayerNormLaunch, RmsNormLaunch};
pub use runtime::Launcher;

#[cfg(test)]
mod table_consistency {
    use crate::{attention, elementwise, matmul, mlp, norm};
    use beacon_adjoint_table::lookup;

    #[test]
    fn launcher_kernel_ids_match_adjoint_table() {
        let pairs = [
            ("matmul", matmul::kid::MATMUL_FWD, matmul::kid::MATMUL_BWD),
            ("linear", matmul::kid::LINEAR_FWD, matmul::kid::LINEAR_BWD),
            ("rmsnorm", norm::kid::RMSNORM_FWD, norm::kid::RMSNORM_BWD),
            (
                "layernorm",
                norm::kid::LAYERNORM_FWD,
                norm::kid::LAYERNORM_BWD,
            ),
            (
                "flash_attention",
                attention::kid::ATTENTION_FWD,
                attention::kid::ATTENTION_BWD,
            ),
            ("swiglu_mlp", mlp::kid::SWIGLU_FWD, mlp::kid::SWIGLU_BWD),
            (
                "residual_add",
                elementwise::kid::RESIDUAL_ADD_FWD,
                elementwise::kid::RESIDUAL_ADD_BWD,
            ),
        ];
        for (op, fwd, bwd) in pairs {
            let entry = lookup(op).unwrap_or_else(|| panic!("missing op {op}"));
            assert_eq!(fwd.name(), entry.forward_launcher, "fwd mismatch for {op}");
            assert_eq!(bwd.name(), entry.backward_launcher, "bwd mismatch for {op}");
        }
    }
}
