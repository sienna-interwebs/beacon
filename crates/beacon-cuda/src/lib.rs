#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

pub mod attention;
pub mod blas;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod device;
pub mod device_arena;
pub mod elementwise;
pub mod error;
pub mod launch;
pub mod launcher;
pub mod matmul;
pub mod mlp;
pub mod modules;
pub mod norm;
pub mod ptx;
pub mod runtime;

#[cfg(test)]
mod testutil;

pub use attention::AttentionLaunch;
pub use blas::BlasHandles;
pub use device::{Device, DeviceId, Stream, StreamHandle};
pub use device_arena::DeviceArena;
pub use elementwise::{CastKind, ElementwiseLaunch};
pub use error::{LaunchError, LaunchResult};
pub use launch::{Dim3, LaunchParams, MAX_DYNAMIC_SMEM_BYTES, MAX_THREADS_PER_BLOCK};
pub use launcher::{Access, KernelArg, KernelId, KernelLauncher};
pub use matmul::MatmulLaunch;
pub use mlp::SwigluLaunch;
pub use modules::ModuleCache;
pub use norm::{LayerNormLaunch, RmsNormLaunch};
pub use ptx::{ptx_path, PtxArtifact, PTX_ARTIFACTS};
pub use runtime::Launcher;

pub const fn cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}

#[cfg(feature = "cuda")]
pub use cuda::{CudaBlas, CudaBlasLT, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, Ptx};

#[cfg(test)]
mod cuda_feature {
    use super::*;

    #[test]
    fn cuda_enabled_matches_cfg() {
        assert_eq!(cuda_enabled(), cfg!(feature = "cuda"));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn ptx_manifest_is_available() {
        let _ = PTX_ARTIFACTS.len();
        assert!(ptx_path("nonexistent_kernel").is_none());
    }
}

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
