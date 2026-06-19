#![allow(dead_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveMode {
    Store,
    Recompute,
}

#[derive(Clone, Copy, Debug)]
pub struct SavedSpec {
    pub name: &'static str,
    pub mode: SaveMode,
}

#[derive(Clone, Copy, Debug)]
pub struct AdjointEntry {
    pub op: &'static str,
    pub forward_launcher: &'static str,
    pub backward_launcher: &'static str,
    pub saved: &'static [SavedSpec],
    pub input_grads: &'static [&'static str],
    pub param_grads: &'static [&'static str],
}

impl AdjointEntry {
    pub const fn num_saved(&self) -> usize {
        self.saved.len()
    }

    pub const fn num_input_grads(&self) -> usize {
        self.input_grads.len()
    }

    pub const fn num_param_grads(&self) -> usize {
        self.param_grads.len()
    }
}

const fn stored(name: &'static str) -> SavedSpec {
    SavedSpec {
        name,
        mode: SaveMode::Store,
    }
}

const fn recomputed(name: &'static str) -> SavedSpec {
    SavedSpec {
        name,
        mode: SaveMode::Recompute,
    }
}

pub const ADJOINT_TABLE: &[AdjointEntry] = &[
    AdjointEntry {
        op: "matmul",
        forward_launcher: "matmul_fwd",
        backward_launcher: "matmul_bwd",
        saved: &[stored("lhs"), stored("rhs")],
        input_grads: &["dlhs", "drhs"],
        param_grads: &[],
    },
    AdjointEntry {
        op: "linear",
        forward_launcher: "linear_fwd",
        backward_launcher: "linear_bwd",
        saved: &[stored("x"), stored("weight")],
        input_grads: &["dx"],
        param_grads: &["dweight", "dbias"],
    },
    AdjointEntry {
        op: "rmsnorm",
        forward_launcher: "rmsnorm_adjoint_fwd",
        backward_launcher: "rmsnorm_adjoint_bwd",
        saved: &[stored("x"), stored("rms")],
        input_grads: &["dx"],
        param_grads: &["dweight"],
    },
    AdjointEntry {
        op: "layernorm",
        forward_launcher: "layernorm_adjoint_fwd",
        backward_launcher: "layernorm_adjoint_bwd",
        saved: &[stored("x"), stored("mean"), stored("invstd")],
        input_grads: &["dx"],
        param_grads: &["dweight", "dbias"],
    },
    AdjointEntry {
        op: "flash_attention",
        forward_launcher: "attention_adjoint_fwd",
        backward_launcher: "attention_adjoint_bwd",
        saved: &[
            stored("q"),
            stored("k"),
            stored("v"),
            stored("lse"),
            recomputed("scores"),
        ],
        input_grads: &["dq", "dk", "dv"],
        param_grads: &[],
    },
    AdjointEntry {
        op: "swiglu_mlp",
        forward_launcher: "swiglu_adjoint_fwd",
        backward_launcher: "swiglu_adjoint_bwd",
        saved: &[stored("x"), recomputed("gate"), recomputed("up")],
        input_grads: &["dx"],
        param_grads: &["dgate_proj", "dup_proj", "ddown_proj"],
    },
    AdjointEntry {
        op: "residual_add",
        forward_launcher: "residual_add_fwd",
        backward_launcher: "residual_add_bwd",
        saved: &[],
        input_grads: &["dx", "dy"],
        param_grads: &[],
    },
];

const fn str_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

pub const fn lookup(op: &str) -> Option<&'static AdjointEntry> {
    let mut i = 0;
    while i < ADJOINT_TABLE.len() {
        if str_eq(ADJOINT_TABLE[i].op, op) {
            return Some(&ADJOINT_TABLE[i]);
        }
        i += 1;
    }
    None
}

pub const fn is_differentiable(op: &str) -> bool {
    lookup(op).is_some()
}

pub const fn len() -> usize {
    ADJOINT_TABLE.len()
}

pub fn op_names() -> impl Iterator<Item = &'static str> {
    ADJOINT_TABLE.iter().map(|e| e.op)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_known_ops() {
        for op in [
            "matmul",
            "linear",
            "rmsnorm",
            "layernorm",
            "flash_attention",
            "swiglu_mlp",
            "residual_add",
        ] {
            let e = lookup(op).unwrap_or_else(|| panic!("missing op {op}"));
            assert_eq!(e.op, op);
            assert!(!e.forward_launcher.is_empty());
            assert!(!e.backward_launcher.is_empty());
        }
    }

    #[test]
    fn lookup_unknown_is_none() {
        assert!(lookup("not_an_op").is_none());
        assert!(!is_differentiable("conv2d"));
    }

    #[test]
    fn const_lookup_in_const_context() {
        const RMS: bool = is_differentiable("rmsnorm");
        const FAKE: bool = is_differentiable("fake");
        assert!(RMS);
        assert!(!FAKE);
    }

    #[test]
    fn op_names_are_unique() {
        let names: Vec<&str> = op_names().collect();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                assert_ne!(names[i], names[j], "duplicate op name {}", names[i]);
            }
        }
        assert_eq!(names.len(), len());
    }

    #[test]
    fn launchers_are_distinct_per_op() {
        for e in ADJOINT_TABLE {
            assert_ne!(e.forward_launcher, e.backward_launcher, "op {}", e.op);
        }
    }

    #[test]
    fn flash_attention_recomputes_scores() {
        let e = lookup("flash_attention").unwrap();
        let scores = e.saved.iter().find(|s| s.name == "scores").unwrap();
        assert_eq!(scores.mode, SaveMode::Recompute);
        let q = e.saved.iter().find(|s| s.name == "q").unwrap();
        assert_eq!(q.mode, SaveMode::Store);
    }
}
