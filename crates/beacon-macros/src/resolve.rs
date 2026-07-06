use beacon_adjoint_table::AdjointEntry;

use crate::ops::OpSite;

#[derive(Clone)]
#[allow(dead_code)]
pub struct ResolvedOpCall {
    pub site: OpSite,
    pub forward_launcher: &'static str,
    pub backward_launcher: &'static str,
    pub entry: &'static AdjointEntry,
}

pub fn resolve_ops(ops: &[OpSite]) -> Result<Vec<ResolvedOpCall>, Vec<syn::Error>> {
    let mut resolved = Vec::with_capacity(ops.len());
    let mut errors = Vec::new();
    for op in ops {
        match beacon_adjoint_table::lookup(&op.name) {
            Some(entry) => resolved.push(ResolvedOpCall {
                site: op.clone(),
                forward_launcher: entry.forward_launcher,
                backward_launcher: entry.backward_launcher,
                entry,
            }),
            None => errors.push(syn::Error::new(
                op.span,
                format!(
                    "no registered adjoint for op `{}`; add it to beacon-adjoint-table",
                    op.name
                ),
            )),
        }
    }
    if errors.is_empty() {
        Ok(resolved)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{OpDest, OpSite};
    use proc_macro2::Span;

    #[test]
    fn resolves_known_ops() {
        let ops = vec![
            OpSite {
                name: "rmsnorm".into(),
                args: vec![],
                dest: OpDest::Tail,
                span: Span::call_site(),
            },
            OpSite {
                name: "linear".into(),
                args: vec![],
                dest: OpDest::Tail,
                span: Span::call_site(),
            },
        ];
        let resolved = resolve_ops(&ops).unwrap();
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].forward_launcher, "rmsnorm_adjoint_fwd");
        assert_eq!(resolved[1].backward_launcher, "linear_bwd");
        assert_eq!(resolved[0].entry.op, "rmsnorm");
    }

    #[test]
    fn unknown_op_collects_error() {
        let ops = vec![OpSite {
            name: "conv2d".into(),
            args: vec![],
            dest: OpDest::Tail,
            span: Span::call_site(),
        }];
        match resolve_ops(&ops) {
            Err(errs) => {
                assert_eq!(errs.len(), 1);
                assert!(errs[0].to_string().contains("conv2d"));
            }
            Ok(_) => panic!("expected error"),
        }
    }
}
