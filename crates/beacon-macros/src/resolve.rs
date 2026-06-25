use beacon_adjoint_table::AdjointEntry;
use proc_macro2::Span;

use crate::ops::OpCall;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResolvedOpCall {
    pub name: String,
    pub arg_count: usize,
    pub span: Span,
    pub forward_launcher: &'static str,
    pub backward_launcher: &'static str,
    pub entry: &'static AdjointEntry,
}

pub fn resolve_ops(ops: &[OpCall]) -> Result<Vec<ResolvedOpCall>, Vec<syn::Error>> {
    let mut resolved = Vec::with_capacity(ops.len());
    let mut errors = Vec::new();
    for op in ops {
        match beacon_adjoint_table::lookup(&op.name) {
            Some(entry) => resolved.push(ResolvedOpCall {
                name: op.name.clone(),
                arg_count: op.arg_count,
                span: op.span,
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
    use crate::ops::OpCall;

    #[test]
    fn resolves_known_ops() {
        let ops = vec![
            OpCall {
                name: "rmsnorm".into(),
                arg_count: 2,
                span: Span::call_site(),
            },
            OpCall {
                name: "linear".into(),
                arg_count: 2,
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
        let ops = vec![OpCall {
            name: "conv2d".into(),
            arg_count: 1,
            span: Span::call_site(),
        }];
        let errs = resolve_ops(&ops).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].to_string().contains("conv2d"));
        assert!(errs[0].to_string().contains("beacon-adjoint-table"));
    }

    #[test]
    fn multiple_unknown_ops_collect_multiple_errors() {
        let ops = vec![
            OpCall {
                name: "foo".into(),
                arg_count: 1,
                span: Span::call_site(),
            },
            OpCall {
                name: "bar".into(),
                arg_count: 1,
                span: Span::call_site(),
            },
        ];
        let errs = resolve_ops(&ops).unwrap_err();
        assert_eq!(errs.len(), 2);
    }
}
