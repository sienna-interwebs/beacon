use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::{Block, Expr, ItemFn, Path, Stmt};

#[derive(Debug, Clone)]
pub struct OpCall {
    pub name: String,
    pub arg_count: usize,
    pub span: Span,
}

pub fn collect_ops(func: &ItemFn) -> syn::Result<Vec<OpCall>> {
    let mut ops = Vec::new();
    walk_block(&func.block, &mut ops);
    Ok(ops)
}

fn walk_block(block: &Block, ops: &mut Vec<OpCall>) {
    for stmt in &block.stmts {
        walk_stmt(stmt, ops);
    }
}

fn walk_stmt(stmt: &Stmt, ops: &mut Vec<OpCall>) {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &local.init {
                walk_expr(&init.expr, ops);
            }
        }
        Stmt::Expr(expr, _) => walk_expr(expr, ops),
        Stmt::Item(_) => {}
        _ => {}
    }
}

fn walk_expr(expr: &Expr, ops: &mut Vec<OpCall>) {
    match expr {
        Expr::Call(call) => {
            for arg in &call.args {
                walk_expr(arg, ops);
            }
            if let Some(name) = callee_op_name(&call.func) {
                ops.push(OpCall {
                    name,
                    arg_count: call.args.len(),
                    span: call.span(),
                });
            }
        }
        Expr::MethodCall(m) => {
            walk_expr(&m.receiver, ops);
            for arg in &m.args {
                walk_expr(arg, ops);
            }
        }
        Expr::Binary(b) => {
            walk_expr(&b.left, ops);
            walk_expr(&b.right, ops);
        }
        Expr::Unary(u) => walk_expr(&u.expr, ops),
        Expr::Cast(c) => walk_expr(&c.expr, ops),
        Expr::Reference(r) => walk_expr(&r.expr, ops),
        Expr::If(i) => {
            walk_expr(&i.cond, ops);
            walk_block(&i.then_branch, ops);
            if let Some((_, else_branch)) = &i.else_branch {
                if let Expr::Block(b) = else_branch.as_ref() {
                    walk_block(&b.block, ops);
                } else {
                    walk_expr(else_branch, ops);
                }
            }
        }
        Expr::Block(b) => walk_block(&b.block, ops),
        Expr::Assign(a) => {
            walk_expr(&a.right, ops);
            walk_expr(&a.left, ops);
        }
        Expr::Field(f) => walk_expr(&f.base, ops),
        Expr::Index(i) => {
            walk_expr(&i.expr, ops);
            walk_expr(&i.index, ops);
        }
        Expr::Array(a) => {
            for elem in &a.elems {
                walk_expr(elem, ops);
            }
        }
        Expr::Tuple(t) => {
            for elem in &t.elems {
                walk_expr(elem, ops);
            }
        }
        Expr::Group(g) => walk_expr(&g.expr, ops),
        Expr::Paren(p) => walk_expr(&p.expr, ops),
        Expr::Match(m) => {
            walk_expr(&m.expr, ops);
            for arm in &m.arms {
                if let Some((_, guard)) = &arm.guard {
                    walk_expr(guard, ops);
                }
                walk_expr(&arm.body, ops);
            }
        }
        Expr::Return(r) => {
            if let Some(v) = &r.expr {
                walk_expr(v, ops);
            }
        }
        _ => {}
    }
}

fn callee_op_name(func: &Expr) -> Option<String> {
    let path = match func {
        Expr::Path(p) => &p.path,
        _ => return None,
    };
    path_op_name(path)
}

fn path_op_name(path: &Path) -> Option<String> {
    let seg = path.segments.last()?;
    let name = seg.ident.to_string();
    if !is_op_callee(&name) {
        return None;
    }
    if path.segments.len() >= 2 {
        let first = path.segments.first()?.ident.to_string();
        if first.starts_with(|c: char| c.is_ascii_uppercase()) {
            return None;
        }
    }
    Some(name)
}

fn is_op_callee(name: &str) -> bool {
    name.starts_with(|c: char| c.is_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn sequential_let_bindings() {
        let f: ItemFn = parse_quote! {
            fn block(x: Tensor<F32, S2<8, 8>>, w: Tensor<F32, S2<8, 8>>) -> Tensor<F32, S2<8, 8>> {
                let a = rmsnorm(x, w);
                let b = linear(a, w);
                b
            }
        };
        let ops = collect_ops(&f).unwrap();
        assert_eq!(
            ops.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
            vec!["rmsnorm", "linear"]
        );
        assert_eq!(ops[0].arg_count, 2);
        assert_eq!(ops[1].arg_count, 2);
    }

    #[test]
    fn nested_call_order_inner_first() {
        let f: ItemFn = parse_quote! {
            fn block(x: Tensor<F32, S2<8, 8>>, w: Tensor<F32, S2<8, 8>>) -> Tensor<F32, S2<8, 8>> {
                linear(rmsnorm(x, w), w)
            }
        };
        let ops = collect_ops(&f).unwrap();
        assert_eq!(
            ops.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
            vec!["rmsnorm", "linear"]
        );
    }

    #[test]
    fn qualified_path_uses_last_segment() {
        let f: ItemFn = parse_quote! {
            fn block(x: Tensor<F32, S2<8, 8>>, w: Tensor<F32, S2<8, 8>>) -> Tensor<F32, S2<8, 8>> {
                beacon_ops::residual_add(x, w)
            }
        };
        let ops = collect_ops(&f).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].name, "residual_add");
    }

    #[test]
    fn skips_type_constructor_calls() {
        let f: ItemFn = parse_quote! {
            fn block(x: Tensor<F32, S2<8, 8>>) -> Tensor<F32, S2<8, 8>> {
                Tensor::<F32, S2<8, 8>>::from_offset(beacon_core::Region::Activation, 0)
            }
        };
        let ops = collect_ops(&f).unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn grad_gate_chain() {
        let f: ItemFn = parse_quote! {
            fn gate(
                x: Tensor<F32, S2<512, 768>>,
                w: Tensor<F32, S2<768, 768>>,
                w2: Tensor<F32, S2<768, 768>>,
            ) -> Tensor<F32, S2<512, 768>> {
                let h = linear(x, w);
                rmsnorm(h, w2)
            }
        };
        let ops = collect_ops(&f).unwrap();
        assert_eq!(
            ops.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
            vec!["linear", "rmsnorm"]
        );
    }
}
