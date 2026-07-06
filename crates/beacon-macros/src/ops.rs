use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::{Block, Expr, Ident, ItemFn, Local, Pat, Path, Stmt, Type};

#[derive(Clone)]
pub struct OpSite {
    pub name: String,
    pub args: Vec<Expr>,
    pub dest: OpDest,
    pub span: Span,
}

#[derive(Clone)]
pub enum OpDest {
    Let { ident: Ident, ty: Type },
    Tail,
}

pub fn collect_ops(func: &ItemFn) -> syn::Result<Vec<OpSite>> {
    let mut ops = Vec::new();
    walk_block(&func.block, &mut ops)?;
    Ok(ops)
}

fn walk_block(block: &Block, ops: &mut Vec<OpSite>) -> syn::Result<()> {
    let n = block.stmts.len();
    for (i, stmt) in block.stmts.iter().enumerate() {
        let tail = i + 1 == n;
        walk_stmt(stmt, ops, tail)?;
    }
    Ok(())
}

fn walk_stmt(stmt: &Stmt, ops: &mut Vec<OpSite>, tail: bool) -> syn::Result<()> {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &local.init {
                let dest = let_dest(local)?;
                walk_expr(&init.expr, ops, Some(dest))?;
            }
        }
        Stmt::Expr(expr, _) => {
            let dest = if tail {
                Some(OpDest::Tail)
            } else {
                None
            };
            walk_expr(expr, ops, dest)?;
        }
        Stmt::Item(_) => {}
        _ => {}
    }
    Ok(())
}

fn let_dest(local: &Local) -> syn::Result<OpDest> {
    match &local.pat {
        Pat::Type(pt) => {
            let ident = pat_ident(&pt.pat)?;
            Ok(OpDest::Let {
                ident,
                ty: (*pt.ty).clone(),
            })
        }
        _ => Err(syn::Error::new_spanned(
            &local.pat,
            "intermediate op results require a type annotation: let x: Tensor<T, S> = op(...)",
        )),
    }
}

fn pat_ident(pat: &Pat) -> syn::Result<Ident> {
    match pat {
        Pat::Ident(p) => Ok(p.ident.clone()),
        _ => Err(syn::Error::new_spanned(
            pat,
            "intermediate bindings must be simple identifiers",
        )),
    }
}

fn walk_expr(expr: &Expr, ops: &mut Vec<OpSite>, dest: Option<OpDest>) -> syn::Result<()> {
    match expr {
        Expr::Call(call) => {
            for arg in &call.args {
                walk_expr(arg, ops, None)?;
            }
            if let Some(name) = callee_op_name(&call.func) {
                let site_dest = dest.ok_or_else(|| {
                    syn::Error::new_spanned(
                        &call.func,
                        "nested op calls are not supported; bind each op to a typed let first",
                    )
                })?;
                ops.push(OpSite {
                    name,
                    args: call.args.iter().cloned().collect(),
                    dest: site_dest,
                    span: call.func.span(),
                });
            } else if dest.is_some() {
                walk_expr_inner(expr, ops)?;
            }
        }
        _ => walk_expr_inner(expr, ops)?,
    }
    Ok(())
}

fn walk_expr_inner(expr: &Expr, ops: &mut Vec<OpSite>) -> syn::Result<()> {
    match expr {
        Expr::MethodCall(m) => {
            walk_expr(&m.receiver, ops, None)?;
            for arg in &m.args {
                walk_expr(arg, ops, None)?;
            }
        }
        Expr::Binary(b) => {
            walk_expr(&b.left, ops, None)?;
            walk_expr(&b.right, ops, None)?;
        }
        Expr::Unary(u) => walk_expr(&u.expr, ops, None)?,
        Expr::Cast(c) => walk_expr(&c.expr, ops, None)?,
        Expr::Reference(r) => walk_expr(&r.expr, ops, None)?,
        Expr::If(i) => {
            walk_expr(&i.cond, ops, None)?;
            walk_block(&i.then_branch, ops)?;
            if let Some((_, else_branch)) = &i.else_branch {
                if let Expr::Block(b) = else_branch.as_ref() {
                    walk_block(&b.block, ops)?;
                } else {
                    walk_expr(else_branch, ops, None)?;
                }
            }
        }
        Expr::Block(b) => walk_block(&b.block, ops)?,
        Expr::Assign(a) => {
            walk_expr(&a.right, ops, None)?;
            walk_expr(&a.left, ops, None)?;
        }
        Expr::Field(f) => walk_expr(&f.base, ops, None)?,
        Expr::Index(i) => {
            walk_expr(&i.expr, ops, None)?;
            walk_expr(&i.index, ops, None)?;
        }
        Expr::Array(a) => {
            for elem in &a.elems {
                walk_expr(elem, ops, None)?;
            }
        }
        Expr::Tuple(t) => {
            for elem in &t.elems {
                walk_expr(elem, ops, None)?;
            }
        }
        Expr::Group(g) => walk_expr(&g.expr, ops, None)?,
        Expr::Paren(p) => walk_expr(&p.expr, ops, None)?,
        Expr::Match(m) => {
            walk_expr(&m.expr, ops, None)?;
            for arm in &m.arms {
                if let Some((_, guard)) = &arm.guard {
                    walk_expr(guard, ops, None)?;
                }
                walk_expr(&arm.body, ops, None)?;
            }
        }
        Expr::Return(r) => {
            if let Some(v) = &r.expr {
                walk_expr(v, ops, None)?;
            }
        }
        _ => {}
    }
    Ok(())
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
                let a: Tensor<F32, S2<8, 8>> = rmsnorm(x, w);
                let b: Tensor<F32, S2<8, 8>> = linear(a, w);
                b
            }
        };
        let ops = collect_ops(&f).unwrap();
        assert_eq!(
            ops.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
            vec!["rmsnorm", "linear"]
        );
    }

    #[test]
    fn rejects_nested_op_calls() {
        let f: ItemFn = parse_quote! {
            fn block(x: Tensor<F32, S2<8, 8>>, w: Tensor<F32, S2<8, 8>>) -> Tensor<F32, S2<8, 8>> {
                linear(rmsnorm(x, w), w)
            }
        };
        assert!(collect_ops(&f).is_err());
    }

    #[test]
    fn tail_op_without_let() {
        let f: ItemFn = parse_quote! {
            fn block(x: Tensor<F32, S2<8, 8>>, w: Tensor<F32, S2<8, 8>>) -> Tensor<F32, S2<8, 8>> {
                rmsnorm(x, w)
            }
        };
        let ops = collect_ops(&f).unwrap();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0].dest, OpDest::Tail));
    }

    #[test]
    fn grad_gate_chain() {
        let f: ItemFn = parse_quote! {
            fn gate(
                x: Tensor<F32, S2<512, 768>>,
                w: Tensor<F32, S2<768, 768>>,
                w2: Tensor<F32, S2<768, 768>>,
            ) -> Tensor<F32, S2<512, 768>> {
                let h: Tensor<F32, S2<512, 768>> = linear(x, w);
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
