use syn::{
    Expr, FnArg, GenericArgument, Ident, ItemFn, Pat, PatType, ReturnType, Type, TypePath,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Lit(usize),
    Const(Ident),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeSig {
    pub rank: usize,
    pub dims: Vec<Dim>,
}

impl ShapeSig {
    pub fn numel_if_const(&self) -> Option<usize> {
        let mut n = 1usize;
        for d in &self.dims {
            n = n.checked_mul(match d {
                Dim::Lit(v) => *v,
                Dim::Const(_) => return None,
            })?;
        }
        Some(n)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSig {
    pub dtype: Ident,
    pub shape: ShapeSig,
}

impl TensorSig {
    pub fn numel_if_const(&self) -> Option<usize> {
        self.shape.numel_if_const()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamSig {
    pub name: Ident,
    pub tensor: TensorSig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSig {
    pub name: Ident,
    pub params: Vec<ParamSig>,
    pub output: TensorSig,
}

pub fn extract(func: &ItemFn) -> syn::Result<FnSig> {
    let name = func.sig.ident.clone();
    let mut params = Vec::new();
    for arg in &func.sig.inputs {
        let FnArg::Typed(PatType { pat, ty, .. }) = arg else {
            continue;
        };
        let name = pat_to_ident(pat)?;
        params.push(ParamSig {
            name,
            tensor: parse_tensor(ty)?,
        });
    }
    let output = match &func.sig.output {
        ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                &func.sig.ident,
                "#[differentiable] functions must return Tensor<T, S>",
            ));
        }
        ReturnType::Type(_, ty) => parse_tensor(ty)?,
    };
    Ok(FnSig {
        name,
        params,
        output,
    })
}

fn pat_to_ident(pat: &Pat) -> syn::Result<Ident> {
    match pat {
        Pat::Ident(p) => Ok(p.ident.clone()),
        _ => Err(syn::Error::new_spanned(
            pat,
            "#[differentiable] parameters must be simple identifiers",
        )),
    }
}

fn parse_tensor(ty: &Type) -> syn::Result<TensorSig> {
    let Type::Path(TypePath { path, .. }) = ty else {
        return Err(syn::Error::new_spanned(ty, "expected Tensor<T, S> type"));
    };
    let segment = path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new_spanned(ty, "expected Tensor<T, S> type"))?;
    if segment.ident != "Tensor" {
        return Err(syn::Error::new_spanned(
            &segment.ident,
            "expected Tensor<T, S> type",
        ));
    }
    let args = match &segment.arguments {
        syn::PathArguments::AngleBracketed(a) => &a.args,
        _ => {
            return Err(syn::Error::new_spanned(
                ty,
                "Tensor requires two generic arguments: Tensor<T, S>",
            ));
        }
    };
    if args.len() != 2 {
        return Err(syn::Error::new_spanned(
            ty,
            "Tensor requires exactly two generic arguments: Tensor<T, S>",
        ));
    }
    let (dtype, shape_ty) = match (&args[0], &args[1]) {
        (GenericArgument::Type(Type::Path(tp)), GenericArgument::Type(shape_ty)) => {
            let dtype = tp
                .path
                .get_ident()
                .cloned()
                .ok_or_else(|| syn::Error::new_spanned(&args[0], "dtype must be a marker type"))?;
            (dtype, shape_ty)
        }
        _ => {
            return Err(syn::Error::new_spanned(
                ty,
                "Tensor generic arguments must be types: Tensor<T, S>",
            ));
        }
    };
    Ok(TensorSig {
        dtype,
        shape: parse_shape(shape_ty)?,
    })
}

fn parse_shape(ty: &Type) -> syn::Result<ShapeSig> {
    let Type::Path(TypePath { path, .. }) = ty else {
        return Err(syn::Error::new_spanned(ty, "shape must be S1/S2/S3/S4<...>"));
    };
    let segment = path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new_spanned(ty, "shape must be S1/S2/S3/S4<...>"))?;
    let rank = match segment.ident.to_string().as_str() {
        "S1" => 1,
        "S2" => 2,
        "S3" => 3,
        "S4" => 4,
        other => {
            return Err(syn::Error::new_spanned(
                &segment.ident,
                format!("unknown shape {other}; expected S1, S2, S3, or S4"),
            ));
        }
    };
    let args = match &segment.arguments {
        syn::PathArguments::AngleBracketed(a) => &a.args,
        _ => {
            return Err(syn::Error::new_spanned(
                ty,
                "shape must include const dimension parameters",
            ));
        }
    };
    if args.len() != rank {
        return Err(syn::Error::new_spanned(
            ty,
            format!("{rank}-rank shape requires {rank} dimension parameters"),
        ));
    }
    let mut dims = Vec::with_capacity(rank);
    for arg in args {
        dims.push(parse_dim(arg)?);
    }
    Ok(ShapeSig { rank, dims })
}

fn parse_dim(arg: &GenericArgument) -> syn::Result<Dim> {
    match arg {
        GenericArgument::Const(expr) => parse_dim_expr(expr),
        GenericArgument::Type(Type::Path(tp)) => {
            let ident = tp.path.get_ident().cloned().ok_or_else(|| {
                syn::Error::new_spanned(arg, "dimension must be a usize literal or const ident")
            })?;
            Ok(Dim::Const(ident))
        }
        _ => Err(syn::Error::new_spanned(
            arg,
            "dimension must be a usize literal or const ident",
        )),
    }
}

fn parse_dim_expr(expr: &Expr) -> syn::Result<Dim> {
    match expr {
        Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(n),
            ..
        }) => Ok(Dim::Lit(n.base10_parse()?)),
        Expr::Path(p) => {
            let ident = p.path.get_ident().cloned().ok_or_else(|| {
                syn::Error::new_spanned(expr, "dimension must be a usize literal or const ident")
            })?;
            Ok(Dim::Const(ident))
        }
        _ => Err(syn::Error::new_spanned(
            expr,
            "dimension must be a usize literal or const ident",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn extracts_literal_shapes() {
        let f: ItemFn = parse_quote! {
            fn gate(
                x: Tensor<F32, S2<512, 768>>,
                w: Tensor<Fp8E4M3, S2<768, 768>>,
            ) -> Tensor<F32, S2<512, 768>> {
                x
            }
        };
        let sig = extract(&f).unwrap();
        assert_eq!(sig.name, "gate");
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0].name, "x");
        assert_eq!(sig.params[0].tensor.dtype, "F32");
        assert_eq!(
            sig.params[0].tensor.shape.dims,
            vec![Dim::Lit(512), Dim::Lit(768)]
        );
        assert_eq!(sig.params[0].tensor.numel_if_const(), Some(512 * 768));
        assert_eq!(sig.params[1].tensor.dtype, "Fp8E4M3");
        assert_eq!(sig.output.numel_if_const(), Some(512 * 768));
    }

    #[test]
    fn extracts_const_generic_dims() {
        let f: ItemFn = parse_quote! {
            fn block(x: Tensor<Bf16, S3<B, S, D>>) -> Tensor<Bf16, S3<B, S, D>> {
                x
            }
        };
        let sig = extract(&f).unwrap();
        let names: Vec<String> = sig.params[0]
            .tensor
            .shape
            .dims
            .iter()
            .map(|d| match d {
                Dim::Const(i) => i.to_string(),
                Dim::Lit(n) => n.to_string(),
            })
            .collect();
        assert_eq!(names, vec!["B", "S", "D"]);
        assert_eq!(sig.params[0].tensor.numel_if_const(), None);
    }

    #[test]
    fn extracts_s4() {
        let f: ItemFn = parse_quote! {
            fn attn(q: Tensor<F32, S4<4, 12, 512, 64>>) -> Tensor<F32, S4<4, 12, 512, 64>> {
                q
            }
        };
        let sig = extract(&f).unwrap();
        assert_eq!(sig.params[0].tensor.shape.rank, 4);
        assert_eq!(sig.params[0].tensor.numel_if_const(), Some(4 * 12 * 512 * 64));
    }

    #[test]
    fn rejects_non_tensor_param() {
        let f: ItemFn = parse_quote! {
            fn bad(x: u32) -> Tensor<F32, S1<8>> {
                x
            }
        };
        assert!(extract(&f).is_err());
    }

    #[test]
    fn rejects_wrong_shape_rank() {
        let f: ItemFn = parse_quote! {
            fn bad(x: Tensor<F32, S2<512>>) -> Tensor<F32, S2<512, 768>> {
                x
            }
        };
        assert!(extract(&f).is_err());
    }
}
