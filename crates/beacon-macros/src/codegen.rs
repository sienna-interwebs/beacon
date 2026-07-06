use beacon_adjoint_table::SaveMode;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{Expr, Ident, ItemFn, Stmt, Type};

use crate::ops::OpDest;
use crate::resolve::ResolvedOpCall;
use crate::signature::{parse_tensor, parse_tensor_type, tensor_parts, FnSig, ShapeSig};

#[derive(Clone)]
struct SavedField {
    name: Ident,
    dtype: Ident,
    shape: Type,
}

#[derive(Clone)]
struct ActCheckpoint {
    field: Ident,
    ident: Ident,
    ty: Type,
}

struct OpStep {
    name: String,
    out_ident: Ident,
    out_ty: Type,
    args: Vec<Expr>,
    arg_tys: Vec<Type>,
    saved: Vec<SavedField>,
    tokens: TokenStream2,
    dim: TokenStream2,
    checkpoint: Option<ActCheckpoint>,
}

pub fn generate(
    func: &ItemFn,
    sig: &FnSig,
    resolved: &[ResolvedOpCall],
) -> syn::Result<TokenStream2> {
    let saved_name = format_ident!("{}Saved", sig.name);
    let grad_buf_name = format_ident!("{}_GRAD_BUF_BYTES", sig.name.to_string().to_uppercase());
    let fwd_name = format_ident!("{}_forward", sig.name);
    let bwd_name = format_ident!("{}_backward", sig.name);

    let mut env: HashMap<String, Type> = HashMap::new();
    for p in &sig.params {
        env.insert(p.name.to_string(), param_ty(func, &p.name)?);
    }

    let output_ty = return_ty(func)?;
    let mut all_saved = Vec::new();
    let mut fwd_body = Vec::new();
    let mut steps = Vec::new();
    let mut final_out = None::<Ident>;
    let mut final_ty = output_ty.clone();

    if resolved.is_empty() {
        if let Some(expr) = tail_expr(func) {
            if let Expr::Path(p) = &expr {
                if let Some(name) = p.path.get_ident() {
                    final_out = Some(name.clone());
                }
            }
        }
    }

    for (idx, r) in resolved.iter().enumerate() {
        let site = &r.site;
        let out_ty = match &site.dest {
            OpDest::Let { ty, .. } => ty.clone(),
            OpDest::Tail => output_ty.clone(),
        };
        parse_tensor_type(&out_ty)?;

        let out_ident = match &site.dest {
            OpDest::Let { ident, .. } => ident.clone(),
            OpDest::Tail => format_ident!("__out{}", idx),
        };

        let arg_tys: Vec<Type> = site
            .args
            .iter()
            .map(|a| ty_of_expr(a, &env))
            .collect::<syn::Result<_>>()?;

        let mut step_saved = Vec::new();
        for spec in r.entry.saved {
            if spec.mode != SaveMode::Store {
                continue;
            }
            let (dtype, shape) = saved_field_types(spec.name, &arg_tys)?;
            step_saved.push(SavedField {
                name: format_ident!("saved_{}_{}", idx, spec.name),
                dtype,
                shape,
            });
        }
        all_saved.extend(step_saved.clone());

        let (tokens, dim) = shape_tokens(&out_ty)?;
        let (out_dtype, out_shape) = tensor_parts(&out_ty)?;
        let out_alloc = quote! {
            let #out_ident: #out_ty = arena.alloc::<#out_dtype, #out_shape>(beacon_core::Region::Activation);
        };
        let saved_allocs = step_saved.iter().map(|s| {
            let n = &s.name;
            let d = &s.dtype;
            let sh = &s.shape;
            quote! { let #n = arena.alloc_saved::<#d, #sh>(); }
        });
        let checkpoint = match &site.dest {
            OpDest::Let { ident, ty } => Some(ActCheckpoint {
                field: format_ident!("act_{}", idx),
                ident: ident.clone(),
                ty: ty.clone(),
            }),
            OpDest::Tail => None,
        };

        let fwd_call = fwd_call(&site.name, &out_ident, &out_ty, &site.args, &arg_tys, &step_saved, &tokens, &dim)?;
        fwd_body.push(quote! {
            #out_alloc
            #(#saved_allocs)*
            #fwd_call?;
        });

        steps.push(OpStep {
            name: site.name.clone(),
            out_ident,
            out_ty,
            args: site.args.clone(),
            arg_tys,
            saved: step_saved,
            tokens,
            dim,
            checkpoint,
        });

        env.insert(steps.last().unwrap().out_ident.to_string(), steps.last().unwrap().out_ty.clone());
        final_out = Some(steps.last().unwrap().out_ident.clone());
        final_ty = steps.last().unwrap().out_ty.clone();
    }

    let final_out = match final_out {
        Some(v) => v,
        None if resolved.is_empty() => {
            return Err(syn::Error::new_spanned(
                func,
                "differentiable functions need at least one op or a passthrough identifier tail expression",
            ));
        }
        None => format_ident!("__missing_out"),
    };
    let saved_struct_fields = all_saved.iter().map(|s| {
        let n = &s.name;
        let d = &s.dtype;
        let sh = &s.shape;
        quote! { pub #n: beacon_core::Saved<#d, #sh> }
    });
    let checkpoints: Vec<_> = steps.iter().filter_map(|s| s.checkpoint.clone()).collect();
    let act_fields = checkpoints.iter().map(|c| {
        let f = &c.field;
        quote! { pub #f: usize }
    });
    let act_inits = checkpoints.iter().map(|c| {
        let f = &c.field;
        let id = &c.ident;
        quote! { #f: #id.offset }
    });
    let act_bindings = checkpoints.iter().map(|c| {
        let id = &c.ident;
        let ty = &c.ty;
        let f = &c.field;
        quote! {
            let #id: #ty = beacon_core::Tensor::from_offset(beacon_core::Region::Activation, saved.#f);
        }
    });

    let saved_init = all_saved.iter().map(|s| {
        let n = &s.name;
        quote! { #n }
    });
    let saved_init = saved_init.chain(act_inits);
    let grad_terms = all_saved.iter().map(|s| {
        let d = &s.dtype;
        let sh = &s.shape;
        quote! { + <beacon_core::Saved<#d, #sh>>::BYTES }
    });

    let param_decls: Vec<_> = func.sig.inputs.iter().filter_map(|arg| {
        let syn::FnArg::Typed(t) = arg else { return None };
        let pat = &t.pat;
        let ty = &t.ty;
        Some(quote! { #pat: #ty })
    }).collect();

    let traits = required_traits(resolved);
    let mut bwd_body = Vec::new();
    for step_idx in (0..steps.len()).rev() {
        let step = &steps[step_idx];
        let upstream: Expr = if step_idx == steps.len() - 1 {
            syn::parse_quote!(dy)
        } else {
            let id = &step.out_ident;
            syn::parse_quote!(#id)
        };
        bwd_body.push(bwd_call(
            &step.name,
            &step.out_ty,
            &step.args,
            &step.arg_tys,
            &step.saved,
            &step.tokens,
            &step.dim,
            &upstream,
        )?);
    }

    Ok(quote! {
        pub const #grad_buf_name: usize = 0usize #( #grad_terms )*;

        pub struct #saved_name {
            #( #saved_struct_fields, )*
            #( #act_fields ),*
        }

        pub fn #fwd_name<L>(launcher: &L, arena: &mut beacon_core::Arena, #( #param_decls ),*) -> beacon_cuda::LaunchResult<(#final_ty, #saved_name)>
        where
            L: beacon_cuda::KernelLauncher #( + #traits )*,
        {
            arena.reset_activations();
            #(#fwd_body)*
            Ok((#final_out, #saved_name { #( #saved_init ),* }))
        }

        pub fn #bwd_name<L>(launcher: &L, _arena: &mut beacon_core::Arena, mut saved: #saved_name, mut dy: #final_ty, #( #param_decls ),*) -> beacon_cuda::LaunchResult<()>
        where
            L: beacon_cuda::KernelLauncher #( + #traits )*,
        {
            #(#act_bindings)*
            #(#bwd_body)*
            Ok(())
        }

        #func
    })
}

fn param_ty(func: &ItemFn, name: &Ident) -> syn::Result<Type> {
    for arg in &func.sig.inputs {
        let syn::FnArg::Typed(t) = arg else { continue };
        if let syn::Pat::Ident(p) = t.pat.as_ref() {
            if p.ident == *name {
                return Ok((*t.ty).clone());
            }
        }
    }
    Err(syn::Error::new_spanned(name, "missing parameter type"))
}

fn return_ty(func: &ItemFn) -> syn::Result<Type> {
    let syn::ReturnType::Type(_, ty) = &func.sig.output else {
        return Err(syn::Error::new_spanned(func, "missing return type"));
    };
    Ok(*ty.clone())
}

fn tail_expr(func: &ItemFn) -> Option<Expr> {
    match func.block.stmts.last()? {
        Stmt::Expr(expr, _) => Some(expr.clone()),
        _ => None,
    }
}

fn ty_of_expr(expr: &Expr, env: &HashMap<String, Type>) -> syn::Result<Type> {
    let Expr::Path(p) = expr else {
        return Err(syn::Error::new_spanned(
            expr,
            "op arguments must be identifier paths",
        ));
    };
    let name = p
        .path
        .get_ident()
        .ok_or_else(|| syn::Error::new_spanned(expr, "op arguments must be simple identifiers"))?;
    env.get(&name.to_string())
        .cloned()
        .ok_or_else(|| syn::Error::new_spanned(expr, format!("unknown tensor `{name}`")))
}

fn saved_field_types(spec: &str, arg_tys: &[Type]) -> syn::Result<(Ident, Type)> {
    match spec {
        "rms" | "mean" | "invstd" | "lse" => {
            let input = arg_tys.first().ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("saved spec `{spec}` needs argument index 0"),
                )
            })?;
            let (dtype, _) = tensor_parts(input)?;
            Ok((dtype, stats_ty(input)?))
        }
        _ => {
            let ty = saved_spec_ty(spec, arg_tys)?;
            tensor_parts(&ty)
        }
    }
}

fn saved_spec_ty(spec: &str, arg_tys: &[Type]) -> syn::Result<Type> {
    let idx = match spec {
        "x" | "lhs" | "q" | "rms" | "mean" | "invstd" | "lse" => 0,
        "weight" | "rhs" | "k" => 1,
        "v" | "bias" => 2,
        other => {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("unknown saved spec `{other}`"),
            ));
        }
    };
    let input = arg_tys.get(idx).ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("saved spec `{spec}` needs argument index {idx}"),
        )
    })?;
    match spec {
        "rms" | "mean" | "invstd" | "lse" => stats_ty(input),
        _ => Ok(input.clone()),
    }
}

fn stats_ty(input: &Type) -> syn::Result<Type> {
    let sig = parse_tensor(input)?;
    let dim = match sig.shape {
        ShapeSig { rank: 2, ref dims, .. } => dims.first(),
        ShapeSig { rank: 3, ref dims, .. } => dims.get(1),
        _ => None,
    }
    .ok_or_else(|| syn::Error::new_spanned(input, "stats saved type needs rank 2 or 3 input"))?;
    match dim {
        crate::signature::Dim::Lit(n) => Ok(syn::parse_quote!(beacon_core::S1<#n>)),
        crate::signature::Dim::Const(c) => Ok(syn::parse_quote!(beacon_core::S1<{ #c }>)),
    }
}

fn shape_tokens(tensor_ty: &Type) -> syn::Result<(TokenStream2, TokenStream2)> {
    let sig = parse_tensor(tensor_ty)?;
    if sig.shape.rank != 2 {
        return Err(syn::Error::new_spanned(
            tensor_ty,
            "launcher dims require rank-2 tensors",
        ));
    }
    Ok((dim_token(&sig.shape.dims[0])?, dim_token(&sig.shape.dims[1])?))
}

fn dim_token(dim: &crate::signature::Dim) -> syn::Result<TokenStream2> {
    match dim {
        crate::signature::Dim::Lit(n) => Ok(quote! { #n }),
        crate::signature::Dim::Const(c) => Ok(quote! { #c }),
    }
}

fn kernel_read(expr: &Expr, ty: &Type) -> TokenStream2 {
    quote! { beacon_cuda::KernelArg::read(#expr.offset, <#ty>::NBYTES) }
}

fn kernel_write(name: &Ident, ty: &Type) -> TokenStream2 {
    quote! { beacon_cuda::KernelArg::write(#name.offset, <#ty>::NBYTES) }
}

fn kernel_rw(expr: &Expr, ty: &Type) -> TokenStream2 {
    quote! { beacon_cuda::KernelArg::read_write(#expr.offset, <#ty>::NBYTES) }
}

fn kernel_saved(saved: &Ident, dtype: &Ident, shape: &Type) -> TokenStream2 {
    quote! { beacon_cuda::KernelArg::write(#saved.offset(), <beacon_core::Saved<#dtype, #shape>>::BYTES) }
}

fn fwd_call(
    op: &str,
    out: &Ident,
    out_ty: &Type,
    args: &[Expr],
    arg_tys: &[Type],
    saved: &[SavedField],
    tokens: &TokenStream2,
    dim: &TokenStream2,
) -> syn::Result<TokenStream2> {
    match op {
        "rmsnorm" => {
            let (x, w) = (&args[0], &args[1]);
            let (xt, wt) = (&arg_tys[0], &arg_tys[1]);
            let sx = saved.iter().find(|s| s.name.to_string().ends_with("_rms")).unwrap();
            let a_out = kernel_write(out, out_ty);
            let a_x = kernel_read(x, xt);
            let a_w = kernel_read(w, wt);
            let a_rms = kernel_saved(&sx.name, &sx.dtype, &sx.shape);
            Ok(quote! {
                beacon_cuda::RmsNormLaunch::rmsnorm_fwd(
                    launcher,
                    #a_out,
                    #a_x,
                    #a_w,
                    #a_rms,
                    #tokens,
                    #dim,
                )
            })
        }
        "linear" => {
            let (x, w) = (&args[0], &args[1]);
            let (xt, wt) = (&arg_tys[0], &arg_tys[1]);
            let a_out = kernel_write(out, out_ty);
            let a_x = kernel_read(x, xt);
            let a_w = kernel_read(w, wt);
            Ok(quote! {
                beacon_cuda::MatmulLaunch::linear_fwd(
                    launcher,
                    #a_out,
                    #a_x,
                    #a_w,
                    beacon_cuda::KernelArg::read(0, 0),
                    #tokens,
                    #dim,
                    #dim,
                )
            })
        }
        "residual_add" => {
            let (a, b) = (&args[0], &args[1]);
            let (at, bt) = (&arg_tys[0], &arg_tys[1]);
            let n = quote! { <#out_ty>::NUMEL };
            let a_out = kernel_write(out, out_ty);
            let a_a = kernel_read(a, at);
            let a_b = kernel_read(b, bt);
            Ok(quote! {
                beacon_cuda::ElementwiseLaunch::residual_add(
                    launcher,
                    #a_out,
                    #a_a,
                    #a_b,
                    #n,
                )
            })
        }
        other => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("forward codegen not implemented for op `{other}`"),
        )),
    }
}

fn bwd_call(
    op: &str,
    out_ty: &Type,
    args: &[Expr],
    arg_tys: &[Type],
    saved: &[SavedField],
    tokens: &TokenStream2,
    dim: &TokenStream2,
    upstream: &Expr,
) -> syn::Result<TokenStream2> {
    match op {
        "rmsnorm" => {
            let (x, w) = (&args[0], &args[1]);
            let (xt, wt) = (&arg_tys[0], &arg_tys[1]);
            let sx = saved.iter().find(|s| s.name.to_string().ends_with("_x")).unwrap();
            let sr = saved.iter().find(|s| s.name.to_string().ends_with("_rms")).unwrap();
            let sxn = &sx.name;
            let srn = &sr.name;
            let srty = &sr.shape;
            let srd = &sr.dtype;
            let a_dx = kernel_rw(x, xt);
            let a_dy = kernel_read(upstream, out_ty);
            let a_x = kernel_read(x, xt);
            let a_w = kernel_read(w, wt);
            Ok(quote! {
                {
                    let _x_off = saved.#sxn.consume();
                    let rms_off = saved.#srn.consume();
                    beacon_cuda::RmsNormLaunch::rmsnorm_bwd(
                        launcher,
                        #a_dx,
                        beacon_cuda::KernelArg::read_write(0, 0),
                        #a_dy,
                        #a_x,
                        beacon_cuda::KernelArg::read(rms_off, <beacon_core::Saved<#srd, #srty>>::BYTES),
                        #a_w,
                        #tokens,
                        #dim,
                    )?;
                }
            })
        }
        "linear" => {
            let (x, w) = (&args[0], &args[1]);
            let (xt, wt) = (&arg_tys[0], &arg_tys[1]);
            let sx = saved.iter().find(|s| s.name.to_string().ends_with("_x")).unwrap();
            let sxn = &sx.name;
            let a_dx = kernel_rw(x, xt);
            let a_dy = kernel_read(upstream, out_ty);
            let a_x = kernel_read(x, xt);
            let a_w = kernel_read(w, wt);
            Ok(quote! {
                {
                    let _x_off = saved.#sxn.consume();
                    beacon_cuda::MatmulLaunch::linear_bwd(
                        launcher,
                        #a_dx,
                        beacon_cuda::KernelArg::read_write(0, 0),
                        beacon_cuda::KernelArg::read_write(0, 0),
                        #a_dy,
                        #a_x,
                        #a_w,
                        #tokens,
                        #dim,
                        #dim,
                    )?;
                }
            })
        }
        other => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("backward codegen not implemented for op `{other}`"),
        )),
    }
}

fn required_traits(resolved: &[ResolvedOpCall]) -> Vec<TokenStream2> {
    let mut out = Vec::new();
    for r in resolved {
        let t = match r.site.name.as_str() {
            "rmsnorm" => quote! { beacon_cuda::RmsNormLaunch },
            "layernorm" => quote! { beacon_cuda::LayerNormLaunch },
            "linear" | "matmul" => quote! { beacon_cuda::MatmulLaunch },
            "residual_add" => quote! { beacon_cuda::ElementwiseLaunch },
            "swiglu_mlp" => quote! { beacon_cuda::SwigluLaunch },
            "flash_attention" => quote! { beacon_cuda::AttentionLaunch },
            _ => continue,
        };
        let s = t.to_string();
        if !out.iter().any(|existing: &TokenStream2| existing.to_string() == s) {
            out.push(t);
        }
    }
    out
}
