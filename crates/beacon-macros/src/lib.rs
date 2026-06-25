use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

mod ops;
mod resolve;
mod signature;
mod validate;

#[proc_macro_attribute]
pub fn differentiable(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    if let Err(e) = validate::check_signature(&func) {
        return e.to_compile_error().into();
    }
    if let Err(e) = signature::extract(&func) {
        return e.to_compile_error().into();
    }
    let ops = match ops::collect_ops(&func) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };
    let _resolved = match resolve::resolve_ops(&ops) {
        Ok(v) => v,
        Err(errs) => return join_errors(errs).into(),
    };
    quote!(#func).into()
}

fn join_errors(errs: Vec<syn::Error>) -> TokenStream2 {
    errs.into_iter()
        .map(|e| e.to_compile_error())
        .collect()
}
