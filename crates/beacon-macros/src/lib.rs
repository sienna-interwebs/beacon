use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

mod ops;
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
    let _ops = match ops::collect_ops(&func) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };
    quote!(#func).into()
}
