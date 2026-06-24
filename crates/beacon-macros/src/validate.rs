use syn::{FnArg, ItemFn, ReturnType, Type};

pub fn check_signature(func: &ItemFn) -> syn::Result<()> {
    let sig = &func.sig;

    if let Some(a) = &sig.asyncness {
        return Err(syn::Error::new_spanned(
            a,
            "#[differentiable] functions cannot be async",
        ));
    }
    if let Some(c) = &sig.constness {
        return Err(syn::Error::new_spanned(
            c,
            "#[differentiable] functions cannot be const",
        ));
    }
    if let Some(v) = &sig.variadic {
        return Err(syn::Error::new_spanned(
            v,
            "#[differentiable] functions cannot be variadic",
        ));
    }
    if sig.inputs.is_empty() {
        return Err(syn::Error::new_spanned(
            &sig.ident,
            "#[differentiable] functions must take at least one tensor input",
        ));
    }
    for arg in &sig.inputs {
        if let FnArg::Receiver(r) = arg {
            return Err(syn::Error::new_spanned(
                r,
                "#[differentiable] functions must be free functions (no `self`)",
            ));
        }
    }
    match &sig.output {
        ReturnType::Default => Err(syn::Error::new_spanned(
            &sig.ident,
            "#[differentiable] functions must return a tensor",
        )),
        ReturnType::Type(_, ty) => {
            if is_unit(ty) {
                Err(syn::Error::new_spanned(
                    ty,
                    "#[differentiable] functions must return a tensor, not ()",
                ))
            } else {
                Ok(())
            }
        }
    }
}

fn is_unit(ty: &Type) -> bool {
    matches!(ty, Type::Tuple(t) if t.elems.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn accepts_typed_fn_with_return() {
        let f: ItemFn = parse_quote!(
            fn block(x: Tensor<F32, S2<8, 8>>, w: Tensor<F32, S2<8, 8>>) -> Tensor<F32, S2<8, 8>> {
                x
            }
        );
        assert!(check_signature(&f).is_ok());
    }

    #[test]
    fn rejects_async() {
        let f: ItemFn = parse_quote!(
            async fn block(x: u32) -> u32 {
                x
            }
        );
        assert!(check_signature(&f).is_err());
    }

    #[test]
    fn rejects_const() {
        let f: ItemFn = parse_quote!(
            const fn block(x: u32) -> u32 {
                x
            }
        );
        assert!(check_signature(&f).is_err());
    }

    #[test]
    fn rejects_no_inputs() {
        let f: ItemFn = parse_quote!(
            fn block() -> u32 {
                0
            }
        );
        assert!(check_signature(&f).is_err());
    }

    #[test]
    fn rejects_unit_return() {
        let f: ItemFn = parse_quote!(
            fn block(x: u32) {
                let _ = x;
            }
        );
        assert!(check_signature(&f).is_err());
    }

    #[test]
    fn rejects_explicit_unit_return() {
        let f: ItemFn = parse_quote!(
            fn block(x: u32) -> () {
                let _ = x;
            }
        );
        assert!(check_signature(&f).is_err());
    }

    #[test]
    fn rejects_receiver() {
        let f: ItemFn = parse_quote!(
            fn block(&self, x: u32) -> u32 {
                x
            }
        );
        assert!(check_signature(&f).is_err());
    }
}
