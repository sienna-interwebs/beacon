use beacon_core::{F32, S2, Tensor};
use beacon_macros::differentiable;

#[differentiable]
fn ident(x: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
    x
}

#[differentiable]
fn pair(
    x: Tensor<F32, S2<8, 8>>,
    y: Tensor<F32, S2<8, 8>>,
) -> Tensor<F32, S2<8, 8>> {
    x
}

#[test]
fn passthrough_preserves_types() {
    let t = Tensor::<F32, S2<4, 4>>::from_offset(beacon_core::Region::Activation, 0);
    let _ = ident(t);
    let a = Tensor::<F32, S2<8, 8>>::from_offset(beacon_core::Region::Activation, 256);
    let b = Tensor::<F32, S2<8, 8>>::from_offset(beacon_core::Region::Activation, 512);
    let _ = pair(a, b);
}
