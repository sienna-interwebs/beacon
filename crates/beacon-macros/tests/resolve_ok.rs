use beacon_core::{F32, S2, Tensor};
use beacon_macros::differentiable;

fn rmsnorm(x: Tensor<F32, S2<4, 4>>, _w: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
    x
}

#[differentiable]
fn gate(x: Tensor<F32, S2<4, 4>>, w: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
    rmsnorm(x, w)
}

#[test]
fn registered_op_expands() {
    let x = Tensor::<F32, S2<4, 4>>::from_offset(beacon_core::Region::Activation, 0);
    let w = Tensor::<F32, S2<4, 4>>::from_offset(beacon_core::Region::Activation, 256);
    let _ = gate(x, w);
}
