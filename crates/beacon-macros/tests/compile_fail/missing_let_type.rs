use beacon_core::{F32, S2, Tensor};
use beacon_macros::differentiable;

fn rmsnorm(x: Tensor<F32, S2<4, 4>>, _w: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
    x
}

#[differentiable]
fn bad(x: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
    let a = rmsnorm(x, x);
    a
}
