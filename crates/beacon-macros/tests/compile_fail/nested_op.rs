use beacon_core::{F32, S2, Tensor};
use beacon_macros::differentiable;

fn rmsnorm(x: Tensor<F32, S2<8, 8>>, _w: Tensor<F32, S2<8, 8>>) -> Tensor<F32, S2<8, 8>> {
    x
}

fn linear(x: Tensor<F32, S2<8, 8>>, _w: Tensor<F32, S2<8, 8>>) -> Tensor<F32, S2<8, 8>> {
    x
}

#[differentiable]
fn bad(x: Tensor<F32, S2<8, 8>>, w: Tensor<F32, S2<8, 8>>) -> Tensor<F32, S2<8, 8>> {
    linear(rmsnorm(x, w), w)
}
