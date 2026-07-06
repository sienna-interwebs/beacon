use beacon_core::{F32, S2, Tensor};
use beacon_macros::differentiable;

fn rmsnorm(x: Tensor<F32, S2<4, 4>>, _w: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
    x
}

#[differentiable]
fn block(x: Tensor<F32, S2<4, 4>>, w: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
    rmsnorm(x, w)
}

fn main() {}