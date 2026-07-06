use beacon_core::{F32, S2, Tensor};
use beacon_macros::differentiable;

fn linear(x: Tensor<F32, S2<4, 4>>, _w: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
    x
}

fn rmsnorm(x: Tensor<F32, S2<4, 4>>, _w: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
    x
}

#[differentiable]
fn gate(
    x: Tensor<F32, S2<4, 4>>,
    w: Tensor<F32, S2<4, 4>>,
    w2: Tensor<F32, S2<4, 4>>,
) -> Tensor<F32, S2<4, 4>> {
    let h: Tensor<F32, S2<4, 4>> = linear(x, w);
    rmsnorm(h, w2)
}

fn main() {}