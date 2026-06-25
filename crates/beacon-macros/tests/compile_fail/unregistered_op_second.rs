use beacon_core::{F32, S2, Tensor};
use beacon_macros::differentiable;

#[differentiable]
fn bad(x: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
    let a = rmsnorm(x, x);
    unknown_op(a)
}
