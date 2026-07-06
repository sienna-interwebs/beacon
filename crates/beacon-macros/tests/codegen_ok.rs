use beacon_core::{F32, S2, Tensor};
use beacon_cuda::{LaunchError, Launcher};
use beacon_macros::differentiable;

fn linear(x: Tensor<F32, S2<4, 4>>, w: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
    x
}

fn rmsnorm(x: Tensor<F32, S2<4, 4>>, w: Tensor<F32, S2<4, 4>>) -> Tensor<F32, S2<4, 4>> {
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

#[test]
fn generated_forward_backward_exist() {
    assert!(GATE_GRAD_BUF_BYTES > 0);
    let launcher = Launcher::on_device(0).unwrap();
    let mut arena = beacon_core::Arena::new(beacon_core::ArenaLayout {
        weight_bytes: 1 << 20,
        activation_bytes: 1 << 20,
        gradient_bytes: GATE_GRAD_BUF_BYTES + 4096,
    });
    let x = Tensor::<F32, S2<4, 4>>::from_offset(beacon_core::Region::Weight, 0);
    let w = Tensor::<F32, S2<4, 4>>::from_offset(beacon_core::Region::Weight, 256);
    let w2 = Tensor::<F32, S2<4, 4>>::from_offset(beacon_core::Region::Weight, 512);
    assert!(matches!(
        gate_forward(&launcher, &mut arena, x, w, w2),
        Err(LaunchError::Unimplemented(_))
    ));
}
