use beacon_macros::differentiable;

#[differentiable]
fn ident(x: u32) -> u32 {
    x
}

#[differentiable]
fn add(x: u32, y: u32) -> u32 {
    x + y
}

#[test]
fn passthrough_preserves_behavior() {
    assert_eq!(ident(5), 5);
    assert_eq!(add(2, 3), 5);
}
