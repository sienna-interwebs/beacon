use beacon_core::{F32, S2, Saved};

fn bad() {
    let s = Saved::<F32, S2<4, 4>>::at(0);
    let _ = s.consume();
    let _ = s.consume();
}
